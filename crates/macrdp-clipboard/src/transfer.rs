use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use ironrdp_cliprdr::pdu::*;

use crate::file::read_file_range;

#[derive(Debug, Clone, Copy, PartialEq)]
enum RequestKind {
    Size,
    Data,
}

type StreamMapping = (u32, RequestKind);

pub struct FileTransferManager {
    local_files: Vec<PathBuf>,
    incoming_descriptors: Vec<FileDescriptor>,
    incoming_data: HashMap<u32, Vec<u8>>,
    incoming_complete: HashSet<u32>,
    stream_to_index: HashMap<u32, StreamMapping>,
    temp_dir: PathBuf,
    max_file_size: u64,
    locked: bool,
    next_stream_id: u32,
}

impl FileTransferManager {
    pub fn new(temp_dir: PathBuf, max_file_size: u64) -> Self {
        fs::create_dir_all(&temp_dir).ok();
        Self {
            local_files: Vec::new(),
            incoming_descriptors: Vec::new(),
            incoming_data: HashMap::new(),
            incoming_complete: HashSet::new(),
            stream_to_index: HashMap::new(),
            temp_dir,
            max_file_size,
            locked: false,
            next_stream_id: 1,
        }
    }

    pub fn set_local_files(&mut self, paths: Vec<PathBuf>) {
        self.local_files = paths;
    }

    pub fn build_file_list(&self) -> PackedFileList {
        let files = self
            .local_files
            .iter()
            .filter_map(|path| {
                let meta = fs::metadata(path).ok()?;
                let name = path.file_name()?.to_string_lossy().to_string();
                Some(FileDescriptor {
                    name,
                    file_size: Some(meta.len()),
                    attributes: None,
                    last_write_time: None,
                })
            })
            .collect();
        PackedFileList { files }
    }

    pub fn handle_contents_request(
        &mut self,
        request: &FileContentsRequest,
    ) -> FileContentsResponse<'static> {
        let index = request.index;
        let stream_id = request.stream_id;

        let Some(path) = self.local_files.get(index as usize) else {
            tracing::warn!(index, "File contents request for invalid file index");
            return FileContentsResponse::new_error(stream_id);
        };

        if request.flags.contains(FileContentsFlags::SIZE) {
            match fs::metadata(path) {
                Ok(meta) => {
                    if meta.len() > self.max_file_size {
                        tracing::warn!(
                            path = %path.display(),
                            size = meta.len(),
                            max = self.max_file_size,
                            "File exceeds size limit"
                        );
                        return FileContentsResponse::new_error(stream_id);
                    }
                    FileContentsResponse::new_size_response(stream_id, meta.len())
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "Failed to read file metadata");
                    FileContentsResponse::new_error(stream_id)
                }
            }
        } else if request.flags.contains(FileContentsFlags::DATA) {
            match read_file_range(
                path,
                request.position,
                request.requested_size,
                self.max_file_size,
            ) {
                Ok(data) => FileContentsResponse::new_data_response(stream_id, data),
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "Failed to read file range");
                    FileContentsResponse::new_error(stream_id)
                }
            }
        } else {
            tracing::warn!("Unknown FileContentsRequest flags");
            FileContentsResponse::new_error(stream_id)
        }
    }

    pub fn set_incoming_descriptors(&mut self, descriptors: Vec<FileDescriptor>) {
        self.incoming_data.clear();
        self.incoming_complete.clear();
        self.stream_to_index.clear();
        self.incoming_descriptors = descriptors;
    }

    pub fn generate_contents_requests(&mut self) -> Vec<FileContentsRequest> {
        let mut requests = Vec::new();

        for (index, desc) in self.incoming_descriptors.iter().enumerate() {
            let size_stream = self.next_stream_id;
            self.next_stream_id += 1;
            self.stream_to_index
                .insert(size_stream, (index as u32, RequestKind::Size));

            requests.push(FileContentsRequest {
                stream_id: size_stream,
                index: index as u32,
                flags: FileContentsFlags::SIZE,
                position: 0,
                requested_size: 8,
                data_id: None,
            });

            let data_stream = self.next_stream_id;
            self.next_stream_id += 1;
            self.stream_to_index
                .insert(data_stream, (index as u32, RequestKind::Data));

            let size = desc.file_size.unwrap_or(0) as u32;
            requests.push(FileContentsRequest {
                stream_id: data_stream,
                index: index as u32,
                flags: FileContentsFlags::DATA,
                position: 0,
                requested_size: size,
                data_id: None,
            });
        }

        requests
    }

    pub fn handle_contents_response(&mut self, response: &FileContentsResponse<'_>) {
        let stream_id = response.stream_id();
        let Some(&(file_index, kind)) = self.stream_to_index.get(&stream_id) else {
            tracing::warn!(stream_id, "Unknown stream_id in file contents response");
            return;
        };

        if kind == RequestKind::Size {
            return;
        }

        let data = response.data();
        if data.is_empty() {
            tracing::warn!(
                stream_id,
                file_index,
                "Empty data in file contents response"
            );
            return;
        }

        self.incoming_data
            .entry(file_index)
            .or_default()
            .extend_from_slice(data);

        if let Some(desc) = self.incoming_descriptors.get(file_index as usize) {
            let expected = desc.file_size.unwrap_or(0) as usize;
            let received = self.incoming_data.get(&file_index).map_or(0, |d| d.len());
            if received >= expected {
                self.incoming_complete.insert(file_index);
            }
        }
    }

    pub fn all_incoming_complete(&self) -> bool {
        !self.incoming_descriptors.is_empty()
            && self.incoming_complete.len() == self.incoming_descriptors.len()
    }

    pub fn flush_incoming_files(&mut self) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        for (index, desc) in self.incoming_descriptors.iter().enumerate() {
            let index = index as u32;
            if let Some(data) = self.incoming_data.remove(&index) {
                if desc.name.contains("..")
                    || desc.name.starts_with('/')
                    || desc.name.starts_with('\\')
                {
                    tracing::warn!(name = desc.name, "Rejected file with suspicious path");
                    continue;
                }

                let file_path = self.temp_dir.join(&desc.name);
                match fs::write(&file_path, &data) {
                    Ok(()) => {
                        tracing::debug!(path = %file_path.display(), bytes = data.len(), "Wrote incoming file");
                        paths.push(file_path);
                    }
                    Err(e) => {
                        tracing::warn!(path = %file_path.display(), error = %e, "Failed to write incoming file");
                    }
                }
            }
        }

        self.incoming_descriptors.clear();
        self.incoming_complete.clear();
        self.stream_to_index.clear();
        paths
    }

    pub fn lock(&mut self) {
        self.locked = true;
    }

    pub fn unlock(&mut self) {
        self.locked = false;
        self.incoming_data.clear();
        self.incoming_complete.clear();
        self.stream_to_index.clear();
    }

    pub fn clear(&mut self) {
        self.local_files.clear();
        self.incoming_descriptors.clear();
        self.incoming_data.clear();
        self.incoming_complete.clear();
        self.stream_to_index.clear();
    }
}

impl Drop for FileTransferManager {
    fn drop(&mut self) {
        if let Ok(entries) = fs::read_dir(&self.temp_dir) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "macrdp-transfer-test-{}-{}",
            std::process::id(),
            name,
        ));
        fs::create_dir_all(&dir).ok();
        dir
    }

    #[test]
    fn handle_size_request() {
        let dir = temp_dir("size");
        let file_path = dir.join("test.txt");
        fs::write(&file_path, b"hello world").unwrap();

        let mut mgr = FileTransferManager::new(dir.clone(), 1024 * 1024);
        mgr.set_local_files(vec![file_path]);

        let request = FileContentsRequest {
            stream_id: 1,
            index: 0,
            flags: FileContentsFlags::SIZE,
            position: 0,
            requested_size: 8,
            data_id: None,
        };

        let response = mgr.handle_contents_request(&request);
        assert_eq!(response.stream_id(), 1);
        let size = response.data_as_size().unwrap();
        assert_eq!(size, 11);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn handle_data_request() {
        let dir = temp_dir("data");
        let file_path = dir.join("data.bin");
        fs::write(&file_path, b"ABCDEF").unwrap();

        let mut mgr = FileTransferManager::new(dir.clone(), 1024 * 1024);
        mgr.set_local_files(vec![file_path]);

        let request = FileContentsRequest {
            stream_id: 2,
            index: 0,
            flags: FileContentsFlags::DATA,
            position: 2,
            requested_size: 3,
            data_id: None,
        };

        let response = mgr.handle_contents_request(&request);
        assert_eq!(response.data(), b"CDE");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn size_limit_enforcement() {
        let dir = temp_dir("limit");
        let file_path = dir.join("big.bin");
        fs::write(&file_path, vec![0u8; 200]).unwrap();

        let mut mgr = FileTransferManager::new(dir.clone(), 100);
        mgr.set_local_files(vec![file_path]);

        let request = FileContentsRequest {
            stream_id: 3,
            index: 0,
            flags: FileContentsFlags::SIZE,
            position: 0,
            requested_size: 8,
            data_id: None,
        };

        let response = mgr.handle_contents_request(&request);
        assert!(response.data().is_empty());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generate_and_handle_incoming() {
        let dir = temp_dir("incoming");
        let mut mgr = FileTransferManager::new(dir.clone(), 1024 * 1024);

        let descriptors = vec![
            FileDescriptor {
                name: "a.txt".to_string(),
                file_size: Some(5),
                attributes: None,
                last_write_time: None,
            },
            FileDescriptor {
                name: "b.txt".to_string(),
                file_size: Some(3),
                attributes: None,
                last_write_time: None,
            },
        ];
        mgr.set_incoming_descriptors(descriptors);

        let requests = mgr.generate_contents_requests();
        assert_eq!(requests.len(), 4);
        assert!(requests[0].flags.contains(FileContentsFlags::SIZE));
        assert!(requests[1].flags.contains(FileContentsFlags::DATA));

        let data_stream_a = requests[1].stream_id;
        let data_stream_b = requests[3].stream_id;

        let resp_a = FileContentsResponse::new_data_response(data_stream_a, b"hello".to_vec());
        mgr.handle_contents_response(&resp_a);
        assert!(!mgr.all_incoming_complete());

        let resp_b = FileContentsResponse::new_data_response(data_stream_b, b"bye".to_vec());
        mgr.handle_contents_response(&resp_b);
        assert!(mgr.all_incoming_complete());

        let paths = mgr.flush_incoming_files();
        assert_eq!(paths.len(), 2);
        assert_eq!(fs::read_to_string(&paths[0]).unwrap(), "hello");
        assert_eq!(fs::read_to_string(&paths[1]).unwrap(), "bye");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unlock_clears_state() {
        let dir = temp_dir("unlock");
        let mut mgr = FileTransferManager::new(dir.clone(), 1024 * 1024);

        mgr.set_local_files(vec![PathBuf::from("/tmp/fake")]);
        mgr.lock();
        assert!(mgr.locked);
        mgr.unlock();
        assert!(!mgr.locked);
        assert!(mgr.incoming_data.is_empty());

        fs::remove_dir_all(&dir).ok();
    }
}
