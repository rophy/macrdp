pub mod file;
pub mod formats;
pub mod html;
pub mod pasteboard;
pub mod transfer;

use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use ironrdp_cliprdr::backend::{ClipboardMessage, CliprdrBackend, CliprdrBackendFactory};
use ironrdp_cliprdr::pdu::*;
use ironrdp_core::{impl_as_any, IntoOwned};
use ironrdp_server::{CliprdrServerFactory, ServerEvent, ServerEventSender};
use tokio::sync::mpsc;

use crate::formats::{
    dib_to_png, png_to_dib, uti_to_rdp_format_id, FORMAT_ID_FILE_LIST, FORMAT_ID_HTML,
};
use crate::pasteboard::PasteboardBridge;
use crate::transfer::FileTransferManager;

/// macOS clipboard backend implementing the CLIPRDR protocol.
///
/// Bridges between NSPasteboard and RDP CLIPRDR channel with:
/// - A polling thread that checks `NSPasteboard.changeCount` every 500ms
/// - Anti-echo mechanism to prevent clipboard feedback loops
/// - Text clipboard support (CF_UNICODETEXT <-> NSPasteboard text)
pub struct MacClipboardBackend {
    event_sender: mpsc::UnboundedSender<ServerEvent>,
    last_change_count: Arc<AtomicI64>,
    remote_formats: Vec<ClipboardFormat>,
    transfer_manager: Arc<Mutex<FileTransferManager>>,
    temp_dir: PathBuf,
    poll_handle: Option<JoinHandle<()>>,
    stop_signal: Arc<AtomicBool>,
    locked: bool,
    pending_paste_format: Option<PasteKind>,
}

/// Kind of remote clipboard data we asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PasteKind {
    FileList,
    Html,
    Dib,
    Text,
}

/// Pick the remote format to paste, best fidelity first, and keep the ID the remote
/// announced. Registered formats (file list, HTML) are matched by name: their IDs are
/// chosen by the remote at run time ([MS-RDPECLIP] 1.3.1.2), e.g. FreeRDP uses 0xC00A
/// for FileGroupDescriptorW.
fn choose_paste_format(formats: &[ClipboardFormat]) -> Option<(PasteKind, ClipboardFormatId)> {
    let by_name = |name: &ClipboardFormatName| {
        formats
            .iter()
            .find(|f| f.name().is_some_and(|n| n.value() == name.value()))
            .map(|f| f.id())
    };
    let by_id = |id: ClipboardFormatId| formats.iter().find(|f| f.id() == id).map(|f| f.id());
    by_name(&ClipboardFormatName::FILE_LIST)
        .map(|id| (PasteKind::FileList, id))
        .or_else(|| by_name(&ClipboardFormatName::HTML).map(|id| (PasteKind::Html, id)))
        .or_else(|| by_id(ClipboardFormatId::CF_DIB).map(|id| (PasteKind::Dib, id)))
        .or_else(|| by_id(ClipboardFormatId::CF_UNICODETEXT).map(|id| (PasteKind::Text, id)))
}

impl fmt::Debug for MacClipboardBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MacClipboardBackend")
            .field("temp_dir", &self.temp_dir)
            .finish_non_exhaustive()
    }
}

impl_as_any!(MacClipboardBackend);

impl MacClipboardBackend {
    pub fn new(
        event_sender: mpsc::UnboundedSender<ServerEvent>,
        temp_dir: PathBuf,
        max_file_size: u64,
    ) -> Self {
        std::fs::create_dir_all(&temp_dir).ok();
        Self {
            event_sender,
            last_change_count: Arc::new(AtomicI64::new(0)),
            remote_formats: Vec::new(),
            transfer_manager: Arc::new(Mutex::new(FileTransferManager::new(
                temp_dir.clone(),
                max_file_size,
            ))),
            temp_dir,
            poll_handle: None,
            stop_signal: Arc::new(AtomicBool::new(false)),
            locked: false,
            pending_paste_format: None,
        }
    }
}

/// Clipboard formats we can offer for the given pasteboard types.
fn format_list(types: &[String]) -> Vec<ClipboardFormat> {
    types
        .iter()
        .filter_map(|uti| {
            let fmt_id = uti_to_rdp_format_id(uti)?;
            let mut fmt = ClipboardFormat::new(ClipboardFormatId::new(fmt_id));
            match fmt_id {
                FORMAT_ID_HTML => {
                    fmt = fmt.with_name(ClipboardFormatName::HTML);
                }
                FORMAT_ID_FILE_LIST => {
                    fmt = fmt.with_name(ClipboardFormatName::FILE_LIST);
                }
                _ => {}
            }
            Some(fmt)
        })
        .collect()
}

/// Offer local files: remember them for file contents requests, and announce their
/// descriptors. The cliprdr crate keeps this list to answer the remote's file list
/// request and to check its file contents requests. `None` if no file can be offered.
fn file_copy_message(
    transfer: &Mutex<FileTransferManager>,
    paths: Vec<PathBuf>,
) -> Option<ClipboardMessage> {
    let files = transfer.lock().unwrap().set_local_files(paths);
    if files.is_empty() {
        return None;
    }
    Some(ClipboardMessage::SendInitiateFileCopy(files))
}

/// The message that announces the current pasteboard contents to the remote, if any.
fn announce_message(
    bridge: &PasteboardBridge,
    transfer: &Mutex<FileTransferManager>,
) -> Option<ClipboardMessage> {
    let mut formats = format_list(&bridge.available_types());
    if formats
        .iter()
        .any(|f| f.id().value() == FORMAT_ID_FILE_LIST)
    {
        if let Some(msg) = file_copy_message(transfer, bridge.read_file_urls()) {
            return Some(msg);
        }
        formats.retain(|f| f.id().value() != FORMAT_ID_FILE_LIST);
    }
    if formats.is_empty() {
        return None;
    }
    Some(ClipboardMessage::SendInitiateCopy(formats))
}

impl CliprdrBackend for MacClipboardBackend {
    fn temporary_directory(&self) -> &str {
        self.temp_dir.to_str().unwrap_or("/tmp/macrdp-clipboard")
    }

    fn client_capabilities(&self) -> ClipboardGeneralCapabilityFlags {
        ClipboardGeneralCapabilityFlags::USE_LONG_FORMAT_NAMES
            | ClipboardGeneralCapabilityFlags::STREAM_FILECLIP_ENABLED
            | ClipboardGeneralCapabilityFlags::FILECLIP_NO_FILE_PATHS
            | ClipboardGeneralCapabilityFlags::CAN_LOCK_CLIPDATA
    }

    fn on_ready(&mut self) {
        // Initialize change count and start polling thread
        let pb = PasteboardBridge::new();
        let initial = pb.change_count();
        self.last_change_count.store(initial, Ordering::SeqCst);

        let last = self.last_change_count.clone();
        let sender = self.event_sender.clone();
        let stop = self.stop_signal.clone();
        let transfer = self.transfer_manager.clone();

        self.poll_handle = Some(std::thread::spawn(move || {
            let bridge = PasteboardBridge::new();
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                let current = bridge.change_count();
                let prev = last.swap(current, Ordering::SeqCst);
                if current != prev {
                    // Clipboard changed -- announce the new contents
                    if let Some(msg) = announce_message(&bridge, &transfer) {
                        let _ = sender.send(ServerEvent::Clipboard(msg));
                    }
                }
            }
        }));
        tracing::info!("Clipboard polling started");
    }

    fn on_process_negotiated_capabilities(&mut self, _caps: ClipboardGeneralCapabilityFlags) {}

    fn on_request_format_list(&mut self) {
        let msg = announce_message(&PasteboardBridge::new(), &self.transfer_manager)
            .unwrap_or(ClipboardMessage::SendInitiateCopy(Vec::new()));
        let _ = self.event_sender.send(ServerEvent::Clipboard(msg));
    }

    fn on_remote_copy(&mut self, available_formats: &[ClipboardFormat]) {
        self.remote_formats = available_formats.to_vec();

        // Request the highest-fidelity format available, with the ID the remote announced.
        if let Some((kind, id)) = choose_paste_format(available_formats) {
            tracing::debug!(
                ?kind,
                format_id = id.value(),
                "Requesting remote clipboard data"
            );
            self.pending_paste_format = Some(kind);
            let _ = self.event_sender.send(ServerEvent::Clipboard(
                ClipboardMessage::SendInitiatePaste(id),
            ));
        }
    }

    fn on_format_data_request(&mut self, request: FormatDataRequest) {
        let pb = PasteboardBridge::new();
        let format_id = request.format.value();

        let response = match format_id {
            13 => {
                // CF_UNICODETEXT -- use ironrdp's built-in unicode string helper
                match pb.read_string() {
                    Some(text) => FormatDataResponse::new_unicode_string(&text),
                    None => FormatDataResponse::new_error(),
                }
            }
            8 => {
                // CF_DIB — convert macOS image to Windows DIB
                match pb.read_image() {
                    Some(png_data) => match png_to_dib(&png_data) {
                        Ok(dib) => FormatDataResponse::new_data(dib),
                        Err(e) => {
                            tracing::warn!("Failed to convert image to DIB: {e}");
                            FormatDataResponse::new_error()
                        }
                    },
                    None => FormatDataResponse::new_error(),
                }
            }
            id if id == FORMAT_ID_HTML => match pb.read_html() {
                Some(html_content) => {
                    let data = crate::html::wrap_html_format(&html_content);
                    FormatDataResponse::new_data(data)
                }
                None => FormatDataResponse::new_error(),
            },
            _ => {
                tracing::warn!(format_id, "Unsupported clipboard format requested");
                FormatDataResponse::new_error()
            }
        };

        // Convert to owned and send
        let owned: OwnedFormatDataResponse = response.into_owned();
        let _ = self
            .event_sender
            .send(ServerEvent::Clipboard(ClipboardMessage::SendFormatData(
                owned,
            )));
    }

    fn on_format_data_response(&mut self, response: FormatDataResponse<'_>) {
        if response.is_error() {
            tracing::warn!("Received error format data response from client");
            return;
        }

        let format = self.pending_paste_format.take();
        match format {
            Some(PasteKind::Text) => {
                if let Ok(text) = response.to_unicode_string() {
                    let pb = PasteboardBridge::new();
                    let new_count = pb.write_string(&text);
                    self.last_change_count.store(new_count, Ordering::SeqCst);
                }
            }
            Some(PasteKind::Dib) => {
                let data = response.data();
                if data.len() >= formats::BITMAPINFOHEADER_SIZE {
                    if let Ok(png_data) = dib_to_png(data) {
                        let pb = PasteboardBridge::new();
                        let new_count = pb.write_image(&png_data);
                        self.last_change_count.store(new_count, Ordering::SeqCst);
                    }
                }
            }
            Some(PasteKind::Html) => {
                if let Some(html_content) = crate::html::unwrap_html_format(response.data()) {
                    let pb = PasteboardBridge::new();
                    let new_count = pb.write_html(&html_content);
                    self.last_change_count.store(new_count, Ordering::SeqCst);
                }
            }
            _ => {
                // Fallback: content-based detection for backwards compatibility
                if let Ok(text) = response.to_unicode_string() {
                    let pb = PasteboardBridge::new();
                    let new_count = pb.write_string(&text);
                    self.last_change_count.store(new_count, Ordering::SeqCst);
                    return;
                }
                let data = response.data();
                if data.len() >= formats::BITMAPINFOHEADER_SIZE {
                    let header_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    if header_size == 40 {
                        if let Ok(png_data) = dib_to_png(data) {
                            let pb = PasteboardBridge::new();
                            let new_count = pb.write_image(&png_data);
                            self.last_change_count.store(new_count, Ordering::SeqCst);
                        }
                    }
                }
            }
        }
    }

    fn on_remote_file_list(&mut self, files: &[FileDescriptor], _clip_data_id: Option<u32>) {
        // The cliprdr crate takes the file list response, checks the names, and gives the
        // list here instead of to on_format_data_response.
        self.pending_paste_format = None;
        let requests = {
            let mut transfer = self.transfer_manager.lock().unwrap();
            transfer.set_incoming_descriptors(files.to_vec());
            transfer.generate_contents_requests()
        };
        for req in requests {
            let _ = self.event_sender.send(ServerEvent::Clipboard(
                ClipboardMessage::SendFileContentsRequest(req),
            ));
        }
    }

    fn on_file_contents_request(&mut self, request: FileContentsRequest) {
        let response = self
            .transfer_manager
            .lock()
            .unwrap()
            .handle_contents_request(&request);
        let _ = self.event_sender.send(ServerEvent::Clipboard(
            ClipboardMessage::SendFileContentsResponse(response),
        ));
    }

    fn on_file_contents_response(&mut self, response: FileContentsResponse<'_>) {
        let mut transfer = self.transfer_manager.lock().unwrap();
        transfer.handle_contents_response(&response);

        if transfer.all_incoming_complete() {
            let paths = transfer.flush_incoming_files();
            if !paths.is_empty() {
                let pb = PasteboardBridge::new();
                let new_count = pb.write_file_urls(&paths);
                self.last_change_count.store(new_count, Ordering::SeqCst);
                tracing::info!(count = paths.len(), "Wrote incoming files to pasteboard");
            }
        }
    }

    fn on_lock(&mut self, _data_id: LockDataId) {
        self.locked = true;
        self.transfer_manager.lock().unwrap().lock();
    }

    fn on_unlock(&mut self, _data_id: LockDataId) {
        self.locked = false;
        self.transfer_manager.lock().unwrap().unlock();
    }
}

impl Drop for MacClipboardBackend {
    fn drop(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        if let Some(handle) = self.poll_handle.take() {
            let _ = handle.join();
        }
    }
}

// --- Factory ---

/// Factory for creating `MacClipboardBackend` instances.
///
/// Implements `CliprdrServerFactory` which combines `CliprdrBackendFactory`
/// and `ServerEventSender`. The event sender is injected by the server
/// before `build_cliprdr_backend` is called.
pub struct MacClipboardFactory {
    event_sender: Option<mpsc::UnboundedSender<ServerEvent>>,
    temp_dir: PathBuf,
    max_file_size: u64,
}

impl MacClipboardFactory {
    pub fn new(temp_dir: PathBuf, max_file_size_mb: u32) -> Self {
        Self {
            event_sender: None,
            temp_dir,
            max_file_size: max_file_size_mb as u64 * 1024 * 1024,
        }
    }
}

impl CliprdrBackendFactory for MacClipboardFactory {
    fn build_cliprdr_backend(&self) -> Box<dyn CliprdrBackend> {
        Box::new(MacClipboardBackend::new(
            self.event_sender
                .clone()
                .expect("event sender not set before building clipboard backend"),
            self.temp_dir.clone(),
            self.max_file_size,
        ))
    }
}

impl ServerEventSender for MacClipboardFactory {
    fn set_sender(&mut self, sender: mpsc::UnboundedSender<ServerEvent>) {
        self.event_sender = Some(sender);
    }
}

impl CliprdrServerFactory for MacClipboardFactory {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_files_announced_as_file_copy() {
        let dir = std::env::temp_dir().join(format!("macrdp-announce-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hello.txt");
        std::fs::write(&path, b"hello world").unwrap();
        let transfer = Mutex::new(FileTransferManager::new(dir.join("tmp"), 1024));

        let msg = file_copy_message(&transfer, vec![path.clone(), dir.join("missing.bin")]);

        let Some(ClipboardMessage::SendInitiateFileCopy(files)) = msg else {
            panic!("expected SendInitiateFileCopy, got {msg:?}");
        };
        let got: Vec<_> = files
            .iter()
            .map(|f| (f.name.as_str(), f.file_size))
            .collect();
        assert_eq!(got, [("hello.txt", Some(11))]);

        // The same files serve the file contents requests.
        let request = FileContentsRequest {
            stream_id: 1,
            index: 0,
            flags: FileContentsFlags::SIZE,
            position: 0,
            requested_size: 8,
            data_id: None,
        };
        let response = transfer.lock().unwrap().handle_contents_request(&request);
        assert_eq!(response.data_as_size().unwrap(), 11);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn paste_matches_registered_formats_by_name() {
        // FreeRDP announces FileGroupDescriptorW at 0xC00A (chosen at run time).
        let remote = [
            ClipboardFormat::new(ClipboardFormatId::CF_UNICODETEXT),
            ClipboardFormat::new(ClipboardFormatId::new(0xC00A))
                .with_name(ClipboardFormatName::FILE_LIST),
            ClipboardFormat::new(ClipboardFormatId::new(0xC00B))
                .with_name(ClipboardFormatName::PREFERRED_DROP_EFFECT),
        ];
        assert_eq!(
            choose_paste_format(&remote),
            Some((PasteKind::FileList, ClipboardFormatId::new(0xC00A)))
        );

        // A different format at our local file list ID is not a file list.
        let remote = [
            ClipboardFormat::new(ClipboardFormatId::new(FORMAT_ID_FILE_LIST))
                .with_name(ClipboardFormatName::new_static("Other")),
            ClipboardFormat::new(ClipboardFormatId::new(0xC0F0))
                .with_name(ClipboardFormatName::HTML),
            ClipboardFormat::new(ClipboardFormatId::CF_DIB),
        ];
        assert_eq!(
            choose_paste_format(&remote),
            Some((PasteKind::Html, ClipboardFormatId::new(0xC0F0)))
        );

        let remote = [
            ClipboardFormat::new(ClipboardFormatId::CF_UNICODETEXT),
            ClipboardFormat::new(ClipboardFormatId::CF_DIB),
        ];
        assert_eq!(
            choose_paste_format(&remote),
            Some((PasteKind::Dib, ClipboardFormatId::CF_DIB))
        );
        assert_eq!(
            choose_paste_format(&remote[..1]),
            Some((PasteKind::Text, ClipboardFormatId::CF_UNICODETEXT))
        );
        assert_eq!(choose_paste_format(&[]), None);
    }

    #[test]
    fn no_file_copy_without_readable_files() {
        let dir = std::env::temp_dir().join(format!("macrdp-announce2-{}", std::process::id()));
        let transfer = Mutex::new(FileTransferManager::new(dir.join("tmp"), 1024));
        assert!(file_copy_message(&transfer, vec![dir.join("missing.bin")]).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
