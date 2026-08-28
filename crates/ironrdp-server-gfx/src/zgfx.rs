//! ZGFX (RDP8 Bulk Compression) compressor
//!
//! LZ77 + fixed Huffman coding per MS-RDPEGFX Section 3.1.8.
//! Token table from the spec's sample decompressor (Section 2.2.5.3).

const HISTORY_SIZE: usize = 2_500_000;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 65535;
const HASH_BITS: usize = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;
const MAX_CHAIN: usize = 32;

struct BitWriter {
    bytes: Vec<u8>,
    current: u32,
    bits_in_current: u8,
}

impl BitWriter {
    fn new(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
            current: 0,
            bits_in_current: 0,
        }
    }

    fn write_bits(&mut self, value: u32, count: u8) {
        debug_assert!(count <= 24);
        for i in (0..count).rev() {
            self.current = (self.current << 1) | ((value >> i) & 1);
            self.bits_in_current += 1;
            if self.bits_in_current == 8 {
                self.bytes.push(self.current as u8);
                self.current = 0;
                self.bits_in_current = 0;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        let padding = if self.bits_in_current > 0 {
            let pad = 8 - self.bits_in_current;
            self.current <<= pad;
            self.bytes.push(self.current as u8);
            pad
        } else {
            0
        };
        self.bytes.push(padding);
        self.bytes
    }
}

struct LiteralCode {
    prefix: u32,
    prefix_len: u8,
    #[allow(dead_code)]
    value_bits: u8,
    value_base: u8,
}

static LITERAL_SHORTCUTS: &[LiteralCode] = &[
    LiteralCode { prefix: 0b11000,    prefix_len: 5, value_bits: 0, value_base: 0x00 },
    LiteralCode { prefix: 0b11001,    prefix_len: 5, value_bits: 0, value_base: 0x01 },
    LiteralCode { prefix: 0b110100,   prefix_len: 6, value_bits: 0, value_base: 0x02 },
    LiteralCode { prefix: 0b110101,   prefix_len: 6, value_bits: 0, value_base: 0x03 },
    LiteralCode { prefix: 0b110110,   prefix_len: 6, value_bits: 0, value_base: 0xFF },
    LiteralCode { prefix: 0b1101110,  prefix_len: 7, value_bits: 0, value_base: 0x04 },
    LiteralCode { prefix: 0b1101111,  prefix_len: 7, value_bits: 0, value_base: 0x05 },
    LiteralCode { prefix: 0b1110000,  prefix_len: 7, value_bits: 0, value_base: 0x06 },
    LiteralCode { prefix: 0b1110001,  prefix_len: 7, value_bits: 0, value_base: 0x07 },
    LiteralCode { prefix: 0b1110010,  prefix_len: 7, value_bits: 0, value_base: 0x08 },
    LiteralCode { prefix: 0b1110011,  prefix_len: 7, value_bits: 0, value_base: 0x09 },
    LiteralCode { prefix: 0b1110100,  prefix_len: 7, value_bits: 0, value_base: 0x0A },
    LiteralCode { prefix: 0b1110101,  prefix_len: 7, value_bits: 0, value_base: 0x0B },
    LiteralCode { prefix: 0b1110110,  prefix_len: 7, value_bits: 0, value_base: 0x3A },
    LiteralCode { prefix: 0b1110111,  prefix_len: 7, value_bits: 0, value_base: 0x3B },
    LiteralCode { prefix: 0b1111000,  prefix_len: 7, value_bits: 0, value_base: 0x3C },
    LiteralCode { prefix: 0b1111001,  prefix_len: 7, value_bits: 0, value_base: 0x3D },
    LiteralCode { prefix: 0b1111010,  prefix_len: 7, value_bits: 0, value_base: 0x3E },
    LiteralCode { prefix: 0b1111011,  prefix_len: 7, value_bits: 0, value_base: 0x3F },
    LiteralCode { prefix: 0b1111100,  prefix_len: 7, value_bits: 0, value_base: 0x40 },
    LiteralCode { prefix: 0b1111101,  prefix_len: 7, value_bits: 0, value_base: 0x80 },
    LiteralCode { prefix: 0b11111100, prefix_len: 8, value_bits: 0, value_base: 0x0C },
    LiteralCode { prefix: 0b11111101, prefix_len: 8, value_bits: 0, value_base: 0x38 },
    LiteralCode { prefix: 0b11111110, prefix_len: 8, value_bits: 0, value_base: 0x39 },
    LiteralCode { prefix: 0b11111111, prefix_len: 8, value_bits: 0, value_base: 0x66 },
];

struct DistanceCode {
    prefix: u32,
    prefix_len: u8,
    value_bits: u8,
    base: u32,
    max: u32,
}

static DISTANCE_TABLE: &[DistanceCode] = &[
    DistanceCode { prefix: 0b10001,     prefix_len: 5, value_bits: 5,  base: 0,       max: 31 },
    DistanceCode { prefix: 0b10010,     prefix_len: 5, value_bits: 7,  base: 32,      max: 159 },
    DistanceCode { prefix: 0b10011,     prefix_len: 5, value_bits: 9,  base: 160,     max: 671 },
    DistanceCode { prefix: 0b10100,     prefix_len: 5, value_bits: 10, base: 672,     max: 1695 },
    DistanceCode { prefix: 0b10101,     prefix_len: 5, value_bits: 12, base: 1696,    max: 5791 },
    DistanceCode { prefix: 0b101100,    prefix_len: 6, value_bits: 14, base: 5792,    max: 22175 },
    DistanceCode { prefix: 0b101101,    prefix_len: 6, value_bits: 15, base: 22176,   max: 54943 },
    DistanceCode { prefix: 0b1011100,   prefix_len: 7, value_bits: 18, base: 54944,   max: 317087 },
    DistanceCode { prefix: 0b1011101,   prefix_len: 7, value_bits: 20, base: 317088,  max: 1365663 },
    DistanceCode { prefix: 0b10111100,  prefix_len: 8, value_bits: 20, base: 1365664, max: 2414239 },
    DistanceCode { prefix: 0b10111101,  prefix_len: 8, value_bits: 21, base: 2414240, max: 4511391 },
];

fn literal_lookup_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    for sc in LITERAL_SHORTCUTS {
        table[sc.value_base as usize] = ((sc.prefix_len as u16) << 9) | sc.prefix as u16;
    }
    table
}

fn write_literal(w: &mut BitWriter, byte: u8, lookup: &[u16; 256]) {
    let entry = lookup[byte as usize];
    if entry != 0 {
        let plen = (entry >> 9) as u8;
        let pcode = entry & 0x1FF;
        w.write_bits(pcode as u32, plen);
    } else {
        w.write_bits(0, 1);
        w.write_bits(byte as u32, 8);
    }
}

fn write_match(w: &mut BitWriter, distance: u32, length: usize) {
    for dc in DISTANCE_TABLE {
        if distance >= dc.base && distance <= dc.max {
            w.write_bits(dc.prefix, dc.prefix_len);
            w.write_bits(distance - dc.base, dc.value_bits);
            write_match_length(w, length);
            return;
        }
    }
}

fn write_match_length(w: &mut BitWriter, length: usize) {
    if length == 3 {
        w.write_bits(0, 1);
        return;
    }
    w.write_bits(1, 1);
    let k = (usize::BITS - 1 - length.leading_zeros()) as u8;
    for _ in 2..k {
        w.write_bits(1, 1);
    }
    w.write_bits(0, 1);
    w.write_bits((length - (1 << k)) as u32, k);
}

fn hash3(data: &[u8], pos: usize) -> usize {
    let h = (data[pos] as u32) | ((data[pos + 1] as u32) << 8) | ((data[pos + 2] as u32) << 16);
    ((h.wrapping_mul(0x9E3779B1)) >> (32 - HASH_BITS)) as usize
}

pub(crate) fn compress(data: &[u8]) -> Vec<u8> {
    if data.len() < MIN_MATCH {
        return uncompressed_segment(data);
    }

    let lookup = literal_lookup_table();
    let mut w = BitWriter::new(data.len());

    let mut head = vec![0u32; HASH_SIZE];
    let prev_size = data.len().min(HISTORY_SIZE);
    let mut prev = vec![0u32; prev_size];

    let mut pos = 0;
    while pos < data.len() {
        if pos + MIN_MATCH > data.len() {
            write_literal(&mut w, data[pos], &lookup);
            pos += 1;
            continue;
        }

        let h = hash3(data, pos);
        let mut best_len = MIN_MATCH - 1;
        let mut best_dist = 0u32;
        let mut chain_idx = head[h];
        let mut chain_count = 0;

        while chain_idx > 0 && chain_count < MAX_CHAIN {
            let candidate = (chain_idx - 1) as usize;
            if candidate >= pos {
                break;
            }
            let dist = (pos - candidate) as u32;
            if dist > DISTANCE_TABLE.last().unwrap().max {
                break;
            }

            let max_len = (data.len() - pos).min(MAX_MATCH);
            let mut len = 0;
            while len < max_len && data[candidate + len] == data[pos + len] {
                len += 1;
            }

            if len > best_len {
                best_len = len;
                best_dist = dist;
                if best_len == max_len {
                    break;
                }
            }

            chain_count += 1;
            let p = prev[candidate % prev_size] as usize;
            chain_idx = if p > 0 && p < pos { p as u32 } else { 0 };
        }

        if pos < prev_size {
            prev[pos % prev_size] = head[h];
            head[h] = (pos + 1) as u32;
        }

        if best_len >= MIN_MATCH {
            write_match(&mut w, best_dist, best_len);
            for i in 1..best_len {
                let fwd = pos + i;
                if fwd + MIN_MATCH <= data.len() && fwd < prev_size {
                    let fh = hash3(data, fwd);
                    prev[fwd % prev_size] = head[fh];
                    head[fh] = (fwd + 1) as u32;
                }
            }
            pos += best_len;
        } else {
            write_literal(&mut w, data[pos], &lookup);
            pos += 1;
        }
    }

    let compressed = w.finish();
    if compressed.len() >= data.len() {
        return uncompressed_segment(data);
    }

    let mut segment = Vec::with_capacity(1 + compressed.len());
    segment.push(0x24);
    segment.extend_from_slice(&compressed);
    segment
}

fn uncompressed_segment(data: &[u8]) -> Vec<u8> {
    let mut segment = Vec::with_capacity(1 + data.len());
    segment.push(0x04);
    segment.extend_from_slice(data);
    segment
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TokenEntry {
        prefix_len: u8,
        prefix_code: u32,
        value_bits: u8,
        token_type: u8,
        value_base: u32,
    }

    static TOKEN_TABLE: &[TokenEntry] = &[
        TokenEntry { prefix_len: 1, prefix_code: 0,   value_bits: 8,  token_type: 0, value_base: 0 },
        TokenEntry { prefix_len: 5, prefix_code: 17,  value_bits: 5,  token_type: 1, value_base: 0 },
        TokenEntry { prefix_len: 5, prefix_code: 18,  value_bits: 7,  token_type: 1, value_base: 32 },
        TokenEntry { prefix_len: 5, prefix_code: 19,  value_bits: 9,  token_type: 1, value_base: 160 },
        TokenEntry { prefix_len: 5, prefix_code: 20,  value_bits: 10, token_type: 1, value_base: 672 },
        TokenEntry { prefix_len: 5, prefix_code: 21,  value_bits: 12, token_type: 1, value_base: 1696 },
        TokenEntry { prefix_len: 5, prefix_code: 24,  value_bits: 0,  token_type: 0, value_base: 0x00 },
        TokenEntry { prefix_len: 5, prefix_code: 25,  value_bits: 0,  token_type: 0, value_base: 0x01 },
        TokenEntry { prefix_len: 6, prefix_code: 44,  value_bits: 14, token_type: 1, value_base: 5792 },
        TokenEntry { prefix_len: 6, prefix_code: 45,  value_bits: 15, token_type: 1, value_base: 22176 },
        TokenEntry { prefix_len: 6, prefix_code: 52,  value_bits: 0,  token_type: 0, value_base: 0x02 },
        TokenEntry { prefix_len: 6, prefix_code: 53,  value_bits: 0,  token_type: 0, value_base: 0x03 },
        TokenEntry { prefix_len: 6, prefix_code: 54,  value_bits: 0,  token_type: 0, value_base: 0xFF },
        TokenEntry { prefix_len: 7, prefix_code: 92,  value_bits: 18, token_type: 1, value_base: 54944 },
        TokenEntry { prefix_len: 7, prefix_code: 93,  value_bits: 20, token_type: 1, value_base: 317088 },
        TokenEntry { prefix_len: 7, prefix_code: 110, value_bits: 0,  token_type: 0, value_base: 0x04 },
        TokenEntry { prefix_len: 7, prefix_code: 111, value_bits: 0,  token_type: 0, value_base: 0x05 },
        TokenEntry { prefix_len: 7, prefix_code: 112, value_bits: 0,  token_type: 0, value_base: 0x06 },
        TokenEntry { prefix_len: 7, prefix_code: 113, value_bits: 0,  token_type: 0, value_base: 0x07 },
        TokenEntry { prefix_len: 7, prefix_code: 114, value_bits: 0,  token_type: 0, value_base: 0x08 },
        TokenEntry { prefix_len: 7, prefix_code: 115, value_bits: 0,  token_type: 0, value_base: 0x09 },
        TokenEntry { prefix_len: 7, prefix_code: 116, value_bits: 0,  token_type: 0, value_base: 0x0A },
        TokenEntry { prefix_len: 7, prefix_code: 117, value_bits: 0,  token_type: 0, value_base: 0x0B },
        TokenEntry { prefix_len: 7, prefix_code: 118, value_bits: 0,  token_type: 0, value_base: 0x3A },
        TokenEntry { prefix_len: 7, prefix_code: 119, value_bits: 0,  token_type: 0, value_base: 0x3B },
        TokenEntry { prefix_len: 7, prefix_code: 120, value_bits: 0,  token_type: 0, value_base: 0x3C },
        TokenEntry { prefix_len: 7, prefix_code: 121, value_bits: 0,  token_type: 0, value_base: 0x3D },
        TokenEntry { prefix_len: 7, prefix_code: 122, value_bits: 0,  token_type: 0, value_base: 0x3E },
        TokenEntry { prefix_len: 7, prefix_code: 123, value_bits: 0,  token_type: 0, value_base: 0x3F },
        TokenEntry { prefix_len: 7, prefix_code: 124, value_bits: 0,  token_type: 0, value_base: 0x40 },
        TokenEntry { prefix_len: 7, prefix_code: 125, value_bits: 0,  token_type: 0, value_base: 0x80 },
        TokenEntry { prefix_len: 8, prefix_code: 188, value_bits: 20, token_type: 1, value_base: 1365664 },
        TokenEntry { prefix_len: 8, prefix_code: 189, value_bits: 21, token_type: 1, value_base: 2414240 },
        TokenEntry { prefix_len: 8, prefix_code: 252, value_bits: 0,  token_type: 0, value_base: 0x0C },
        TokenEntry { prefix_len: 8, prefix_code: 253, value_bits: 0,  token_type: 0, value_base: 0x38 },
        TokenEntry { prefix_len: 8, prefix_code: 254, value_bits: 0,  token_type: 0, value_base: 0x39 },
        TokenEntry { prefix_len: 8, prefix_code: 255, value_bits: 0,  token_type: 0, value_base: 0x66 },
        TokenEntry { prefix_len: 9, prefix_code: 380, value_bits: 22, token_type: 1, value_base: 4511392 },
        TokenEntry { prefix_len: 9, prefix_code: 381, value_bits: 23, token_type: 1, value_base: 8705696 },
        TokenEntry { prefix_len: 9, prefix_code: 382, value_bits: 24, token_type: 1, value_base: 17094304 },
    ];

    struct BitReader<'a> {
        data: &'a [u8],
        pos: usize,
        end: usize,
        bits_remaining: usize,
        current: u32,
        bits_current: u8,
    }

    impl<'a> BitReader<'a> {
        fn new(encoded: &'a [u8]) -> Self {
            let end = encoded.len() - 1;
            let padding = encoded[end] as usize;
            Self {
                data: encoded,
                pos: 0,
                end,
                bits_remaining: 8 * end - padding,
                current: 0,
                bits_current: 0,
            }
        }

        fn get_bits(&mut self, count: u8) -> u32 {
            while self.bits_current < count {
                self.current <<= 8;
                if self.pos < self.end {
                    self.current += self.data[self.pos] as u32;
                    self.pos += 1;
                }
                self.bits_current += 8;
            }
            self.bits_remaining -= count as usize;
            self.bits_current -= count;
            let result = self.current >> self.bits_current;
            self.current &= (1 << self.bits_current) - 1;
            result
        }
    }

    fn decompress_segment(segment: &[u8]) -> Vec<u8> {
        if segment[0] & 0x20 == 0 {
            return segment[1..].to_vec();
        }
        let encoded = &segment[1..];
        let mut reader = BitReader::new(encoded);
        let mut history = vec![0u8; HISTORY_SIZE];
        let mut hist_idx = 0usize;
        let mut output = Vec::new();

        while reader.bits_remaining > 0 {
            let mut have_bits: u8 = 0;
            let mut prefix: u32 = 0;
            let mut found = false;

            for entry in TOKEN_TABLE {
                while have_bits < entry.prefix_len {
                    prefix = (prefix << 1) + reader.get_bits(1);
                    have_bits += 1;
                }
                if prefix == entry.prefix_code {
                    if entry.token_type == 0 {
                        let c = (entry.value_base + reader.get_bits(entry.value_bits)) as u8;
                        history[hist_idx] = c;
                        hist_idx = (hist_idx + 1) % HISTORY_SIZE;
                        output.push(c);
                    } else {
                        let distance = entry.value_base + reader.get_bits(entry.value_bits);
                        if distance != 0 {
                            let count = if reader.get_bits(1) == 0 {
                                3
                            } else {
                                let mut cnt = 4u32;
                                let mut extra = 2u8;
                                while reader.get_bits(1) == 1 {
                                    cnt *= 2;
                                    extra += 1;
                                }
                                cnt + reader.get_bits(extra)
                            };
                            let mut prev_idx = (hist_idx + HISTORY_SIZE - distance as usize) % HISTORY_SIZE;
                            for _ in 0..count {
                                let c = history[prev_idx];
                                prev_idx = (prev_idx + 1) % HISTORY_SIZE;
                                history[hist_idx] = c;
                                hist_idx = (hist_idx + 1) % HISTORY_SIZE;
                                output.push(c);
                            }
                        } else {
                            let count = reader.get_bits(15);
                            reader.bits_remaining -= reader.bits_current as usize;
                            reader.bits_current = 0;
                            reader.current = 0;
                            for _ in 0..count {
                                let c = reader.data[reader.pos];
                                reader.pos += 1;
                                reader.bits_remaining -= 8;
                                history[hist_idx] = c;
                                hist_idx = (hist_idx + 1) % HISTORY_SIZE;
                                output.push(c);
                            }
                        }
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        output
    }

    fn roundtrip(data: &[u8]) {
        let segment = compress(data);
        let decoded = decompress_segment(&segment);
        assert_eq!(data, decoded.as_slice(), "roundtrip failed for {} bytes (segment type 0x{:02x}, compressed {} bytes)", data.len(), segment[0], segment.len());
    }

    #[test]
    fn test_compress_small() {
        let data = b"hello";
        let result = compress(data);
        assert_eq!(result[0], 0x04);
        assert_eq!(&result[1..], b"hello");
        roundtrip(data);
    }

    #[test]
    fn test_compress_repetitive() {
        let data = vec![0u8; 1024];
        let result = compress(&data);
        assert_eq!(result[0], 0x24);
        assert!(result.len() < data.len() / 2);
        roundtrip(&data);
    }

    #[test]
    fn test_compress_bgra_pattern() {
        let mut data = Vec::with_capacity(4096);
        for _ in 0..1024 {
            data.extend_from_slice(&[0x00, 0x80, 0xFF, 0xFF]);
        }
        let result = compress(&data);
        assert_eq!(result[0], 0x24);
        assert!(result.len() < data.len() / 2);
        roundtrip(&data);
    }

    #[test]
    fn test_roundtrip_all_byte_values() {
        let data: Vec<u8> = (0..=255).collect();
        roundtrip(&data);
    }

    #[test]
    fn test_roundtrip_random_like() {
        let mut data = vec![0u8; 4096];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i * 7 + 13) % 256) as u8;
        }
        roundtrip(&data);
    }

    #[test]
    fn test_roundtrip_mixed_matches() {
        let mut data = Vec::with_capacity(8192);
        for i in 0..100 {
            let val = (i % 4) as u8;
            data.extend_from_slice(&[val, val + 1, val + 2, val + 3]);
        }
        for _ in 0..50 {
            data.extend_from_slice(&data[..400].to_vec());
        }
        roundtrip(&data);
    }

    #[test]
    fn test_compression_ratios() {
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("all_zeros_64k", vec![0u8; 65536]),
            ("bgra_uniform_64k", {
                let mut d = Vec::with_capacity(65536);
                for _ in 0..16384 { d.extend_from_slice(&[0x1A, 0x2B, 0x3C, 0xFF]); }
                d
            }),
            ("bgra_gradient_64k", {
                let mut d = Vec::with_capacity(65536);
                for i in 0..16384 {
                    let v = (i % 256) as u8;
                    d.extend_from_slice(&[v, v.wrapping_add(1), v.wrapping_add(2), 0xFF]);
                }
                d
            }),
        ];
        for (name, data) in &cases {
            let seg = compress(data);
            let ratio = seg.len() as f64 / data.len() as f64 * 100.0;
            eprintln!("{name}: {} -> {} bytes ({ratio:.1}%)", data.len(), seg.len());
            roundtrip(data);
            assert!(seg[0] == 0x24, "{name} should compress");
        }
    }

    // --- Edge cases ---

    #[test]
    fn test_empty_data() {
        let data = b"";
        let result = compress(data);
        assert_eq!(result[0], 0x04);
        assert_eq!(result.len(), 1);
        roundtrip(data);
    }

    #[test]
    fn test_one_byte() {
        let data = b"\x42";
        let result = compress(data);
        assert_eq!(result[0], 0x04);
        roundtrip(data);
    }

    #[test]
    fn test_two_bytes() {
        let data = b"\xAB\xCD";
        let result = compress(data);
        assert_eq!(result[0], 0x04);
        roundtrip(data);
    }

    #[test]
    fn test_exactly_three_bytes_no_match() {
        let data = b"\x10\x20\x30";
        roundtrip(data);
    }

    #[test]
    fn test_exactly_min_match() {
        let data = b"abcabc";
        roundtrip(data);
    }

    // --- Segment format ---

    #[test]
    fn test_compressed_segment_format() {
        let data = vec![0u8; 256];
        let seg = compress(&data);
        assert_eq!(seg[0], 0x24, "first byte must be PACKET_COMPRESSED | PACKET_COMPR_TYPE_RDP8");
        let padding = seg[seg.len() - 1];
        assert!(padding < 8, "padding byte must be 0..7, got {padding}");
    }

    #[test]
    fn test_uncompressed_segment_format() {
        let data = b"xy";
        let seg = compress(data);
        assert_eq!(seg[0], 0x04, "first byte must be PACKET_COMPR_TYPE_RDP8 (uncompressed)");
        assert_eq!(&seg[1..], data);
    }

    #[test]
    fn test_incompressible_falls_back() {
        let data: Vec<u8> = (0..=255).cycle().take(256).collect();
        let seg = compress(&data);
        assert_eq!(seg[0], 0x04, "incompressible data should use uncompressed segment");
        assert_eq!(&seg[1..], data.as_slice());
    }

    // --- Literal shortcut codes ---

    #[test]
    fn test_literal_shortcut_bytes() {
        let shortcut_bytes: &[u8] = &[
            0x00, 0x01, 0x02, 0x03, 0xFF, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E,
            0x3F, 0x40, 0x80, 0x0C, 0x38, 0x39, 0x66,
        ];
        for &b in shortcut_bytes {
            let data = vec![b; 64];
            roundtrip(&data);
        }
    }

    #[test]
    fn test_all_shortcut_bytes_mixed() {
        let data: Vec<u8> = vec![
            0x00, 0x01, 0x02, 0x03, 0xFF, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E,
            0x3F, 0x40, 0x80, 0x0C, 0x38, 0x39, 0x66,
        ];
        roundtrip(&data);
    }

    // --- Match lengths ---

    #[test]
    fn test_match_length_3() {
        let mut data = vec![0xAA, 0xBB, 0xCC, 0xDD];
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        roundtrip(&data);
    }

    #[test]
    fn test_match_length_4_to_7() {
        for len in 4..=7 {
            let mut data = vec![0u8; 32];
            for (i, b) in data.iter_mut().enumerate() {
                *b = (i % 11) as u8 + 0x30;
            }
            data.extend_from_slice(&data[5..5 + len].to_vec());
            roundtrip(&data);
        }
    }

    #[test]
    fn test_match_length_8_to_15() {
        for len in 8..=15 {
            let pattern: Vec<u8> = (0..32).map(|i| (i * 3 + 7) as u8).collect();
            let mut data = pattern.clone();
            data.extend_from_slice(&pattern[..len]);
            roundtrip(&data);
        }
    }

    #[test]
    fn test_match_length_powers_of_two() {
        for &len in &[16, 32, 64, 128, 256, 512] {
            let pattern: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let mut data = pattern.clone();
            data.extend_from_slice(&pattern);
            roundtrip(&data);
        }
    }

    #[test]
    fn test_long_match() {
        let pattern: Vec<u8> = (0..1000).map(|i| (i % 199) as u8).collect();
        let mut data = pattern.clone();
        data.extend_from_slice(&pattern);
        roundtrip(&data);
    }

    // --- Match distances ---

    #[test]
    fn test_match_distance_ranges() {
        let distances: &[usize] = &[3, 10, 31, 32, 100, 160, 500, 672, 1000, 1696, 4000, 5792];
        for &dist in distances {
            let prefix_len = dist + 3;
            let mut data = Vec::with_capacity(prefix_len + 8);
            for i in 0..prefix_len {
                data.push((i % 251) as u8);
            }
            data.extend_from_slice(&data[0..3].to_vec());
            roundtrip(&data);
        }
    }

    // --- Realistic data patterns ---

    #[test]
    fn test_desktop_like_scanlines() {
        let width = 256;
        let mut data = Vec::with_capacity(width * 4 * 10);
        for row in 0..10u8 {
            let bg = if row < 3 { [0x2D, 0x2D, 0x2D, 0xFF] } else { [0xF0, 0xF0, 0xF0, 0xFF] };
            for col in 0..width {
                if col > 50 && col < 200 && row > 1 && row < 8 {
                    data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
                } else {
                    data.extend_from_slice(&bg);
                }
            }
        }
        roundtrip(&data);
        let seg = compress(&data);
        assert_eq!(seg[0], 0x24);
        assert!(seg.len() < data.len() / 3);
    }

    #[test]
    fn test_text_like_sparse_changes() {
        let mut data = vec![0xFF_u8; 4096];
        for i in (0..4096).step_by(97) {
            data[i] = 0x00;
        }
        roundtrip(&data);
    }

    #[test]
    fn test_alternating_pixels() {
        let mut data = Vec::with_capacity(4096);
        for i in 0..1024 {
            if i % 2 == 0 {
                data.extend_from_slice(&[0x00, 0x00, 0x00, 0xFF]);
            } else {
                data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
            }
        }
        roundtrip(&data);
    }

    // --- Bit writer ---

    #[test]
    fn test_bit_writer_byte_aligned() {
        let mut w = BitWriter::new(16);
        w.write_bits(0xAB, 8);
        w.write_bits(0xCD, 8);
        let bytes = w.finish();
        assert_eq!(bytes[0], 0xAB);
        assert_eq!(bytes[1], 0xCD);
        assert_eq!(bytes[2], 0);
    }

    #[test]
    fn test_bit_writer_single_bits() {
        let mut w = BitWriter::new(16);
        w.write_bits(1, 1);
        w.write_bits(0, 1);
        w.write_bits(1, 1);
        w.write_bits(1, 1);
        w.write_bits(0, 1);
        w.write_bits(0, 1);
        w.write_bits(1, 1);
        w.write_bits(0, 1);
        let bytes = w.finish();
        assert_eq!(bytes[0], 0b10110010);
        assert_eq!(bytes[1], 0);
    }

    #[test]
    fn test_bit_writer_roundtrip() {
        let mut w = BitWriter::new(16);
        w.write_bits(0b10110, 5);
        w.write_bits(0b111, 3);
        let bytes = w.finish();
        assert_eq!(bytes[0], 0b10110_111);
        assert_eq!(bytes[1], 0);
    }

    #[test]
    fn test_bit_writer_padding() {
        let mut w = BitWriter::new(16);
        w.write_bits(0b101, 3);
        let bytes = w.finish();
        assert_eq!(bytes[0], 0b10100000);
        assert_eq!(bytes[1], 5);
    }

    // --- Match length encoding ---

    #[test]
    fn test_match_length_encoding() {
        let mut w = BitWriter::new(16);
        write_match_length(&mut w, 3);
        let bytes = w.finish();
        assert_eq!(bytes[0] >> 7, 0);

        let mut w = BitWriter::new(16);
        write_match_length(&mut w, 4);
        let bytes = w.finish();
        assert_eq!(bytes[0] >> 4, 0b1_0_00);
    }

    #[test]
    fn test_match_length_boundary_values() {
        for len in [3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128] {
            let pattern: Vec<u8> = (0..200).map(|i| ((i * 7 + 3) % 251) as u8).collect();
            let mut data = pattern.clone();
            let actual_len = len.min(pattern.len());
            data.extend_from_slice(&pattern[..actual_len]);
            roundtrip(&data);
        }
    }

    // --- Larger data ---

    #[test]
    fn test_large_compressible() {
        let mut data = Vec::with_capacity(262144);
        let block: Vec<u8> = (0..256).map(|i| i as u8).collect();
        for _ in 0..1024 {
            data.extend_from_slice(&block);
        }
        let seg = compress(&data);
        assert_eq!(seg[0], 0x24);
        roundtrip(&data);
    }
}
