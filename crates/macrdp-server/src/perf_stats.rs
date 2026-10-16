use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Frame-level performance statistics collector.
/// When enabled, records per-frame metrics and generates summary reports.
pub struct PerfStats {
    enabled: bool,
    encode_times: Vec<f64>,
    frame_sizes: Vec<u32>,
    keyframe_count: u32,
    total_frames: u64,
    start_time: Instant,
}

impl PerfStats {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            encode_times: Vec::with_capacity(if enabled { 4096 } else { 0 }),
            frame_sizes: Vec::with_capacity(if enabled { 4096 } else { 0 }),
            keyframe_count: 0,
            total_frames: 0,
            start_time: Instant::now(),
        }
    }

    /// Record metrics for one encoded frame.
    #[allow(dead_code)] // reason: video is bitmap only for now; H.264 on EGFX records frames again
    pub fn record_frame(&mut self, encode_ms: f64, frame_bytes: u32, is_keyframe: bool) {
        if !self.enabled {
            return;
        }
        self.encode_times.push(encode_ms);
        self.frame_sizes.push(frame_bytes);
        self.total_frames += 1;
        if is_keyframe {
            self.keyframe_count += 1;
        }
    }

    /// Print formatted performance summary to stdout.
    pub fn print_summary(&self) {
        if !self.enabled || self.total_frames == 0 {
            println!("Performance stats not available (no frames recorded).");
            return;
        }

        let duration = self.start_time.elapsed().as_secs_f64();
        let avg_fps = self.total_frames as f64 / duration;
        let kf_pct = if self.total_frames > 0 {
            self.keyframe_count as f64 / self.total_frames as f64 * 100.0
        } else {
            0.0
        };

        // Encode latency stats
        let enc = LatencyStats::compute(&self.encode_times);

        // Frame size stats
        let sizes_f64: Vec<f64> = self.frame_sizes.iter().map(|&s| s as f64).collect();
        let size_stats = LatencyStats::compute(&sizes_f64);

        // Throughput
        let total_bytes: u64 = self.frame_sizes.iter().map(|&s| s as u64).sum();
        let avg_mbps = total_bytes as f64 * 8.0 / duration / 1_000_000.0;

        println!();
        println!("═══════════════════════════════════════════");
        println!("  macrdp Performance Report");
        println!("═══════════════════════════════════════════");
        println!("  Duration:      {:.1}s", duration);
        println!("  Total frames:  {}", self.total_frames);
        println!("  Avg FPS:       {:.1}", avg_fps);
        println!("  Keyframes:     {} ({:.1}%)", self.keyframe_count, kf_pct);
        println!("───────────────────────────────────────────");
        println!("  Encode Latency (ms):");
        println!("    Mean:   {:<7.1} Stddev: {:.1}", enc.mean, enc.stddev);
        println!(
            "    P50:    {:<7.1} P95:    {:<7.1} P99:   {:.1}",
            enc.p50, enc.p95, enc.p99
        );
        println!("    Min:    {:<7.1} Max:    {:.1}", enc.min, enc.max);
        println!("───────────────────────────────────────────");
        println!("  Frame Size (KB):");
        println!(
            "    Mean:   {:<7.1} P95:   {:.1}",
            size_stats.mean / 1024.0,
            size_stats.p95 / 1024.0
        );
        println!("  Throughput:");
        println!("    Avg:    {:.1} Mbps", avg_mbps);
        println!("═══════════════════════════════════════════");
        println!();
    }
}

struct LatencyStats {
    mean: f64,
    stddev: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    min: f64,
    max: f64,
}

impl LatencyStats {
    fn compute(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self {
                mean: 0.0,
                stddev: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();

        Self {
            mean,
            stddev,
            p50: sorted[n * 50 / 100],
            p95: sorted[(n * 95 / 100).min(n - 1)],
            p99: sorted[(n * 99 / 100).min(n - 1)],
            min: sorted[0],
            max: sorted[n - 1],
        }
    }
}

/// Shared, thread-safe wrapper around PerfStats for use across threads.
pub type SharedPerfStats = Arc<Mutex<PerfStats>>;

/// Create a new shared PerfStats.
pub fn new_shared(enabled: bool) -> SharedPerfStats {
    Arc::new(Mutex::new(PerfStats::new(enabled)))
}
