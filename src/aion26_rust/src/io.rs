/// Disk I/O for Deep CFR trajectory storage
///
/// This module provides efficient binary file I/O for storing and loading
/// CFR trajectory data. Key features:
/// - Zero-copy writes via direct byte casting
/// - Buffered I/O for high throughput
/// - Simple binary format (no parsing overhead)
///
/// File Format:
/// - Each record: 560 bytes (136 × f32 state + 4 × f32 target)
/// - Files are named: epoch_{N}.bin
/// - Number of samples = file_size / 560

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write, Result, BufReader, Read};
use std::path::Path;

/// Size constants
pub const STATE_DIM: usize = 136;
pub const TARGET_DIM: usize = 4;
pub const RECORD_SIZE: usize = (STATE_DIM + TARGET_DIM) * std::mem::size_of::<f32>();  // 560 bytes

/// Target normalization constant (Big Blind size)
/// Dividing regrets by this makes loss values human-readable
const TARGET_NORMALIZER: f32 = 100.0;

/// Binary trajectory writer for disk-native Deep CFR
///
/// Writes (state, target) pairs to disk in a compact binary format.
/// Uses buffered I/O for high throughput (1MB buffer).
pub struct TrajectoryWriter {
    writer: BufWriter<File>,
    samples_written: usize,
    path: String,
}

impl TrajectoryWriter {
    /// Create a new writer for the given path
    ///
    /// Creates or truncates the file at `path`.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        Ok(Self {
            writer: BufWriter::with_capacity(1024 * 1024, file),  // 1MB buffer
            samples_written: 0,
            path: path_str,
        })
    }

    /// Append a (state, target) pair to the file
    ///
    /// # Arguments
    /// * `state` - 136-dimensional state encoding
    /// * `target` - 4-dimensional regret target (in raw chips, will be normalized)
    ///
    /// # Safety
    /// Uses unsafe byte casting for zero-copy performance.
    /// This is safe because f32 has well-defined representation.
    ///
    /// # Note
    /// Targets are normalized by dividing by 100.0 (Big Blind) before writing.
    /// This makes loss values human-readable (typical range: 0.1 - 10.0 instead of 100-1000).
    pub fn append(&mut self, state: &[f32; STATE_DIM], target: &[f32; TARGET_DIM]) -> Result<()> {
        // Write state (136 × 4 = 544 bytes)
        let state_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                state.as_ptr() as *const u8,
                STATE_DIM * std::mem::size_of::<f32>()
            )
        };
        self.writer.write_all(state_bytes)?;

        // Normalize targets to Big Blinds before writing
        let normalized_target: [f32; TARGET_DIM] = [
            target[0] / TARGET_NORMALIZER,
            target[1] / TARGET_NORMALIZER,
            target[2] / TARGET_NORMALIZER,
            target[3] / TARGET_NORMALIZER,
        ];

        // Write normalized target (4 × 4 = 16 bytes)
        let target_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                normalized_target.as_ptr() as *const u8,
                TARGET_DIM * std::mem::size_of::<f32>()
            )
        };
        self.writer.write_all(target_bytes)?;

        self.samples_written += 1;
        Ok(())
    }

    /// Append from slices (for compatibility with variable-size inputs)
    /// Note: Targets are normalized by dividing by 100.0 (Big Blind) before writing.
    pub fn append_slice(&mut self, state: &[f32], target: &[f32]) -> Result<()> {
        assert_eq!(state.len(), STATE_DIM, "State must have {} elements", STATE_DIM);
        assert_eq!(target.len(), TARGET_DIM, "Target must have {} elements", TARGET_DIM);

        // Write state
        let state_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                state.as_ptr() as *const u8,
                STATE_DIM * std::mem::size_of::<f32>()
            )
        };
        self.writer.write_all(state_bytes)?;

        // Normalize targets to Big Blinds before writing
        let normalized_target: [f32; TARGET_DIM] = [
            target[0] / TARGET_NORMALIZER,
            target[1] / TARGET_NORMALIZER,
            target[2] / TARGET_NORMALIZER,
            target[3] / TARGET_NORMALIZER,
        ];

        // Write normalized target
        let target_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                normalized_target.as_ptr() as *const u8,
                TARGET_DIM * std::mem::size_of::<f32>()
            )
        };
        self.writer.write_all(target_bytes)?;

        self.samples_written += 1;
        Ok(())
    }

    /// Flush the buffer and sync to disk
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()
    }

    /// Get number of samples written
    pub fn len(&self) -> usize {
        self.samples_written
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples_written == 0
    }

    /// Get the file path
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl Drop for TrajectoryWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Binary trajectory reader for loading samples
///
/// Provides random access to stored samples via memory mapping
/// or sequential reads via buffered I/O.
pub struct TrajectoryReader {
    reader: BufReader<File>,
    num_samples: usize,
    path: String,
}

impl TrajectoryReader {
    /// Open an existing trajectory file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::open(&path)?;

        // Get file size to compute sample count
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;
        let num_samples = file_size / RECORD_SIZE;

        Ok(Self {
            reader: BufReader::with_capacity(1024 * 1024, file),  // 1MB buffer
            num_samples,
            path: path_str,
        })
    }

    /// Get total number of samples in file
    pub fn len(&self) -> usize {
        self.num_samples
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_samples == 0
    }

    /// Read all samples into memory
    ///
    /// Returns (states, targets) as flattened vectors.
    pub fn read_all(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut states = vec![0.0f32; self.num_samples * STATE_DIM];
        let mut targets = vec![0.0f32; self.num_samples * TARGET_DIM];

        let mut record_buf = vec![0u8; RECORD_SIZE];

        for i in 0..self.num_samples {
            self.reader.read_exact(&mut record_buf)?;

            // Parse state
            let state_bytes = &record_buf[0..STATE_DIM * 4];
            let state: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    state_bytes.as_ptr() as *const f32,
                    STATE_DIM
                )
            };
            states[i * STATE_DIM..(i + 1) * STATE_DIM].copy_from_slice(state);

            // Parse target
            let target_bytes = &record_buf[STATE_DIM * 4..];
            let target: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    target_bytes.as_ptr() as *const f32,
                    TARGET_DIM
                )
            };
            targets[i * TARGET_DIM..(i + 1) * TARGET_DIM].copy_from_slice(target);
        }

        Ok((states, targets))
    }

    /// Get file path
    pub fn path(&self) -> &str {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_write_read_roundtrip() {
        let test_path = "/tmp/test_trajectory.bin";

        // Write some samples
        {
            let mut writer = TrajectoryWriter::new(test_path).unwrap();

            for i in 0..100 {
                let mut state = [0.0f32; STATE_DIM];
                let mut target = [0.0f32; TARGET_DIM];

                // Fill with test data
                state[0] = i as f32;
                state[135] = (i * 2) as f32;
                // Targets will be normalized by /100.0 when written
                target[0] = (i * 10) as f32;
                target[3] = (i * 20) as f32;

                writer.append(&state, &target).unwrap();
            }
            writer.flush().unwrap();
        }

        // Read back and verify
        {
            let mut reader = TrajectoryReader::new(test_path).unwrap();
            assert_eq!(reader.len(), 100);

            let (states, targets) = reader.read_all().unwrap();

            for i in 0..100 {
                assert_eq!(states[i * STATE_DIM], i as f32);
                assert_eq!(states[i * STATE_DIM + 135], (i * 2) as f32);
                // Targets are normalized (divided by 100.0) when written
                assert_eq!(targets[i * TARGET_DIM], (i * 10) as f32 / TARGET_NORMALIZER);
                assert_eq!(targets[i * TARGET_DIM + 3], (i * 20) as f32 / TARGET_NORMALIZER);
            }
        }

        // Cleanup
        fs::remove_file(test_path).unwrap();
    }

    #[test]
    fn test_record_size() {
        // Verify our size calculation is correct
        assert_eq!(RECORD_SIZE, 560);
        assert_eq!(STATE_DIM * 4 + TARGET_DIM * 4, 560);
    }
}
