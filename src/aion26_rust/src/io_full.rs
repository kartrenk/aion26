#![allow(dead_code)]
/// Disk I/O for Full HUNL Deep CFR trajectory storage
///
/// This module provides efficient binary file I/O for storing and loading
/// CFR trajectory data for the full multi-street game.
///
/// File Format:
/// - Each record: 912 bytes (220 × f32 state + 8 × f32 target)
/// - Files are named: epoch_{N}.bin
/// - Number of samples = file_size / 912

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write, Result, BufReader, Read};
use std::path::Path;

/// Size constants for Full HUNL
pub const STATE_DIM: usize = 220;
pub const TARGET_DIM: usize = 8;
pub const RECORD_SIZE: usize = (STATE_DIM + TARGET_DIM) * std::mem::size_of::<f32>();  // 912 bytes

/// Target normalization constant (Big Blind size)
const TARGET_NORMALIZER: f32 = 100.0;

/// Binary trajectory writer for disk-native Deep CFR (Full HUNL)
pub struct TrajectoryWriterFull {
    writer: BufWriter<File>,
    samples_written: usize,
    path: String,
}

impl TrajectoryWriterFull {
    /// Create a new writer for the given path
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
    pub fn append(&mut self, state: &[f32; STATE_DIM], target: &[f32; TARGET_DIM]) -> Result<()> {
        // Write state (220 × 4 = 880 bytes)
        let state_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                state.as_ptr() as *const u8,
                STATE_DIM * std::mem::size_of::<f32>()
            )
        };
        self.writer.write_all(state_bytes)?;

        // Normalize targets to Big Blinds before writing
        let mut normalized_target = [0.0f32; TARGET_DIM];
        for i in 0..TARGET_DIM {
            normalized_target[i] = target[i] / TARGET_NORMALIZER;
        }

        // Write normalized target (8 × 4 = 32 bytes)
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

    /// Append from slices
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

        // Normalize targets
        let mut normalized_target = [0.0f32; TARGET_DIM];
        for i in 0..TARGET_DIM {
            normalized_target[i] = target[i] / TARGET_NORMALIZER;
        }

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

impl Drop for TrajectoryWriterFull {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Binary trajectory reader for Full HUNL
pub struct TrajectoryReaderFull {
    reader: BufReader<File>,
    num_samples: usize,
    path: String,
}

impl TrajectoryReaderFull {
    /// Open an existing trajectory file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::open(&path)?;

        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;
        let num_samples = file_size / RECORD_SIZE;

        Ok(Self {
            reader: BufReader::with_capacity(1024 * 1024, file),
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

    #[test]
    fn test_record_size_full() {
        // Verify our size calculation is correct
        assert_eq!(RECORD_SIZE, 912);
        assert_eq!(STATE_DIM * 4 + TARGET_DIM * 4, 912);
    }
}
