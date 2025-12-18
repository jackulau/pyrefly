/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Protocol types for daemon IPC communication.
//!
//! Uses length-prefixed JSON messages over Unix domain sockets.

use std::path::PathBuf;

use serde::Deserialize;
use serde::Serialize;

/// Request sent from client to daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonRequest {
    /// Check specific files for type errors.
    Check {
        /// Files to check. If empty with check_all=false, checks all project files.
        files: Vec<PathBuf>,
        /// Whether to report errors from all reachable modules.
        check_all: bool,
    },
    /// Get daemon status information.
    Status,
    /// Request graceful shutdown.
    Shutdown,
    /// Force daemon to re-index the project.
    Reindex,
    /// Ping to verify daemon is alive.
    Ping,
}

/// Response from daemon to client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonResponse {
    /// Type check results.
    CheckResult {
        /// List of errors found.
        errors: Vec<SerializableError>,
        /// Total error count.
        error_count: usize,
        /// Number of files checked.
        checked_files: usize,
        /// Time taken in milliseconds.
        duration_ms: u64,
    },
    /// Daemon status information.
    Status {
        /// Daemon process ID.
        pid: u32,
        /// Time since daemon started in seconds.
        uptime_secs: u64,
        /// Number of indexed modules.
        indexed_modules: usize,
        /// Memory usage in megabytes.
        memory_usage_mb: f64,
        /// Project root being watched.
        project_root: PathBuf,
    },
    /// Simple acknowledgement.
    Ok,
    /// Pong response to ping.
    Pong,
    /// Error occurred.
    Error {
        /// Error message.
        message: String,
    },
}

/// A serializable error for IPC transmission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableError {
    /// Path to the file containing the error.
    pub path: PathBuf,
    /// Line number (1-indexed).
    pub line: u32,
    /// Column number (1-indexed).
    pub column: u32,
    /// Error message.
    pub message: String,
    /// Severity level (error, warning, etc.).
    pub severity: String,
    /// Error code if available.
    pub code: Option<String>,
}

/// Protocol version for future compatibility.
#[allow(dead_code)]
pub const PROTOCOL_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let request = DaemonRequest::Check {
            files: vec![PathBuf::from("test.py")],
            check_all: false,
        };
        let json = serde_json::to_string(&request).unwrap();
        let parsed: DaemonRequest = serde_json::from_str(&json).unwrap();
        match parsed {
            DaemonRequest::Check { files, check_all } => {
                assert_eq!(files, vec![PathBuf::from("test.py")]);
                assert!(!check_all);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_response_serialization() {
        let response = DaemonResponse::CheckResult {
            errors: vec![SerializableError {
                path: PathBuf::from("test.py"),
                line: 10,
                column: 5,
                message: "Type error".to_string(),
                severity: "error".to_string(),
                code: Some("E001".to_string()),
            }],
            error_count: 1,
            checked_files: 1,
            duration_ms: 100,
        };
        let json = serde_json::to_string(&response).unwrap();
        let parsed: DaemonResponse = serde_json::from_str(&json).unwrap();
        match parsed {
            DaemonResponse::CheckResult {
                error_count,
                checked_files,
                ..
            } => {
                assert_eq!(error_count, 1);
                assert_eq!(checked_files, 1);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_message_roundtrip() {
        use std::io::{Read, Write};

        // Test length-prefixed JSON roundtrip
        let request = DaemonRequest::Status;
        let json = serde_json::to_vec(&request).unwrap();

        // Write with length prefix
        let mut buf = Vec::new();
        buf.write_all(&(json.len() as u32).to_be_bytes()).unwrap();
        buf.write_all(&json).unwrap();

        // Read with length prefix
        let mut cursor = std::io::Cursor::new(&buf);
        let mut len_buf = [0u8; 4];
        cursor.read_exact(&mut len_buf).unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;

        let mut payload = vec![0u8; len];
        cursor.read_exact(&mut payload).unwrap();

        let parsed: DaemonRequest = serde_json::from_slice(&payload).unwrap();
        assert!(matches!(parsed, DaemonRequest::Status));
    }
}
