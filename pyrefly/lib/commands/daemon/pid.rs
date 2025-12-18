/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PID file management for daemon lifecycle.
//!
//! Handles writing, reading, and validating PID files to detect running daemons
//! and clean up stale state.

use std::fs;
use std::path::Path;

use anyhow::Context;

/// Write the current process ID to a PID file.
pub fn write_pid(path: &Path) -> anyhow::Result<()> {
    let pid = std::process::id();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("Failed to create PID file directory")?;
    }
    fs::write(path, pid.to_string()).context("Failed to write PID file")?;
    Ok(())
}

/// Read a PID from a PID file.
pub fn read_pid(path: &Path) -> Option<u32> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Check if a process with the given PID is alive.
#[cfg(unix)]
pub fn is_process_alive(pid: u32) -> bool {
    // On Unix, sending signal 0 checks if process exists without actually sending a signal
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(windows)]
pub fn is_process_alive(pid: u32) -> bool {
    use winapi::um::handleapi::CloseHandle;
    use winapi::um::processthreadsapi::OpenProcess;
    use winapi::um::winnt::PROCESS_QUERY_LIMITED_INFORMATION;

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            false
        } else {
            CloseHandle(handle);
            true
        }
    }
}

#[cfg(not(any(unix, windows)))]
pub fn is_process_alive(_pid: u32) -> bool {
    // On other platforms, assume process is alive if we can't check
    // This is a conservative approach that prevents accidental cleanup
    true
}

/// Clean up stale socket and PID files.
pub fn cleanup_stale_files(socket_path: &Path, pid_path: &Path) -> anyhow::Result<()> {
    if socket_path.exists() {
        fs::remove_file(socket_path).context("Failed to remove stale socket file")?;
    }
    if pid_path.exists() {
        fs::remove_file(pid_path).context("Failed to remove stale PID file")?;
    }
    Ok(())
}

/// Remove the PID file on daemon shutdown.
pub fn remove_pid(path: &Path) {
    let _ = fs::remove_file(path);
}

/// Check if the daemon for a given socket/PID file pair is running.
///
/// Returns `Some(pid)` if a daemon is running, `None` if no daemon is running
/// (and cleans up stale files if needed).
pub fn check_daemon_running(socket_path: &Path, pid_path: &Path) -> anyhow::Result<Option<u32>> {
    // Check PID file first
    if let Some(pid) = read_pid(pid_path) {
        if is_process_alive(pid) {
            // Process exists, check if socket is accessible
            if socket_path.exists() {
                return Ok(Some(pid));
            }
            // Process exists but socket doesn't - unusual state, but treat as running
            return Ok(Some(pid));
        }
        // Process is dead, clean up stale files
        cleanup_stale_files(socket_path, pid_path)?;
    } else if socket_path.exists() {
        // No PID file but socket exists - clean up orphaned socket
        let _ = fs::remove_file(socket_path);
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_process_is_alive() {
        let pid = std::process::id();
        assert!(is_process_alive(pid));
    }

    #[test]
    fn test_invalid_pid_not_alive() {
        // Use a very high but valid PID that's unlikely to exist
        // Note: u32::MAX becomes -1 when cast to i32, which is special for kill()
        // PID 4194304 (2^22) is above typical PID limits on most systems
        assert!(!is_process_alive(4194304));
    }
}
