/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Daemon lifecycle management: start, stop, restart, and status operations.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::bail;
use anyhow::Context;

use super::client::send_request;
use super::pid::check_daemon_running;
use super::pid::cleanup_stale_files;
use super::pid::is_process_alive;
use super::protocol::DaemonRequest;
use super::protocol::DaemonResponse;

/// Default timeout for daemon operations.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Shutdown wait timeout.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Get the socket/pipe path for a project root.
///
/// On Unix: Prefers `.pyrefly/daemon.sock` in the project root.
/// Falls back to `~/.pyrefly/daemon/<hash>.sock` if project-local path isn't usable.
///
/// On Windows: Uses named pipe `\\.\pipe\pyrefly-daemon-<hash>`.
#[cfg(unix)]
pub fn get_socket_path(project_root: &Path) -> PathBuf {
    let project_socket = project_root.join(".pyrefly").join("daemon.sock");

    // Try to use project-local path
    let pyrefly_dir = project_root.join(".pyrefly");
    if pyrefly_dir.exists() || std::fs::create_dir_all(&pyrefly_dir).is_ok() {
        return project_socket;
    }

    // Fall back to user-level directory
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let hash = hash_path(project_root);
    home.join(".pyrefly")
        .join("daemon")
        .join(format!("{}.sock", hash))
}

#[cfg(windows)]
pub fn get_socket_path(project_root: &Path) -> PathBuf {
    // On Windows, use named pipes which have a special path format
    let hash = hash_path(project_root);
    PathBuf::from(format!(r"\\.\pipe\pyrefly-daemon-{}", hash))
}

/// Get the directory for PID and log files on Windows.
#[cfg(windows)]
pub fn get_state_dir(project_root: &Path) -> PathBuf {
    let pyrefly_dir = project_root.join(".pyrefly");
    if pyrefly_dir.exists() || std::fs::create_dir_all(&pyrefly_dir).is_ok() {
        return pyrefly_dir;
    }
    // Fall back to user-level directory
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("C:\\temp"));
    let hash = hash_path(project_root);
    home.join(".pyrefly").join("daemon").join(hash)
}

/// Get the PID file path corresponding to a socket path.
#[cfg(unix)]
pub fn get_pid_path(socket_path: &Path) -> PathBuf {
    socket_path.with_extension("pid")
}

#[cfg(windows)]
pub fn get_pid_path(socket_path: &Path) -> PathBuf {
    // On Windows, socket_path is a named pipe like \\.\pipe\pyrefly-daemon-<hash>
    // Extract the hash and create a proper file path
    let pipe_name = socket_path.to_string_lossy();
    let hash = pipe_name.strip_prefix(r"\\.\pipe\pyrefly-daemon-").unwrap_or("default");
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("C:\\temp"));
    home.join(".pyrefly")
        .join("daemon")
        .join(format!("{}.pid", hash))
}

/// Get the log file path corresponding to a socket path.
#[cfg(unix)]
pub fn get_log_path(socket_path: &Path) -> PathBuf {
    socket_path.with_extension("log")
}

#[cfg(windows)]
pub fn get_log_path(socket_path: &Path) -> PathBuf {
    // On Windows, socket_path is a named pipe like \\.\pipe\pyrefly-daemon-<hash>
    // Extract the hash and create a proper file path
    let pipe_name = socket_path.to_string_lossy();
    let hash = pipe_name.strip_prefix(r"\\.\pipe\pyrefly-daemon-").unwrap_or("default");
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("C:\\temp"));
    home.join(".pyrefly")
        .join("daemon")
        .join(format!("{}.log", hash))
}

/// Hash a path to create a unique identifier.
fn hash_path(path: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Resolve project root from optional path or current directory.
pub fn resolve_project_root(project_root: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    match project_root {
        Some(path) => {
            let path = path.canonicalize().context("Invalid project root path")?;
            if !path.is_dir() {
                bail!("Project root must be a directory: {}", path.display());
            }
            Ok(path)
        }
        None => std::env::current_dir().context("Failed to get current directory"),
    }
}

/// Resolve socket path from optional override or project root.
pub fn resolve_socket_path(
    socket: Option<PathBuf>,
    project_root: &Path,
) -> anyhow::Result<PathBuf> {
    match socket {
        Some(path) => Ok(path),
        None => Ok(get_socket_path(project_root)),
    }
}

/// Check if a daemon is already running and return its PID if so.
pub fn get_running_daemon(socket_path: &Path) -> anyhow::Result<Option<u32>> {
    let pid_path = get_pid_path(socket_path);
    check_daemon_running(socket_path, &pid_path)
}

/// Stop a running daemon gracefully.
pub async fn stop_daemon(socket_path: &Path, force: bool) -> anyhow::Result<()> {
    let pid_path = get_pid_path(socket_path);

    // Check if daemon is running
    let Some(pid) = check_daemon_running(socket_path, &pid_path)? else {
        eprintln!("No daemon is running");
        return Ok(());
    };

    // Try graceful shutdown first
    match send_request(socket_path, &DaemonRequest::Shutdown, DEFAULT_TIMEOUT).await {
        Ok(DaemonResponse::Ok) => {
            // Wait for daemon to exit
            if wait_for_shutdown(pid, SHUTDOWN_TIMEOUT) {
                cleanup_stale_files(socket_path, &pid_path)?;
                eprintln!("Daemon stopped gracefully (pid: {})", pid);
                return Ok(());
            }
            if !force {
                bail!("Daemon did not exit within timeout. Use --force to kill it.");
            }
        }
        Err(e) if !force => {
            bail!("Failed to send shutdown request: {}. Use --force to kill it.", e);
        }
        _ => {}
    }

    // Force kill if requested
    if force {
        force_kill(pid)?;
        cleanup_stale_files(socket_path, &pid_path)?;
        eprintln!("Daemon killed forcefully (pid: {})", pid);
    }

    Ok(())
}

/// Wait for a process to exit.
fn wait_for_shutdown(pid: u32, timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if !is_process_alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

/// Force kill a process.
#[cfg(unix)]
fn force_kill(pid: u32) -> anyhow::Result<()> {
    unsafe {
        if libc::kill(pid as i32, libc::SIGKILL) != 0 {
            bail!("Failed to kill process {}", pid);
        }
    }
    Ok(())
}

#[cfg(windows)]
fn force_kill(pid: u32) -> anyhow::Result<()> {
    use winapi::um::handleapi::CloseHandle;
    use winapi::um::processthreadsapi::{OpenProcess, TerminateProcess};
    use winapi::um::winnt::PROCESS_TERMINATE;

    unsafe {
        let handle = OpenProcess(PROCESS_TERMINATE, 0, pid);
        if handle.is_null() {
            bail!("Failed to open process {}", pid);
        }
        let result = TerminateProcess(handle, 1);
        CloseHandle(handle);
        if result == 0 {
            bail!("Failed to terminate process {}", pid);
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn force_kill(_pid: u32) -> anyhow::Result<()> {
    bail!("Force kill not supported on this platform");
}

/// Get daemon status.
pub async fn get_daemon_status(socket_path: &Path) -> anyhow::Result<Option<DaemonResponse>> {
    let pid_path = get_pid_path(socket_path);

    // Check if daemon is running
    if check_daemon_running(socket_path, &pid_path)?.is_none() {
        return Ok(None);
    }

    // Get status from daemon
    match send_request(socket_path, &DaemonRequest::Status, DEFAULT_TIMEOUT).await {
        Ok(response) => Ok(Some(response)),
        Err(e) => {
            // Daemon might have just died
            let _ = cleanup_stale_files(socket_path, &pid_path);
            Err(e)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(unix)]
    fn test_get_socket_path() {
        let project = PathBuf::from("/tmp/test-project");
        let socket = get_socket_path(&project);
        assert!(socket.to_string_lossy().contains("daemon.sock"));
    }

    #[test]
    #[cfg(windows)]
    fn test_get_socket_path() {
        let project = PathBuf::from("C:\\temp\\test-project");
        let socket = get_socket_path(&project);
        // Windows uses named pipes
        assert!(socket.to_string_lossy().contains("pyrefly-daemon-"));
    }

    #[test]
    #[cfg(unix)]
    fn test_get_pid_path() {
        let socket = PathBuf::from("/tmp/.pyrefly/daemon.sock");
        let pid = get_pid_path(&socket);
        assert_eq!(pid, PathBuf::from("/tmp/.pyrefly/daemon.pid"));
    }

    #[test]
    #[cfg(windows)]
    fn test_get_pid_path() {
        let socket = PathBuf::from(r"\\.\pipe\pyrefly-daemon-abc123");
        let pid = get_pid_path(&socket);
        // Windows PID files go in user's home directory
        assert!(pid.to_string_lossy().contains("abc123.pid"));
    }

    #[test]
    fn test_hash_path_deterministic() {
        let path = PathBuf::from("/some/project");
        let hash1 = hash_path(&path);
        let hash2 = hash_path(&path);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_path_different_paths() {
        let path1 = PathBuf::from("/project1");
        let path2 = PathBuf::from("/project2");
        assert_ne!(hash_path(&path1), hash_path(&path2));
    }
}
