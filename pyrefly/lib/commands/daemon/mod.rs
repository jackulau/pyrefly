/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Daemon mode for persistent background type checking.
//!
//! The daemon maintains type-checking state in memory and automatically watches
//! for file changes, enabling fast incremental checks.
//!
//! # Usage
//!
//! ```sh
//! # Start daemon for current project
//! pyrefly daemon start
//!
//! # Check files via daemon
//! pyrefly daemon check src/main.py
//!
//! # Get daemon status
//! pyrefly daemon status
//!
//! # Stop daemon
//! pyrefly daemon stop
//! ```

mod client;
mod lifecycle;
mod pid;
mod protocol;
mod server;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anstream::eprintln;
use anyhow::bail;
use clap::Parser;
use clap::Subcommand;
use pyrefly_config::args::ConfigOverrideArgs;
use serde_json;

use self::client::send_request;
use self::lifecycle::get_running_daemon;
use self::lifecycle::resolve_project_root;
use self::lifecycle::resolve_socket_path;
use self::lifecycle::stop_daemon;
use self::protocol::DaemonRequest;
use self::protocol::DaemonResponse;
use self::server::run_daemon_server;
use self::server::DaemonState;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;

/// Default timeout for daemon requests.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Arguments for the daemon command.
#[derive(Debug, Clone, Parser)]
pub struct DaemonArgs {
    /// Daemon subcommand.
    #[command(subcommand)]
    pub action: DaemonAction,
}

/// Daemon subcommands.
#[derive(Debug, Clone, Subcommand)]
pub enum DaemonAction {
    /// Start the daemon in the background.
    Start {
        /// Project root directory (defaults to current directory).
        #[arg(long, short)]
        project_root: Option<PathBuf>,

        /// Run in foreground (don't daemonize).
        #[arg(long)]
        foreground: bool,

        /// Socket path override.
        #[arg(long)]
        socket: Option<PathBuf>,

        /// Which files to check.
        #[command(flatten)]
        files: FilesArgs,

        /// Configuration override options.
        #[command(flatten, next_help_heading = "Config Overrides")]
        config_override: ConfigOverrideArgs,
    },

    /// Stop a running daemon.
    Stop {
        /// Project root directory (defaults to current directory).
        #[arg(long, short)]
        project_root: Option<PathBuf>,

        /// Socket path override.
        #[arg(long)]
        socket: Option<PathBuf>,

        /// Force kill if graceful shutdown fails.
        #[arg(long)]
        force: bool,
    },

    /// Restart the daemon.
    Restart {
        /// Project root directory (defaults to current directory).
        #[arg(long, short)]
        project_root: Option<PathBuf>,

        /// Socket path override.
        #[arg(long)]
        socket: Option<PathBuf>,

        /// Which files to check.
        #[command(flatten)]
        files: FilesArgs,

        /// Configuration override options.
        #[command(flatten, next_help_heading = "Config Overrides")]
        config_override: ConfigOverrideArgs,
    },

    /// Show daemon status.
    Status {
        /// Project root directory (defaults to current directory).
        #[arg(long, short)]
        project_root: Option<PathBuf>,

        /// Socket path override.
        #[arg(long)]
        socket: Option<PathBuf>,

        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Check files via daemon.
    Check {
        /// Files to check.
        files: Vec<PathBuf>,

        /// Project root directory (defaults to current directory).
        #[arg(long, short)]
        project_root: Option<PathBuf>,

        /// Socket path override.
        #[arg(long)]
        socket: Option<PathBuf>,

        /// Check all reachable modules.
        #[arg(long, short = 'a')]
        check_all: bool,
    },
}

impl DaemonArgs {
    /// Run the daemon command.
    pub async fn run(self) -> anyhow::Result<CommandExitStatus> {
        match self.action {
            DaemonAction::Start {
                project_root,
                foreground,
                socket,
                files,
                config_override,
            } => run_start(project_root, foreground, socket, files, config_override).await,

            DaemonAction::Stop {
                project_root,
                socket,
                force,
            } => run_stop(project_root, socket, force).await,

            DaemonAction::Restart {
                project_root,
                socket,
                files,
                config_override,
            } => run_restart(project_root, socket, files, config_override).await,

            DaemonAction::Status {
                project_root,
                socket,
                json,
            } => run_status(project_root, socket, json).await,

            DaemonAction::Check {
                files,
                project_root,
                socket,
                check_all,
            } => run_check(files, project_root, socket, check_all).await,
        }
    }
}

/// Run the start command.
async fn run_start(
    project_root: Option<PathBuf>,
    foreground: bool,
    socket: Option<PathBuf>,
    files: FilesArgs,
    config_override: ConfigOverrideArgs,
) -> anyhow::Result<CommandExitStatus> {
    let project_root = resolve_project_root(project_root)?;
    let socket_path = resolve_socket_path(socket, &project_root)?;

    // Check if daemon already running
    if let Some(pid) = get_running_daemon(&socket_path)? {
        eprintln!("Daemon already running (pid: {})", pid);
        return Ok(CommandExitStatus::Success);
    }

    // Resolve files and config
    config_override.validate()?;
    let (files_to_check, config_finder) = files.resolve(config_override)?;

    eprintln!(
        "Starting daemon for project: {}",
        project_root.display()
    );

    if foreground {
        // Run in foreground
        let state = Arc::new(DaemonState::new(
            config_finder,
            project_root,
            files_to_check,
        ));
        run_daemon_server(state, socket_path).await?;
    } else {
        // Daemonize
        daemonize_and_start(project_root, socket_path, files_to_check, config_finder)?;
        eprintln!("Daemon started in background");
    }

    Ok(CommandExitStatus::Success)
}

/// Daemonize the process and start the daemon server.
#[cfg(unix)]
fn daemonize_and_start(
    project_root: PathBuf,
    socket_path: PathBuf,
    files_to_check: Box<dyn pyrefly_util::includes::Includes>,
    config_finder: crate::config::finder::ConfigFinder,
) -> anyhow::Result<()> {
    use std::os::unix::process::CommandExt;

    // Get the path to the current executable
    let exe = std::env::current_exe()?;

    // Re-exec ourselves with --foreground
    // This is simpler than using fork() directly
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("daemon")
        .arg("start")
        .arg("--foreground")
        .arg("--project-root")
        .arg(&project_root);

    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    cmd.arg("--socket").arg(&socket_path);

    // Detach from terminal
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    // Use setsid on Unix to create a new session
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }

    // Spawn the daemon
    cmd.spawn()?;

    // Wait a moment for the daemon to start
    std::thread::sleep(Duration::from_millis(500));

    // Verify daemon started
    if get_running_daemon(&socket_path)?.is_none() {
        bail!("Failed to start daemon. Check log file for details.");
    }

    // Drop the files_to_check and config_finder since they were only needed
    // to validate the configuration before daemonizing
    drop(files_to_check);
    drop(config_finder);

    Ok(())
}

#[cfg(windows)]
fn daemonize_and_start(
    project_root: PathBuf,
    socket_path: PathBuf,
    files_to_check: Box<dyn pyrefly_util::includes::Includes>,
    config_finder: crate::config::finder::ConfigFinder,
) -> anyhow::Result<()> {
    use std::os::windows::process::CommandExt;

    const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
    const DETACHED_PROCESS: u32 = 0x00000008;

    // Get the path to the current executable
    let exe = std::env::current_exe()?;

    // Re-exec ourselves with --foreground
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("daemon")
        .arg("start")
        .arg("--foreground")
        .arg("--project-root")
        .arg(&project_root)
        .arg("--socket")
        .arg(&socket_path);

    // Detach from terminal on Windows
    cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS);
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    // Spawn the daemon
    cmd.spawn()?;

    // Wait a moment for the daemon to start
    std::thread::sleep(Duration::from_millis(500));

    // Verify daemon started
    if get_running_daemon(&socket_path)?.is_none() {
        bail!("Failed to start daemon. Check log file for details.");
    }

    // Drop the files_to_check and config_finder since they were only needed
    // to validate the configuration before daemonizing
    drop(files_to_check);
    drop(config_finder);

    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn daemonize_and_start(
    _project_root: PathBuf,
    _socket_path: PathBuf,
    _files_to_check: Box<dyn pyrefly_util::includes::Includes>,
    _config_finder: crate::config::finder::ConfigFinder,
) -> anyhow::Result<()> {
    bail!("Background daemon mode is not supported on this platform. Use --foreground instead.");
}

/// Run the stop command.
async fn run_stop(
    project_root: Option<PathBuf>,
    socket: Option<PathBuf>,
    force: bool,
) -> anyhow::Result<CommandExitStatus> {
    let project_root = resolve_project_root(project_root)?;
    let socket_path = resolve_socket_path(socket, &project_root)?;

    stop_daemon(&socket_path, force).await?;
    Ok(CommandExitStatus::Success)
}

/// Run the restart command.
async fn run_restart(
    project_root: Option<PathBuf>,
    socket: Option<PathBuf>,
    files: FilesArgs,
    config_override: ConfigOverrideArgs,
) -> anyhow::Result<CommandExitStatus> {
    let project_root_resolved = resolve_project_root(project_root.clone())?;
    let socket_path = resolve_socket_path(socket.clone(), &project_root_resolved)?;

    // Stop existing daemon if running
    let _ = stop_daemon(&socket_path, false).await;

    // Start new daemon
    run_start(project_root, false, socket, files, config_override).await
}

/// Run the status command.
async fn run_status(
    project_root: Option<PathBuf>,
    socket: Option<PathBuf>,
    json: bool,
) -> anyhow::Result<CommandExitStatus> {
    let project_root = resolve_project_root(project_root)?;
    let socket_path = resolve_socket_path(socket, &project_root)?;

    match lifecycle::get_daemon_status(&socket_path).await? {
        Some(DaemonResponse::Status {
            pid,
            uptime_secs,
            indexed_modules,
            memory_usage_mb,
            project_root,
        }) => {
            if json {
                let status = serde_json::json!({
                    "running": true,
                    "pid": pid,
                    "uptime_secs": uptime_secs,
                    "indexed_modules": indexed_modules,
                    "memory_usage_mb": memory_usage_mb,
                    "project_root": project_root,
                    "socket": socket_path,
                });
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                eprintln!("Daemon is running");
                eprintln!("  PID: {}", pid);
                eprintln!("  Uptime: {} seconds", uptime_secs);
                eprintln!("  Indexed modules: {}", indexed_modules);
                eprintln!("  Memory usage: {:.1} MB", memory_usage_mb);
                eprintln!("  Project root: {}", project_root.display());
                eprintln!("  Socket: {}", socket_path.display());
            }
            Ok(CommandExitStatus::Success)
        }
        Some(DaemonResponse::Error { message }) => {
            eprintln!("Daemon error: {}", message);
            Ok(CommandExitStatus::UserError)
        }
        None => {
            if json {
                let status = serde_json::json!({
                    "running": false,
                    "socket": socket_path,
                });
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                eprintln!("No daemon is running for this project");
                eprintln!("  Socket: {}", socket_path.display());
            }
            Ok(CommandExitStatus::Success)
        }
        Some(other) => {
            eprintln!("Unexpected response: {:?}", other);
            Ok(CommandExitStatus::UserError)
        }
    }
}

/// Run the check command.
async fn run_check(
    files: Vec<PathBuf>,
    project_root: Option<PathBuf>,
    socket: Option<PathBuf>,
    check_all: bool,
) -> anyhow::Result<CommandExitStatus> {
    let project_root = resolve_project_root(project_root)?;
    let socket_path = resolve_socket_path(socket, &project_root)?;

    // Check if daemon is running
    if get_running_daemon(&socket_path)?.is_none() {
        bail!(
            "No daemon is running. Start one with: pyrefly daemon start"
        );
    }

    // Send check request
    let request = DaemonRequest::Check { files, check_all };
    let response = send_request(&socket_path, &request, DEFAULT_TIMEOUT).await?;

    match response {
        DaemonResponse::CheckResult {
            errors,
            error_count,
            checked_files,
            duration_ms,
        } => {
            // Print errors
            for error in &errors {
                let severity = &error.severity;
                let path = error.path.display();
                let line = error.line;
                let column = error.column;
                let message = &error.message;
                let code = error.code.as_deref().unwrap_or("");

                eprintln!("{} {}:{}:{}: {} [{}]", severity, path, line, column, message, code);
            }

            // Print summary
            if error_count > 0 {
                eprintln!();
                eprintln!(
                    "Found {} error(s) in {} file(s) ({} ms)",
                    error_count, checked_files, duration_ms
                );
                Ok(CommandExitStatus::UserError)
            } else {
                eprintln!(
                    "No errors in {} file(s) ({} ms)",
                    checked_files, duration_ms
                );
                Ok(CommandExitStatus::Success)
            }
        }
        DaemonResponse::Error { message } => {
            eprintln!("Error: {}", message);
            Ok(CommandExitStatus::UserError)
        }
        other => {
            eprintln!("Unexpected response: {:?}", other);
            Ok(CommandExitStatus::UserError)
        }
    }
}
