/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Daemon server implementation.
//!
//! The server maintains persistent type-checking state and handles requests
//! from clients over a Unix domain socket (Unix) or named pipe (Windows).

use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::RwLock as StdRwLock;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use pyrefly_build::handle::Handle;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::includes::Includes;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tracing::debug;
use tracing::error;
use tracing::info;

#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

#[cfg(windows)]
use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeServer, ServerOptions};

use super::lifecycle::get_log_path;
use super::lifecycle::get_pid_path;
use super::pid::remove_pid;
use super::pid::write_pid;
use super::protocol::DaemonRequest;
use super::protocol::DaemonResponse;
use super::protocol::SerializableError;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::state::require::Require;
use crate::state::state::State;

/// Daemon server state.
pub struct DaemonState {
    /// Core type checking state.
    state: State,
    /// When the daemon started.
    start_time: Instant,
    /// Project root being watched.
    project_root: PathBuf,
    /// Files to include in checks (wrapped in Mutex for thread safety).
    files_to_check: std::sync::Mutex<Box<dyn Includes>>,
    /// Shutdown signal.
    shutdown: AtomicBool,
    /// Number of indexed modules.
    indexed_modules: StdRwLock<usize>,
}

// Safety: DaemonState is Send + Sync because all fields are thread-safe.
// The Mutex protects the non-Send Box<dyn Includes>.
unsafe impl Send for DaemonState {}
unsafe impl Sync for DaemonState {}

impl DaemonState {
    /// Create a new daemon state.
    pub fn new(
        config_finder: ConfigFinder,
        project_root: PathBuf,
        files_to_check: Box<dyn Includes>,
    ) -> Self {
        Self {
            state: State::new(config_finder),
            start_time: Instant::now(),
            project_root,
            files_to_check: std::sync::Mutex::new(files_to_check),
            shutdown: AtomicBool::new(false),
            indexed_modules: StdRwLock::new(0),
        }
    }

    /// Check if shutdown has been requested.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Request shutdown.
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

/// Run the daemon server.
pub async fn run_daemon_server(
    daemon_state: Arc<DaemonState>,
    socket_path: PathBuf,
) -> anyhow::Result<()> {
    let pid_path = get_pid_path(&socket_path);
    let log_path = get_log_path(&socket_path);

    // Set up logging
    setup_logging(&log_path)?;

    // Write PID file
    write_pid(&pid_path)?;
    info!(
        "Daemon starting (pid: {}, socket: {})",
        std::process::id(),
        socket_path.display()
    );

    // Set up signal handling for graceful shutdown
    let shutdown_state = daemon_state.clone();
    tokio::spawn(async move {
        if let Ok(()) = tokio::signal::ctrl_c().await {
            info!("Received shutdown signal");
            shutdown_state.request_shutdown();
        }
    });

    // Start file watcher in dedicated thread (uses blocking I/O)
    let watcher_state = daemon_state.clone();
    let watcher_handle = std::thread::spawn(move || {
        if let Err(e) = run_file_watcher_blocking(watcher_state) {
            error!("File watcher error: {}", e);
        }
    });

    // Initial indexing
    {
        info!("Starting initial indexing...");
        let files = {
            let files_to_check = daemon_state.files_to_check.lock().unwrap();
            match daemon_state
                .state
                .config_finder()
                .checkpoint(files_to_check.files())
            {
                Ok(files) => files,
                Err(e) => {
                    // No files found is not fatal for daemon - just start with empty index
                    info!("No files to index: {}", e);
                    Vec::new()
                }
            }
        };
        if !files.is_empty() {
            let handles = create_handles(&files, daemon_state.state.config_finder());
            let mut transaction =
                daemon_state
                    .state
                    .new_committable_transaction(Require::Exports, None);
            transaction.as_mut().run(&handles, Require::Exports);
            daemon_state.state.commit_transaction(transaction);
        }
        *daemon_state.indexed_modules.write().unwrap() = files.len();
        info!("Initial indexing complete ({} modules)", files.len());
    }

    // Platform-specific server loop
    #[cfg(unix)]
    {
        run_unix_server(daemon_state.clone(), &socket_path).await?;
    }

    #[cfg(windows)]
    {
        run_windows_server(daemon_state.clone(), &socket_path).await?;
    }

    // Cleanup
    info!("Daemon shutting down");
    // The watcher thread will exit when it sees the shutdown flag
    let _ = watcher_handle.join();
    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&socket_path);
    }
    remove_pid(&pid_path);

    Ok(())
}

/// Unix-specific server loop using Unix domain sockets.
#[cfg(unix)]
async fn run_unix_server(daemon_state: Arc<DaemonState>, socket_path: &Path) -> anyhow::Result<()> {
    // Remove existing socket if present
    let _ = std::fs::remove_file(socket_path);

    // Create socket directory if needed
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create socket directory")?;
    }

    // Bind Unix socket
    let listener = UnixListener::bind(socket_path).context("Failed to bind Unix socket")?;
    info!("Daemon listening on {}", socket_path.display());

    // Accept connections
    loop {
        if daemon_state.is_shutdown() {
            break;
        }

        // Use timeout to check shutdown periodically
        let accept_result =
            tokio::time::timeout(Duration::from_millis(500), listener.accept()).await;

        match accept_result {
            Ok(Ok((stream, _))) => {
                let state = daemon_state.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection_unix(stream, state).await {
                        error!("Connection error: {}", e);
                    }
                });
            }
            Ok(Err(e)) => {
                error!("Accept error: {}", e);
            }
            Err(_) => {
                // Timeout, continue to check shutdown
                continue;
            }
        }
    }

    Ok(())
}

/// Windows-specific server loop using named pipes.
#[cfg(windows)]
async fn run_windows_server(
    daemon_state: Arc<DaemonState>,
    pipe_path: &Path,
) -> anyhow::Result<()> {
    let pipe_name = pipe_path.as_os_str();
    info!("Daemon listening on {}", pipe_path.display());

    // Create the first pipe instance
    let mut server = ServerOptions::new()
        .first_pipe_instance(true)
        .create(pipe_name)
        .context("Failed to create named pipe")?;

    loop {
        if daemon_state.is_shutdown() {
            break;
        }

        // Wait for a client to connect with timeout
        let connect_result =
            tokio::time::timeout(Duration::from_millis(500), server.connect()).await;

        match connect_result {
            Ok(Ok(())) => {
                // Client connected - create a new pipe instance for the next client FIRST
                // before spawning the handler (to avoid missing connections)
                let connected_server = server;
                server = match ServerOptions::new()
                    .first_pipe_instance(false)
                    .create(pipe_name)
                {
                    Ok(s) => s,
                    Err(e) => {
                        error!("Failed to create new pipe instance: {}", e);
                        // Try to recover by creating first instance
                        ServerOptions::new()
                            .first_pipe_instance(true)
                            .create(pipe_name)
                            .context("Failed to recover pipe instance")?
                    }
                };

                // Now spawn handler for the connected client
                let state = daemon_state.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection_windows(connected_server, state).await {
                        error!("Connection error: {}", e);
                    }
                });
            }
            Ok(Err(e)) => {
                error!("Pipe connect error: {}", e);
                // Try to recreate the pipe
                tokio::time::sleep(Duration::from_millis(100)).await;
                server = ServerOptions::new()
                    .first_pipe_instance(true)
                    .create(pipe_name)
                    .context("Failed to recreate named pipe")?;
            }
            Err(_) => {
                // Timeout, continue to check shutdown
                // Keep the same server instance - don't recreate
                continue;
            }
        }
    }

    Ok(())
}

/// Handle a client connection on Unix.
#[cfg(unix)]
async fn handle_connection_unix(
    mut stream: UnixStream,
    state: Arc<DaemonState>,
) -> anyhow::Result<()> {
    handle_connection_generic(&mut stream, state).await
}

/// Handle a client connection on Windows.
#[cfg(windows)]
async fn handle_connection_windows(
    mut pipe: NamedPipeServer,
    state: Arc<DaemonState>,
) -> anyhow::Result<()> {
    handle_connection_generic(&mut pipe, state).await
}

/// Generic connection handler that works with any AsyncRead + AsyncWrite stream.
async fn handle_connection_generic<S>(stream: &mut S, state: Arc<DaemonState>) -> anyhow::Result<()>
where
    S: AsyncReadExt + AsyncWriteExt + Unpin,
{
    // Read request length
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    // Sanity check
    if len > 10 * 1024 * 1024 {
        anyhow::bail!("Request too large: {} bytes", len);
    }

    // Read request
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    let request: DaemonRequest = serde_json::from_slice(&buf)?;

    debug!("Received request: {:?}", request);

    // Process request
    let response = match request {
        DaemonRequest::Check { files, check_all } => handle_check(&state, files, check_all),
        DaemonRequest::Status => handle_status(&state),
        DaemonRequest::Shutdown => {
            state.request_shutdown();
            DaemonResponse::Ok
        }
        DaemonRequest::Ping => DaemonResponse::Pong,
        DaemonRequest::Reindex => handle_reindex(&state),
    };

    // Write response
    let json = serde_json::to_vec(&response)?;
    let len = json.len() as u32;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(&json).await?;
    stream.flush().await?;

    Ok(())
}

/// Handle a check request.
fn handle_check(
    state: &DaemonState,
    files: Vec<PathBuf>,
    check_all: bool,
) -> DaemonResponse {
    let start = Instant::now();

    // Determine files to check
    let files_result = if files.is_empty() {
        let files_to_check = state.files_to_check.lock().unwrap();
        files_to_check.files()
    } else {
        Ok(files)
    };

    // Expand file list
    let expanded_files = match state.state.config_finder().checkpoint(files_result) {
        Ok(files) => files,
        Err(e) => {
            return DaemonResponse::Error {
                message: format!("Failed to expand files: {}", e),
            };
        }
    };

    if expanded_files.is_empty() {
        return DaemonResponse::CheckResult {
            errors: vec![],
            error_count: 0,
            checked_files: 0,
            duration_ms: start.elapsed().as_millis() as u64,
        };
    }

    // Create handles
    let handles = create_handles(&expanded_files, state.state.config_finder());

    // Run type check
    let mut transaction = state
        .state
        .new_committable_transaction(Require::Errors, None);
    transaction.as_mut().run(&handles, Require::Errors);

    // Collect errors
    let loads = if check_all {
        transaction.as_mut().get_all_errors()
    } else {
        transaction.as_mut().get_errors(&handles)
    };
    let errors_result = loads.collect_errors_with_baseline(None, &state.project_root);

    // Commit transaction
    state.state.commit_transaction(transaction);

    // Convert errors to serializable format
    let errors = convert_errors(&errors_result.shown, &state.project_root);
    let error_count = errors.len();
    let checked_files = expanded_files.len();

    DaemonResponse::CheckResult {
        errors,
        error_count,
        checked_files,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

/// Handle a status request.
fn handle_status(state: &DaemonState) -> DaemonResponse {
    let uptime = state.start_time.elapsed();
    let indexed_modules = *state.indexed_modules.read().unwrap();

    // Get memory usage (rough estimate)
    let memory_usage_mb = get_memory_usage_mb();

    DaemonResponse::Status {
        pid: std::process::id(),
        uptime_secs: uptime.as_secs(),
        indexed_modules,
        memory_usage_mb,
        project_root: state.project_root.clone(),
    }
}

/// Handle a reindex request.
fn handle_reindex(state: &DaemonState) -> DaemonResponse {
    let files = {
        let files_to_check = state.files_to_check.lock().unwrap();
        match state
            .state
            .config_finder()
            .checkpoint(files_to_check.files())
        {
            Ok(files) => files,
            Err(e) => {
                return DaemonResponse::Error {
                    message: format!("Failed to list files: {}", e),
                };
            }
        }
    };

    let handles = create_handles(&files, state.state.config_finder());
    let mut transaction = state
        .state
        .new_committable_transaction(Require::Exports, None);
    transaction.as_mut().run(&handles, Require::Exports);
    state.state.commit_transaction(transaction);

    *state.indexed_modules.write().unwrap() = files.len();

    info!("Reindex complete ({} modules)", files.len());
    DaemonResponse::Ok
}

/// Run the file watcher (blocking version for dedicated thread).
fn run_file_watcher_blocking(state: Arc<DaemonState>) -> anyhow::Result<()> {
    use notify::RecursiveMode;
    use notify::Watcher as NotifyWatcher;
    use std::sync::mpsc::channel;

    let roots = {
        let files_to_check = state.files_to_check.lock().unwrap();
        files_to_check.roots()
    };

    // Create channel for file events
    let (tx, rx) = channel();

    // Create watcher
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            let _ = tx.send(event);
        }
    })?;

    // Watch all roots
    for root in &roots {
        if let Err(e) = watcher.watch(root, RecursiveMode::Recursive) {
            error!("Failed to watch {}: {}", root.display(), e);
        }
    }

    info!("File watcher started for {} roots", roots.len());

    loop {
        if state.is_shutdown() {
            break;
        }

        // Wait for events with timeout
        let events = match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(event) => {
                // Collect any additional events that arrived
                let mut events = vec![event];
                while let Ok(e) = rx.try_recv() {
                    events.push(e);
                }
                events
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                error!("File watcher channel disconnected");
                break;
            }
        };

        // Convert notify events to CategorizedEvents
        let mut created = Vec::new();
        let mut modified = Vec::new();
        let mut removed = Vec::new();

        for event in events {
            for path in event.paths {
                match event.kind {
                    notify::EventKind::Create(_) => created.push(path),
                    notify::EventKind::Modify(_) => modified.push(path),
                    notify::EventKind::Remove(_) => removed.push(path),
                    _ => {}
                }
            }
        }

        if created.is_empty() && modified.is_empty() && removed.is_empty() {
            continue;
        }

        debug!(
            "File changes detected: {} created, {} modified, {} removed",
            created.len(),
            modified.len(),
            removed.len()
        );

        let categorized = CategorizedEvents {
            created,
            modified,
            removed,
            unknown: Vec::new(),
        };

        // Invalidate affected files and re-run to drain the dirty queue
        let mut transaction = state
            .state
            .new_committable_transaction(Require::Exports, None);
        transaction.as_mut().invalidate_events(&categorized);

        // Get handles for all tracked files to re-run after invalidation
        let files = {
            let files_to_check = state.files_to_check.lock().unwrap();
            match state
                .state
                .config_finder()
                .checkpoint(files_to_check.files())
            {
                Ok(files) => files,
                Err(_) => {
                    // No Python files - just update count and continue
                    Vec::new()
                }
            }
        };

        // Run to drain the dirty queue (required after invalidation)
        if !files.is_empty() {
            let handles = create_handles(&files, state.state.config_finder());
            transaction.as_mut().run(&handles, Require::Exports);
        }
        state.state.commit_transaction(transaction);

        *state.indexed_modules.write().unwrap() = files.len();
        info!("Re-indexed {} modules after file changes", files.len());
    }

    Ok(())
}

/// Create handles from file paths.
fn create_handles(files: &[PathBuf], config_finder: &ConfigFinder) -> Vec<Handle> {
    // Collect roots for module name resolution
    let roots: Vec<PathBuf> = files
        .iter()
        .filter_map(|p| p.parent().map(|p| p.to_path_buf()))
        .collect();

    files
        .iter()
        .filter_map(|path| {
            let module_path = pyrefly_python::module_path::ModulePath::filesystem(path.clone());
            let module_name =
                pyrefly_python::module_name::ModuleName::from_path(path, roots.iter())?;
            let python_file = config_finder.python_file(module_name, &module_path);
            let sys_info = python_file.get_sys_info();
            Some(Handle::new(module_name, module_path, sys_info))
        })
        .collect()
}

/// Convert errors to serializable format.
fn convert_errors(errors: &[Error], project_root: &Path) -> Vec<SerializableError> {
    errors
        .iter()
        .map(|e| {
            let display_range = e.display_range();
            SerializableError {
                path: e
                    .path()
                    .as_path()
                    .strip_prefix(project_root)
                    .unwrap_or(e.path().as_path())
                    .to_path_buf(),
                line: display_range.start.line_within_file().get(),
                column: display_range.start.column().get(),
                message: e.msg_header().to_string(),
                severity: e.severity().label().trim().to_string(),
                code: Some(e.error_kind().to_name().to_string()),
            }
        })
        .collect()
}

/// Get current process memory usage in MB.
fn get_memory_usage_mb() -> f64 {
    // Simple cross-platform memory usage estimation
    // On Unix, we can read from /proc/self/statm
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss.parse::<u64>() {
                    // Assume 4KB page size
                    return (pages * 4) as f64 / 1024.0;
                }
            }
        }
    }

    // Fallback: return 0
    0.0
}

/// Set up logging to a file.
fn setup_logging(log_path: &Path) -> anyhow::Result<()> {
    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::EnvFilter;

    // Create log directory
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Open log file
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .context("Failed to open log file")?;

    // Set up tracing subscriber
    let file_layer = fmt::layer()
        .with_writer(file)
        .with_ansi(false)
        .with_target(true);

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .try_init()
        .ok(); // Ignore error if already initialized

    Ok(())
}
