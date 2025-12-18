/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Client for communicating with the pyrefly daemon.

use std::path::Path;
use std::time::Duration;

use anyhow::bail;
use anyhow::Context;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::time::timeout;

#[cfg(unix)]
use tokio::net::UnixStream;

#[cfg(windows)]
use tokio::net::windows::named_pipe::ClientOptions;

use super::protocol::DaemonRequest;
use super::protocol::DaemonResponse;

/// Send a request to the daemon and receive a response.
pub async fn send_request(
    socket_path: &Path,
    request: &DaemonRequest,
    request_timeout: Duration,
) -> anyhow::Result<DaemonResponse> {
    let result = timeout(request_timeout, send_request_inner(socket_path, request)).await;

    match result {
        Ok(inner_result) => inner_result,
        Err(_) => bail!("Request timed out after {:?}", request_timeout),
    }
}

#[cfg(unix)]
async fn send_request_inner(
    socket_path: &Path,
    request: &DaemonRequest,
) -> anyhow::Result<DaemonResponse> {
    // Connect to daemon
    let mut stream = UnixStream::connect(socket_path)
        .await
        .context("Failed to connect to daemon. Is it running?")?;

    send_request_generic(&mut stream, request).await
}

#[cfg(windows)]
async fn send_request_inner(
    pipe_path: &Path,
    request: &DaemonRequest,
) -> anyhow::Result<DaemonResponse> {
    // Connect to named pipe
    let mut pipe = ClientOptions::new()
        .open(pipe_path)
        .context("Failed to connect to daemon. Is it running?")?;

    send_request_generic(&mut pipe, request).await
}

/// Generic request sender that works with any AsyncRead + AsyncWrite stream.
async fn send_request_generic<S>(
    stream: &mut S,
    request: &DaemonRequest,
) -> anyhow::Result<DaemonResponse>
where
    S: AsyncReadExt + AsyncWriteExt + Unpin,
{
    // Serialize request
    let json = serde_json::to_vec(request).context("Failed to serialize request")?;

    // Write length prefix (4 bytes, big-endian)
    let len = json.len() as u32;
    stream
        .write_all(&len.to_be_bytes())
        .await
        .context("Failed to write request length")?;

    // Write JSON payload
    stream
        .write_all(&json)
        .await
        .context("Failed to write request")?;
    stream.flush().await.context("Failed to flush request")?;

    // Read response length
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .context("Failed to read response length")?;
    let len = u32::from_be_bytes(len_buf) as usize;

    // Sanity check on response size (max 100MB)
    if len > 100 * 1024 * 1024 {
        bail!("Response too large: {} bytes", len);
    }

    // Read response payload
    let mut buf = vec![0u8; len];
    stream
        .read_exact(&mut buf)
        .await
        .context("Failed to read response")?;

    // Deserialize response
    let response: DaemonResponse =
        serde_json::from_slice(&buf).context("Failed to deserialize response")?;

    Ok(response)
}

/// Check if daemon is reachable by sending a ping.
#[allow(dead_code)]
pub async fn ping_daemon(socket_path: &Path, timeout_duration: Duration) -> bool {
    matches!(
        send_request(socket_path, &DaemonRequest::Ping, timeout_duration).await,
        Ok(DaemonResponse::Pong)
    )
}

#[cfg(test)]
mod tests {
    // Client tests require a running server, so they are integration tests
}
