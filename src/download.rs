use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::Deserialize;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

/// A file entry in a HuggingFace model repository.
#[derive(Debug, Deserialize)]
pub struct HfSibling {
    #[serde(rename = "rfilename")]
    pub filename: String,
    pub size: Option<u64>,
}

/// List all files in a HuggingFace repo, returning GGUF files first.
pub async fn hf_list_files(
    repo: &str,
    token: Option<&str>,
    endpoint: &str,
) -> Result<Vec<HfSibling>> {
    let url = format!("{endpoint}/api/models/{repo}");
    let client = build_client(token)?;
    let resp: serde_json::Value = client.get(&url).send().await?.json().await?;
    let siblings: Vec<HfSibling> =
        serde_json::from_value(resp["siblings"].clone()).unwrap_or_default();
    let mut gguf: Vec<_> = siblings
        .into_iter()
        .filter(|s| s.filename.ends_with(".gguf"))
        .collect();
    gguf.sort_by_key(|s| s.filename.clone());
    Ok(gguf)
}

/// Download a single file from HuggingFace, reporting progress via callback.
///
/// `progress_cb(downloaded_bytes, total_bytes_option)`
pub async fn hf_download<F>(
    repo: &str,
    filename: &str,
    dest_dir: &PathBuf,
    token: Option<&str>,
    endpoint: &str,
    mut progress_cb: F,
) -> Result<PathBuf>
where
    F: FnMut(u64, Option<u64>),
{
    let url = format!("{endpoint}/{repo}/resolve/main/{filename}");
    let client = build_client(token)?;
    let resp = client.get(&url).send().await?.error_for_status()?;

    let total = resp.content_length();
    std::fs::create_dir_all(dest_dir)?;
    let dest = dest_dir.join(filename);
    let mut file = tokio::fs::File::create(&dest).await?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream error")?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        progress_cb(downloaded, total);
    }

    file.flush().await?;
    Ok(dest)
}

/// Download from an arbitrary direct URL.
pub async fn download_url<F>(
    url: &str,
    dest_dir: &PathBuf,
    filename: &str,
    mut progress_cb: F,
) -> Result<PathBuf>
where
    F: FnMut(u64, Option<u64>),
{
    let client = reqwest::Client::new();
    let resp = client.get(url).send().await?.error_for_status()?;
    let total = resp.content_length();

    std::fs::create_dir_all(dest_dir)?;
    let dest = dest_dir.join(filename);
    let mut file = tokio::fs::File::create(&dest).await?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream error")?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        progress_cb(downloaded, total);
    }

    file.flush().await?;
    Ok(dest)
}

fn build_client(token: Option<&str>) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder();
    if let Some(tok) = token {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {tok}").parse().context("invalid token")?,
        );
        builder = builder.default_headers(headers);
    }
    Ok(builder.build()?)
}

/// Parse a source string into (repo, filename).
/// Accepts formats:
///   hf:owner/repo/filename.gguf
///   hf:owner/repo  (list files)
///   https://...    (direct URL)
pub enum DownloadSource {
    HuggingFace { repo: String, file: Option<String> },
    DirectUrl { url: String, filename: String },
}

pub fn parse_source(source: &str) -> Result<DownloadSource> {
    if let Some(rest) = source.strip_prefix("hf:") {
        let parts: Vec<&str> = rest.splitn(3, '/').collect();
        match parts.len() {
            2 => Ok(DownloadSource::HuggingFace {
                repo: rest.to_string(),
                file: None,
            }),
            3 => Ok(DownloadSource::HuggingFace {
                repo: format!("{}/{}", parts[0], parts[1]),
                file: Some(parts[2].to_string()),
            }),
            _ => bail!("invalid hf source: {source}"),
        }
    } else if source.starts_with("https://") || source.starts_with("http://") {
        let filename = source
            .split('/')
            .next_back()
            .unwrap_or("model.gguf")
            .to_string();
        Ok(DownloadSource::DirectUrl {
            url: source.to_string(),
            filename,
        })
    } else {
        bail!("unknown source format: {source}  (use hf:owner/repo/file.gguf or https://...)")
    }
}
