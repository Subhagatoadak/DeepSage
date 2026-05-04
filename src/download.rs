use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

/// A file entry in a HuggingFace model repository.
#[derive(Debug, Deserialize)]
pub struct HfSibling {
    #[serde(rename = "rfilename")]
    pub filename: String,
    pub size: Option<u64>,
}

/// An entry from the HuggingFace tree API (used for SHA256 lookup).
#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    path: String,
    #[serde(default)]
    lfs: Option<HfLfs>,
}

#[derive(Debug, Deserialize)]
struct HfLfs {
    oid: String, // "sha256:<hex>"
}

/// List all GGUF files in a HuggingFace repo.
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

/// Fetch the SHA-256 hex digest of a specific file in a HuggingFace repo
/// by reading its LFS metadata from the tree API.
pub async fn hf_file_sha256(
    repo: &str,
    filename: &str,
    token: Option<&str>,
    endpoint: &str,
) -> Result<String> {
    let url = format!("{endpoint}/api/models/{repo}/tree/main");
    let client = build_client(token)?;
    let entries: Vec<HfTreeEntry> = client
        .get(&url)
        .send()
        .await?
        .json()
        .await
        .context("failed to parse HF tree API response")?;

    for entry in &entries {
        if entry.path == filename {
            if let Some(lfs) = &entry.lfs {
                let sha = lfs.oid.strip_prefix("sha256:").unwrap_or(&lfs.oid);
                return Ok(sha.to_string());
            }
            bail!("'{filename}' found in '{repo}' but has no LFS metadata (not a binary file?)");
        }
    }
    bail!("'{filename}' not found in '{repo}' tree")
}

/// Download a file from HuggingFace and return its local path.
pub async fn hf_download<F>(
    repo: &str,
    filename: &str,
    dest_dir: &PathBuf,
    token: Option<&str>,
    endpoint: &str,
    progress_cb: F,
) -> Result<PathBuf>
where
    F: FnMut(u64, Option<u64>),
{
    let (path, _sha256) =
        hf_download_with_hash(repo, filename, dest_dir, token, endpoint, progress_cb).await?;
    Ok(path)
}

/// Download a file from HuggingFace; returns `(local_path, sha256_hex)`.
pub async fn hf_download_with_hash<F>(
    repo: &str,
    filename: &str,
    dest_dir: &PathBuf,
    token: Option<&str>,
    endpoint: &str,
    mut progress_cb: F,
) -> Result<(PathBuf, String)>
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
    let mut hasher = Sha256::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream error")?;
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        progress_cb(downloaded, total);
    }

    file.flush().await?;
    let sha256 = format!("{:x}", hasher.finalize());
    Ok((dest, sha256))
}

/// Compute the SHA-256 hex digest of a local file (blocking).
pub fn sha256_file(path: &str) -> Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).context("cannot open file for hashing")?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 65_536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
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

/// Parse a source string into a `DownloadSource`.
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
