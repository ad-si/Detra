//! Installing models: download a checkpoint from Hugging Face at a pinned revision
//! (or find its files in a local Hugging Face cache), convert it with the integer-only
//! converter and check that the result is the published model, bit for bit.

use detra::qwen35::{self, Checkpoint};
use detra::sha256::{hex, Sha256};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct Download {
    /// File name in the Hugging Face repository.
    pub remote: &'static str,
    /// File name in detra's download directory.
    pub local: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub struct Release {
    pub name: &'static str,
    /// Hugging Face repository.
    pub repo: &'static str,
    /// Pinned revision of the repository.
    pub revision: &'static str,
    /// Config, tokenizer and weights, in this order.
    pub files: [Download; 3],
    /// SHA-256 of the converted model file, which identifies the model.
    pub sha256: &'static str,
}

pub const DEFAULT: &str = "qwen3.5-2b";

pub const RELEASES: [Release; 2] = [
    Release {
        name: "qwen3.5-2b",
        repo: "Qwen/Qwen3.5-2B",
        revision: "15852e8c16360a2fea060d615a32b45270f8a8fc",
        files: [
            Download {
                remote: "config.json",
                local: "config.json",
                size: 2908,
                sha256: "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
            },
            Download {
                remote: "tokenizer.json",
                local: "tokenizer.json",
                size: 12_807_982,
                sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            },
            Download {
                remote: "model.safetensors-00001-of-00001.safetensors",
                local: "model.safetensors",
                size: 4_548_221_488,
                sha256: "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
            },
        ],
        sha256: "7ddcff1fc73deaf59b3242ec5b53e13b9eb0626103262650753196ed33adcf9c",
    },
    Release {
        name: "qwen3.5-0.8b",
        repo: "Qwen/Qwen3.5-0.8B",
        revision: "2fc06364715b967f1860aea9cf38778875588b17",
        files: [
            Download {
                remote: "config.json",
                local: "config.json",
                size: 2907,
                sha256: "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
            },
            Download {
                remote: "tokenizer.json",
                local: "tokenizer.json",
                size: 12_807_982,
                sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            },
            Download {
                remote: "model.safetensors-00001-of-00001.safetensors",
                local: "model.safetensors",
                size: 1_746_942_600,
                sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
            },
        ],
        sha256: "cb09481df4b531deaf63d314720396023c1fc7cd54066514993cfcbe5a60f203",
    },
];

pub fn find(name: &str) -> Option<&'static Release> {
    RELEASES.iter().find(|r| r.name == name)
}

pub fn names() -> String {
    RELEASES.iter().map(|r| r.name).collect::<Vec<_>>().join(", ")
}

fn env_dir(var: &str, fallback: &str) -> Result<PathBuf, String> {
    match (super::env_var(var), super::env_var("HOME")) {
        (Some(dir), _) => Ok(PathBuf::from(dir)),
        (None, Some(home)) => Ok(Path::new(&home).join(fallback)),
        (None, None) => Err(format!("neither {var} nor HOME is set")),
    }
}

/// Where installed models live: `$XDG_DATA_HOME/detra`, by default `~/.local/share/detra`.
pub fn data_dir() -> Result<PathBuf, String> {
    Ok(env_dir("XDG_DATA_HOME", ".local/share")?.join("detra"))
}

/// The file of an installed model.
pub fn installed(name: &str) -> Result<PathBuf, String> {
    Ok(data_dir()?.join(format!("{name}.dllm")))
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

/// Hugging Face hub caches, in the order the Hugging Face libraries choose them:
/// `$HF_HUB_CACHE`, `$HF_HOME/hub`, and `$XDG_CACHE_HOME/huggingface/hub`
/// (default `~/.cache/huggingface/hub`).
fn hf_caches() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    dirs.extend(super::env_var("HF_HUB_CACHE").map(PathBuf::from));
    dirs.extend(super::env_var("HF_HOME").map(|d| Path::new(&d).join("hub")));
    dirs.extend(env_dir("XDG_CACHE_HOME", ".cache").ok().map(|d| d.join("huggingface").join("hub")));
    dirs.dedup();
    dirs
}

/// A file in a Hugging Face cache with exactly the bytes of `file`, if there is one.
/// Weights are stored under their SHA-256 in `blobs/`, other files are found through
/// the snapshots of any revision.
fn find_cached(release: &Release, file: &Download) -> Option<PathBuf> {
    let repo_dir = format!("models--{}", release.repo.replace('/', "--"));
    for hub in hf_caches() {
        let repo = hub.join(&repo_dir);
        let mut candidates = vec![repo.join("blobs").join(file.sha256)];
        if let Ok(entries) = std::fs::read_dir(repo.join("snapshots")) {
            let mut snapshots: Vec<PathBuf> = entries.filter_map(|e| Some(e.ok()?.path().join(file.remote))).collect();
            snapshots.sort();
            candidates.extend(snapshots);
        }
        for path in candidates {
            if !std::fs::metadata(&path).is_ok_and(|m| m.is_file() && m.len() == file.size) {
                continue;
            }
            eprintln!("Checking {}", path.display());
            if hash_file(&path).is_ok_and(|h| h == file.sha256) {
                return Some(path);
            }
        }
    }
    None
}

/// SHA-256 of a file, in hex.
fn hash_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(io(path))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(io(path))?;
        if n == 0 {
            return Ok(hex(&hasher.finalize()));
        }
        hasher.update(&buf[..n]);
    }
}

/// Fetch `url` to `dest` with curl unless `dest` already holds the expected bytes.
/// An interrupted download is resumed.
fn download(url: &str, file: &Download, dest: &Path) -> Result<(), String> {
    if dest.exists() && hash_file(dest)? == file.sha256 {
        return Ok(());
    }
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let have = std::fs::metadata(&part).map_or(0, |m| m.len());
    if have > file.size {
        std::fs::remove_file(&part).map_err(io(&part))?;
    }
    if have != file.size {
        let size = match file.size {
            n if n >= 1_000_000 => format!("{} MB", n.div_ceil(1_000_000)),
            n => format!("{} kB", n.div_ceil(1_000)),
        };
        eprintln!("Downloading {} ({size})", file.remote);
        let status = Command::new("curl")
            .args(["--location", "--fail", "--retry", "5", "--continue-at", "-", "--progress-bar", "--output"])
            .arg(&part)
            .arg(url)
            .status()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => "downloading needs `curl`, which was not found".to_string(),
                _ => format!("cannot run curl: {e}"),
            })?;
        if !status.success() {
            return Err(format!("downloading {url} failed ({status}); run the command again to resume"));
        }
    }
    eprintln!("Checking {}", file.remote);
    let got = hash_file(&part)?;
    if got != file.sha256 {
        std::fs::remove_file(&part).map_err(io(&part))?;
        return Err(format!("{url} has SHA-256 {got}, expected {}; the download was deleted", file.sha256));
    }
    std::fs::rename(&part, dest).map_err(io(dest))
}

/// Passes bytes through while hashing them.
struct HashingWriter<W> {
    inner: W,
    hasher: Sha256,
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Convert `checkpoint` to `out` and return the SHA-256 of the result. The file
/// appears at `out` only once it is complete.
pub fn convert(checkpoint: &Checkpoint, out: &Path) -> Result<String, String> {
    let model = qwen35::convert(checkpoint, |step| eprint!("\rConverting: {step:<20}")).map_err(|e| {
        eprintln!();
        e.to_string()
    })?;
    eprintln!("\rConverting: done{:<20}", "");
    let mut part = out.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let file = File::create(&part).map_err(io(&part))?;
    let mut writer = HashingWriter { inner: BufWriter::with_capacity(1 << 20, file), hasher: Sha256::new() };
    let written = model.write(&mut writer).map_err(|e| e.to_string()).and_then(|()| writer.flush().map_err(io(&part)));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    std::fs::rename(&part, out).map_err(io(out))?;
    Ok(hex(&writer.hasher.finalize()))
}

/// Download, convert and install `release`.
pub fn pull(release: &Release) -> Result<(), String> {
    let dest = installed(release.name)?;
    if dest.exists() {
        eprintln!("Checking the installed {}", release.name);
        if hash_file(&dest)? == release.sha256 {
            eprintln!("{} is already installed at {}", release.name, dest.display());
            return Ok(());
        }
        eprintln!("{} does not hold {}; it will be replaced", dest.display(), release.name);
    }
    let cache = env_dir("XDG_CACHE_HOME", ".cache")?.join("detra").join(release.name);
    let mut paths = Vec::new();
    for file in &release.files {
        if let Some(path) = find_cached(release, file) {
            eprintln!("Using {}", path.display());
            paths.push(path);
            continue;
        }
        std::fs::create_dir_all(&cache).map_err(io(&cache))?;
        let url = format!("https://huggingface.co/{}/resolve/{}/{}", release.repo, release.revision, file.remote);
        let dest = cache.join(file.local);
        download(&url, file, &dest)?;
        paths.push(dest);
    }
    let [config, tokenizer, weights] = <[PathBuf; 3]>::try_from(paths).map_err(|_| "missing checkpoint file")?;
    let checkpoint = Checkpoint { config, tokenizer, weights };
    let dir = data_dir()?;
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    let mut converted = dest.as_os_str().to_owned();
    converted.push(".new");
    let converted = PathBuf::from(converted);
    let got = convert(&checkpoint, &converted)?;
    if got != release.sha256 {
        let _ = std::fs::remove_file(&converted);
        return Err(format!(
            "the converted model has SHA-256 {got}, expected {}. This is a bug in the converter; \
             please report it. The checkpoint was not deleted.",
            release.sha256
        ));
    }
    std::fs::rename(&converted, &dest).map_err(io(&dest))?;
    if cache.exists() {
        std::fs::remove_dir_all(&cache).map_err(io(&cache))?;
        if let Some(parent) = cache.parent() {
            let _ = std::fs::remove_dir(parent); // only succeeds if no other download is in it
        }
    }
    eprintln!("Installed {} (sha256 {}) at {}", release.name, &got[..16], dest.display());
    Ok(())
}
