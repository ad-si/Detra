//! Installing models: download a checkpoint from Hugging Face at a pinned revision,
//! convert it with the integer-only converter and check that the result is the
//! published model, bit for bit.

use detra::qwen35;
use detra::sha256::{hex, Sha256};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct Download {
    /// File name in the Hugging Face repository.
    pub remote: &'static str,
    /// File name the converter expects.
    pub local: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub struct Release {
    pub name: &'static str,
    /// Hugging Face download URL of the repository at a pinned revision.
    pub url: &'static str,
    pub files: [Download; 3],
    /// SHA-256 of the converted model file, which identifies the model.
    pub sha256: &'static str,
}

pub const DEFAULT: &str = "qwen3.5-2b";

pub const RELEASES: [Release; 2] = [
    Release {
        name: "qwen3.5-2b",
        url: "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/15852e8c16360a2fea060d615a32b45270f8a8fc",
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
        url: "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17",
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
        eprintln!("Downloading {} ({} MB)", file.remote, file.size.div_ceil(1_000_000));
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

/// Convert the checkpoint in `dir` to `out` and return the SHA-256 of the result.
/// The file appears at `out` only once it is complete.
pub fn convert(dir: &Path, out: &Path) -> Result<String, String> {
    let model = qwen35::convert(dir, |step| eprint!("\rConverting: {step:<20}")).map_err(|e| {
        eprintln!();
        format!("{}: {e}", dir.display())
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
    std::fs::create_dir_all(&cache).map_err(io(&cache))?;
    for file in &release.files {
        download(&format!("{}/{}", release.url, file.remote), file, &cache.join(file.local))?;
    }
    let dir = data_dir()?;
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    let mut converted = dest.as_os_str().to_owned();
    converted.push(".new");
    let converted = PathBuf::from(converted);
    let got = convert(&cache, &converted)?;
    if got != release.sha256 {
        let _ = std::fs::remove_file(&converted);
        return Err(format!(
            "the converted model has SHA-256 {got}, expected {}. This is a bug in the converter; \
             please report it. The download is kept in {}.",
            release.sha256,
            cache.display()
        ));
    }
    std::fs::rename(&converted, &dest).map_err(io(&dest))?;
    std::fs::remove_dir_all(&cache).map_err(io(&cache))?;
    if let Some(parent) = cache.parent() {
        let _ = std::fs::remove_dir(parent); // only succeeds if no other download is in it
    }
    eprintln!("Installed {} (sha256 {}) at {}", release.name, &got[..16], dest.display());
    Ok(())
}
