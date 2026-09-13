// Build script: link compatibility shims on systems whose system libraries
// lack symbols referenced by prebuilt ONNX Runtime binaries (pyke CDN,
// built with GCC 14+/glibc 2.38+):
// - __isoc23_strtol family (glibc < 2.38)
// - std::__cxx11::basic_string::_M_replace_cold (libstdc++ < 13)

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let host = std::env::var("HOST").unwrap_or_default();
    let target = std::env::var("TARGET").unwrap_or_default();
    if host != target || !target.contains("linux") {
        return;
    }

    let needs_shim = match detect_glibc_version() {
        Some((major, minor)) => (major, minor) < (2, 38),
        None => true,
    };
    if !needs_shim {
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());

    let c_shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".shim/isoc23_shim.c");
    let c_obj = out_dir.join("isoc23_shim.o");
    let status = Command::new(&cc)
        .args(["-c", "-O2", "-fPIC"])
        .arg(&c_shim)
        .arg("-o")
        .arg(&c_obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke `{cc}` for isoc23 shim: {e}"));
    if !status.success() {
        panic!("`{cc}` failed to build isoc23 shim");
    }
    println!("cargo:rustc-link-arg={}", c_obj.display());

    // _M_replace_cold needs a C++ compiler (name mangling + std::string).
    let cxx_shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".shim/replace_cold_shim.cc");
    let cxx_obj = out_dir.join("replace_cold_shim.o");
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let status = Command::new(&cxx)
        .args(["-c", "-O2", "-fPIC"])
        .arg(&cxx_shim)
        .arg("-o")
        .arg(&cxx_obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke `{cxx}` for replace_cold shim: {e}"));
    if !status.success() {
        panic!("`{cxx}` failed to build replace_cold shim");
    }
    println!("cargo:rustc-link-arg={}", cxx_obj.display());

    println!("cargo:rerun-if-changed={}", c_shim.display());
    println!("cargo:rerun-if-changed={}", cxx_shim.display());
}

fn detect_glibc_version() -> Option<(u32, u32)> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let probe = r#"
#include <stdlib.h>
#if defined(__GLIBC__) && defined(__GLIBC_MINOR__)
GLIBC_VERSION_MARK __GLIBC__ __GLIBC_MINOR__
#endif
"#;
    let mut child = Command::new(&cc)
        .args(["-E", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = std::io::Write::write_all(&mut stdin, probe.as_bytes());
    }
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("GLIBC_VERSION_MARK") {
            let mut it = rest.split_whitespace();
            let major = it.next()?.parse().ok()?;
            let minor = it.next()?.parse().ok()?;
            return Some((major, minor));
        }
    }
    None
}
