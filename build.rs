// Build script: embed the userspace programs (rustos-hello and
// rustos-sh) into the kernel image.
//
// The kernel carries the programs as embedded ELFs (see
// `src/userspace.rs`): there is no filesystem or ramdisk to load
// them from at boot, so each ELF is baked into the kernel binary
// via `include_bytes!(concat!(env!("OUT_DIR"), "/hello"))` (and
// `/sh` for the shell).
//
// The orchestrator builds rustos-hello and rustos-sh before the
// kernel, so in the normal flow the ELFs are at
// `../rustos-hello/target/x86_64-unknown-none/debug/hello` (and
// the matching `../rustos-sh/.../sh`) (release as a fallback). If
// one is not there yet (e.g. `cargo test` on a fresh checkout), an
// empty placeholder is written instead: the kernel still compiles
// and boots, but reports `MissingElf` and runs without that
// userspace program (see `src/userspace.rs`).

use std::path::PathBuf;

/// Copy one program ELF into `OUT_DIR` (or write an empty
/// placeholder when it was not built yet).
fn embed(out: &PathBuf, name: &str, repo: &str, bin: &str) {
    let dest = out.join(name);
    for cand in [
        format!("../{repo}/target/x86_64-unknown-none/debug/{bin}"),
        format!("../{repo}/target/x86_64-unknown-none/release/{bin}"),
    ] {
        let p = PathBuf::from(&cand);
        if p.is_file() {
            std::fs::copy(&p, &dest).expect("copy userspace ELF into OUT_DIR");
            println!("cargo:rerun-if-changed={}", p.display());
            return;
        }
    }
    // No ELF yet: empty placeholder (see the header comment).
    std::fs::write(&dest, []).expect("write empty userspace placeholder");
    println!(
        "cargo:warning={repo} ELF not found (build it: cd ../{repo} && cargo build); kernel will boot without that userspace program"
    );
    println!(
        "cargo:rerun-if-changed=../{repo}/target/x86_64-unknown-none/debug/{bin}"
    );
}

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    embed(&out, "hello", "rustos-hello", "hello");
    embed(&out, "sh", "rustos-sh", "sh");
}
