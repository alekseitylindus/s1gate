# The Metal backend is disabled through a vendored mlx-sys

Status: accepted. Deviates from ADR-0004 in one respect — MLX is still built from source and still
pinned exactly, but its Metal backend is compiled out.

Apple's Metal compiler ships with full Xcode. Only CommandLineTools is installed on the development
machine, so `mlx-sys` cannot build its Metal kernels. The feature cannot be turned off from this
crate's manifest: `mlx-sys` sets `MLX_BUILD_METAL=ON` behind `#[cfg(feature = "metal")]`, its own
default feature set includes `metal`, and `mlx-rs` declares `mlx-sys = "=0.6.0"` without
`default-features = false`, so feature unification forces the backend on regardless of what we ask for.
`mlx-sys` reads only `MLX_RS_METAL_PATH`, `HOME` and `OUT_DIR` from the environment and has no
`find_package`, `MLX_DIR` or pkg-config path, so an externally installed MLX cannot be substituted
either.

We therefore vendor `mlx-sys` 0.6.0 under `vendor/mlx-sys/` with `default = ["accelerate"]`, substitute
it through `[patch.crates-io]`, and select a narrower feature set from `mlx-rs` in our own manifest.
Inference runs on MLX's CPU backend.

**The deviation is confined to the FFI layer.** `mlx-rs` stays the upstream, unmodified binding that
ADR-0004 requires; nothing forks it, vendors it, or wraps it. `mlx-sys` keeps every other feature and
only loses its default `metal`.

Consequences, all of them live:

- Inference is CPU-only. Published latency figures from other MLX ports of this model are not
  reproducible in this milestone, and this milestone makes no performance claim.
- The vendored tree is third-party source in this repository (~1 MB, including its vendored `mlx-c`).
  Any MLX version bump means re-vendoring, not just editing a version string.
- Reverting this is a manifest change plus deleting the vendored tree, and is the intended end state
  once Xcode is available. Checking in third-party source was chosen to keep the repository buildable
  without a 10 GB developer-tools install, not because CPU execution is desirable.

Rejected alternatives:

- **Installing full Xcode** — the honest fix, but 10+ GB and a developer-directory change for a feature
  this milestone does not need.
- **Vendoring or forking `mlx-rs` instead**, patching its dependency declaration so `mlx-sys` arrives
  from crates.io untouched. It keeps our patch on the crate we actually depend on and keeps third-party
  C code out of the tree, but it forks the binding, which is required to stay upstream.
- **Overriding `MLX_BUILD_METAL` from outside** — not a supported override: `mlx-sys` only ever sets
  that define and never reads it, and the binding's Metal linkage is decided by the same Cargo feature,
  so the build would still fail.
