# Native Candle CPU inference with Accelerate

Judgement runs natively in Rust through Candle 0.11.0 on the CPU, with Apple's Accelerate framework
underneath its matrix multiplications. Runtime inference requires no Python, PyTorch, Transformers,
subprocess, GPU, or GPU build toolchain: the build and the run need Rust, the Xcode Command Line
Tools, the macOS SDK, and the Accelerate framework that ships with macOS.

The original implementation is Python, so the cheapest path was to shell out to it or embed an
interpreter; that was rejected because the runtime is meant to ship as a single native binary with
no interpreter or package environment to provision on the target host. A GPU path was rejected too:
this milestone is macOS CPU only, and Candle's Metal backend would take on a device selection, a
second numerical path to hold in Parity, and the Metal toolchain this decision removes. The runtime
before this one linked `mlx-rs`, whose build needs the Metal compiler from a full Xcode
installation; Laya inference never used the GPU, so that requirement bought nothing.

`candle-core` and `candle-nn` are pinned exactly at `0.11.0` with the `accelerate` feature enabled.
The only dependency that feature adds is `accelerate-src`, whose build script emits the link to the
system framework: no vendor library is compiled from source. Every parameter of the published
Checkpoint is converted to F32 before computation, which the Accelerate path requires — its matrix
multiplication is F32 and F64 only — and which matches the F32 computation the original
implementation performs on its own Apple path.

Candle's fused `sdpa` has no CPU implementation, so attention is written out as scaled `QK^T`, the
Checkpoint's additive `-1e4` mask, softmax, and `V`. That also keeps the mask semantics of the
original implementation rather than a fused operation's own.

Consequence: every numerical detail of the original — serialization, tokenization, budgets, marker
positions, attention, calibration, confidence, rounding — still has to be reproduced by hand and
defended by Parity tests rather than inherited by construction.
