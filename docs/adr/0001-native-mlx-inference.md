# Native MLX inference on Apple Silicon

Judgement runs natively in Rust on Apple Silicon through MLX. Runtime inference requires no Python,
PyTorch, Transformers, or subprocess. The original implementation is Python, so the cheapest path was
to shell out to it or embed an interpreter; that was rejected because the runtime is meant to ship as
a single native binary with no interpreter or package environment to provision on the target host.

Consequence: every numerical detail of the original — serialization, tokenization, budgets, marker
positions, attention, calibration, confidence, rounding — has to be reproduced by hand and defended
by Parity tests rather than inherited by construction.
