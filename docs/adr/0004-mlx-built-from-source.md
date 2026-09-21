# MLX arrives through mlx-rs's from-source build

The binding is `mlx-rs`, https://github.com/oxiglade/mlx-rs, consumed unmodified and pinned exactly.
It is the only MLX dependency declared by s1gate and the only MLX API the runtime calls. There is no
fork, vendored copy, patch override, re-implementation, wrapper layer, or direct call below its public
Rust API.

Accepted over linking a system-provided MLX because the numerical behaviour of a thin binding depends
on the MLX version underneath it: a floating system library would silently change the very numbers the
Parity comparison measures. The project accepts the build requirements and transitive dependencies of
the pinned upstream crate instead of replacing or overriding them locally.

All required operations must come from the pinned `mlx-rs` public API. If that API cannot express a
required operation, the dependency choice must be reconsidered explicitly rather than bypassed.
