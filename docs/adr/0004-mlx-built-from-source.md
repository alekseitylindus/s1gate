# MLX arrives through mlx-rs's from-source build

The binding is `mlx-rs`, https://github.com/oxiglade/mlx-rs, consumed unmodified and pinned exactly: no
fork, no vendored copy, no re-implementation, and no wrapper layer over it. It and its FFI layer
`mlx-sys` pin MLX and mlx-c to an exact version and build both from source with CMake during the first
`cargo build`. That is surprising — a first build requires cmake, a C++20 toolchain, bindgen's libclang,
and network access, and leaves a compiled metallib in a cache directory.

Accepted over linking a system-provided MLX because the numerical behaviour of a thin binding depends
on the MLX version underneath it: a floating system library would silently change the very numbers the
Parity comparison measures. A missing operation is called through mlx-c at the point of use rather than
worked around by forking the binding or adding a wrapper layer; at the time of writing no operation
this model needs is missing.

Our manifest selects a subset of the binding's features. That is not a modification of it — the crate
itself stays exactly what upstream published.
