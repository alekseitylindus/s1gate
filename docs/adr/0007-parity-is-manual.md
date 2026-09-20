# Parity with the original implementation is checked by hand, not by the test suite

The whole project exists to reproduce another implementation's answers, so the obvious move is a
checked-in parity test. We deliberately do not have one: no fixtures, no expected values, and no
oracle harness in the tree. The comparison is a manual procedure run locally, and its evidence is a
reported measurement — token ids, marker positions, parameter shapes, logits, calibrated
probabilities, selected answers, and the public output, compared stage by stage.

The reason is that the repo stays a pure Rust artifact. A parity test can only exist if something
generated from outside Rust is committed alongside it, and that artifact would then have to be trusted
without being reproducible from the repo itself.

Consequence, and the thing to know before "fixing" this: `cargo test` asserts s1gate's internal
consistency only. It can pass while Parity is broken. Re-establishing Parity is a deliberate,
repeatable-by-hand procedure, and the measured tolerances are the record of the last time it ran.
