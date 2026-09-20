# Published checksums are sha256 or git blob object ids

The Model Source publishes one checksum per file in `x-linked-etag`, and it is not always the same
kind of hash: files it holds in LFS are published as a sha256 of the bytes, while files it tracks in
git are published as a git blob object id — sha1 over `blob <size>\0` and the bytes.

**Decided: record the algorithm beside the checksum, and verify against the matching local digest.**
A 64-hex value is a sha256, a 40-hex value is a git blob object id; the git blob digest is computable
while streaming only because the size arrives with the response headers, which is why the record
carries the size it was hashed against. Any other form is not a content checksum, so Pull records none
and verifies the size alone.

Recording the raw value without its algorithm would have left four of the five allowlisted files
unverified at Pull time: only `model.safetensors` is in LFS. Verification of the others would then
wait for `verify`, which re-hashes against the Provenance record rather than against the Model Source.
Consequence: a truncated or swapped configuration file fails the Pull that streamed it, with the file
name and both hashes in the message, instead of producing a Checkpoint that loads and behaves subtly
differently.

A git blob object id can be recomputed only when the response announces the size, which is where its
prefix comes from. A Model Source that announces no size leaves the published value recorded but
unchecked, and the file's local sha256 is still recorded; the size check and `verify` are what close
that case.
