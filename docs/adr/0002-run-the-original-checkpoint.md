# Execute the published checkpoint, never a converted one

The runtime loads the Model Source's own files unmodified: the checkpoint weights, the encoder's
configuration, the tokenizer, and the agent configuration, all in their published layout. A
third-party port publishes a converted F32 checkpoint that loads faster and maps directly onto a
tensor library's own loader, and it was rejected. A converted artifact hides which bytes actually
execute, and Provenance could no longer name a revision of the Model Source.

Consequence: the runtime accepts the published layout as-is — 16-bit weights with a single 32-bit
parameter, the encoder configuration that describes the architecture, and the tokenizer files — and
does any adaptation in memory.
