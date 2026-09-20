# Pull is the only operation that reaches the network

Only Pull may access the network. When a Checkpoint is missing, judgement and serving fail with an
actionable error pointing at `s1gate pull` instead of fetching it implicitly. Chosen because the
runtime is expected to run on hosts with no egress, where an implicit fetch would turn a configuration
mistake into a hang.

Enforced structurally rather than by review: the HTTP client is a dependency of the Pull code path
alone, so no other command can reach the network even by accident. Consequence: a missing or moved
Checkpoint never self-heals; recovering is always an explicit operator action.
