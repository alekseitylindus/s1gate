# The Model Store holds one Checkpoint per Model Source, under the Model Source's name

A Checkpoint is stored under its Model Source's full name, `<owner>/<name>`, at
`<store root>/<owner>/<name>`. The operator does not name a Checkpoint: `pull` takes a Model Source
and nothing else, so the name the Model Store, the runtime and `verify` address is the name the Model
Source publishes.

The alternative, a caller-chosen name, was the earlier decision (ADR-0006, removed). It was rejected
because it made every operator learn two strings to pull one model — a repository to fetch and an
unrelated name to invent — and because the set of Model Sources is curated, so s1gate already knows
which names exist. The cost of deriving the name is that it tracks the Model Source: a Model Source
renamed or republished under another owner changes the key, and its Checkpoint must be pulled again.
That is accepted, because the Model Source is the identity of what is stored.

Consequence: one Model Source holds one Checkpoint. Pulling another revision over it is refused unless
`--force` replaces it, so two revisions of one Model Source cannot sit side by side in the store.

The store lives at `$XDG_DATA_HOME/s1gate/models` (default `~/.local/share/s1gate/models`) and
configuration will live at `$XDG_CONFIG_HOME/s1gate` (default `~/.config/s1gate`). The platform-native
macOS location would be `~/Library/Application Support`; XDG was chosen deliberately so one layout and
one environment variable address the store on every platform. Moving the store later would move users'
pulled Checkpoints, so the location is recorded now.

The first milestone reads no configuration file; the location is fixed here so the second milestone
does not have to invent one.
