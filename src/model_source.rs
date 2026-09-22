//! The curated Model Sources s1gate can execute, and the Checkpoint allowlist each one publishes.

use crate::error::{Error, Result};

/// An upstream repository that publishes Checkpoints s1gate can execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSource {
    /// `<owner>/<name>` on the Model Source's host: the name the operator pulls by, and the name
    /// the Checkpoint is stored under (ADR-0011).
    pub repo: &'static str,
    /// The Checkpoint allowlist in Pull order: paths relative to the repository root. Pull fails if
    /// the Model Source does not publish one of them.
    pub files: &'static [&'static str],
}

/// `convaiinnovations/laya` — Laya, the only Backend of the first milestone.
pub const LAYA: ModelSource = ModelSource {
    repo: "convaiinnovations/laya",
    files: &[
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ],
};

/// Every Model Source s1gate supports.
pub const SOURCES: &[ModelSource] = &[LAYA];

/// The supported Model Source repositories, in curation order.
pub(crate) fn supported() -> impl Iterator<Item = &'static str> {
    SOURCES.iter().map(|source| source.repo)
}

/// The curated Model Source for `repo`, or the error naming the supported ones.
///
/// # Errors
///
/// [`Error::UnsupportedSource`], exit code 2, when `repo` is not one of the curated repositories.
pub fn lookup(repo: &str) -> Result<&'static ModelSource> {
    SOURCES
        .iter()
        .find(|source| source.repo == repo)
        .ok_or_else(|| Error::UnsupportedSource {
            requested: repo.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laya_is_curated_with_its_five_file_allowlist() {
        let source = lookup("convaiinnovations/laya").expect("laya is curated");
        assert_eq!(
            source.files,
            [
                "model.safetensors",
                "rl_agent_config.json",
                "encoder/config.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
            ]
        );
    }

    #[test]
    fn every_curated_model_source_names_one_checkpoint_directory() {
        for source in SOURCES {
            assert!(
                crate::store::Store::at("")
                    .checkpoint_dir(source.repo)
                    .is_ok(),
                "`{}` is not a name the Model Store can hold a Checkpoint under",
                source.repo
            );
        }
    }

    #[test]
    fn unsupported_source_lists_the_supported_one() {
        let error = lookup("some/other-model").expect_err("a foreign repository is not curated");
        assert!(matches!(error, Error::UnsupportedSource { .. }));
        assert_eq!(
            error.to_string(),
            "unsupported Model Source `some/other-model` (supported: convaiinnovations/laya)"
        );
        assert_eq!(error.exit_code(), 2);
    }

    #[test]
    fn repository_names_are_matched_exactly() {
        assert!(lookup("ConvaiInnovations/Laya").is_err());
        assert!(lookup("convaiinnovations/laya/").is_err());
    }
}
