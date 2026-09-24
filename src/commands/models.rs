//! The `models` command: list the Model Identifiers available to `infer`, one per line, as the
//! Model Router answers for this process (ADR-0018).

use crate::router::ModelRouter;

pub fn run() -> crate::error::Result<()> {
    for available in ModelRouter::from_env().available()? {
        println!("{}", available.identifier);
    }
    Ok(())
}
