//! The validated public input to `infer`.

use std::collections::BTreeSet;
use std::fmt;

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

use crate::error::{Error, Result};

/// One System One Call: evidence plus the Questions to judge against it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    /// The Model Identifier for the Backend that judges the Questions in this call.
    pub model: String,
    /// The evidence every Question of the call is judged against.
    pub state: Value,
    /// The Questions to judge, each under its caller-chosen id.
    pub questions: Questions,
}

impl Call {
    /// Parse and validate exactly one System One Call from `bytes`.
    ///
    /// # Errors
    ///
    /// The input is not UTF-8, is not exactly one JSON System One Call, carries trailing input,
    /// holds no Questions, names a Question with an unusable or repeated id, or defines Criteria of
    /// the wrong shape, with fewer than two names, or with repeated names.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let json = std::str::from_utf8(bytes)
            .map_err(|error| Error::invalid_call(format!("the input is not UTF-8: {error}")))?;

        let mut deserializer = serde_json::Deserializer::from_str(json);
        let call = Self::deserialize(&mut deserializer)
            .map_err(|error| Error::invalid_call(error.to_string()))?;
        deserializer
            .end()
            .map_err(|error| Error::invalid_call(format!("trailing input ({error})")))?;
        call.validate()?;
        Ok(call)
    }

    fn validate(&self) -> Result<()> {
        if self.questions.is_empty() {
            return Err(Error::invalid_call("at least one Question is required"));
        }
        for (id, question) in self.questions.iter() {
            if !name_is_usable(id) {
                return Err(Error::invalid_call(format!(
                    "Question id {} must be non-empty and contain no control characters",
                    Name(id)
                )));
            }
            question.validate(id)?;
        }
        Ok(())
    }
}

/// Caller-chosen Question ids in their input order.
#[derive(Debug, Clone, PartialEq)]
pub struct Questions(Vec<(String, Question)>);

impl Questions {
    /// Whether the call holds no Question.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The Questions in input order, each with its caller-chosen id.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Question)> {
        self.0.iter().map(|(id, question)| (id.as_str(), question))
    }
}

impl<'de> Deserialize<'de> for Questions {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct QuestionsVisitor;

        impl<'de> Visitor<'de> for QuestionsVisitor {
            type Value = Questions;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object of Question ids and Questions")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut questions = Vec::new();
                while let Some(id) = map.next_key::<String>()? {
                    if questions.iter().any(|(existing, _)| existing == &id) {
                        return Err(de::Error::custom(format!(
                            "duplicate Question id {}",
                            Name(&id)
                        )));
                    }
                    questions.push((id, map.next_value()?));
                }
                Ok(Questions(questions))
            }
        }

        deserializer.deserialize_map(QuestionsVisitor)
    }
}

/// One Question and its answer-space definition.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    /// The kind of Answer this Question expects.
    #[serde(rename = "type")]
    pub kind: QuestionType,
    /// What the Question asks, rendered into its prompt.
    pub instructions: String,
    /// The answer space the Question defines; a `noul` Question may leave it out.
    pub criteria: Option<Criteria>,
}

impl Question {
    fn validate(&self, id: &str) -> Result<()> {
        match self.kind {
            QuestionType::Choice => match self.criteria.as_ref() {
                Some(Criteria::Object(options)) => {
                    let names: Vec<&str> = options.iter().map(|(name, _)| name.as_str()).collect();
                    validate_names(id, self.kind, "Option", &names)
                }
                Some(Criteria::List(values)) => {
                    let names = strings(id, self.kind, "Option", values)?;
                    validate_names(id, self.kind, "Option", &names)
                }
                None => Err(Error::invalid_call(format!(
                    "Question `{id}` choice requires Criteria with at least two Options"
                ))),
            },
            QuestionType::Score => match self.criteria.as_ref() {
                Some(Criteria::List(values)) => {
                    let names = strings(id, self.kind, "Level", values)?;
                    validate_names(id, self.kind, "Level", &names)
                }
                Some(Criteria::Object(_)) => Err(Error::invalid_call(format!(
                    "Question `{id}` score Criteria must be an array of Levels"
                ))),
                None => Err(Error::invalid_call(format!(
                    "Question `{id}` score requires Criteria with at least two Levels"
                ))),
            },
            QuestionType::Noul => match self.criteria.as_ref() {
                None => Ok(()),
                Some(Criteria::Object(options)) => {
                    let names: BTreeSet<&str> =
                        options.iter().map(|(name, _)| name.as_str()).collect();
                    if options.len() == 2 && names == BTreeSet::from(["false", "true"]) {
                        Ok(())
                    } else {
                        Err(Error::invalid_call(format!(
                            "Question `{id}` noul Criteria must contain the false and true Options"
                        )))
                    }
                }
                Some(Criteria::List(_)) => Err(Error::invalid_call(format!(
                    "Question `{id}` noul Criteria must be an object with false and true Options"
                ))),
            },
        }
    }
}

/// The only Question Types supported by the first Backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    /// One of the Question's labelled Options is chosen.
    Choice,
    /// One of the Question's ordered Levels is reported.
    Score,
    /// The probability of the true side is reported.
    Noul,
}

impl fmt::Display for QuestionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        })
    }
}

impl QuestionType {
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }
}

/// Criteria are either named Options or ordered Levels.
#[derive(Debug, Clone, PartialEq)]
pub enum Criteria {
    /// Named Options, in the order the caller wrote them.
    Object(Vec<(String, Value)>),
    /// Ordered Levels, in the order the caller wrote them.
    List(Vec<Value>),
}

impl<'de> Deserialize<'de> for Criteria {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CriteriaVisitor;

        impl<'de> Visitor<'de> for CriteriaVisitor {
            type Value = Criteria;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object of Options or an array of Levels")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut options = Vec::new();
                while let Some(name) = map.next_key::<String>()? {
                    options.push((name, map.next_value()?));
                }
                Ok(Criteria::Object(options))
            }

            fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut levels = Vec::new();
                while let Some(level) = sequence.next_element()? {
                    levels.push(level);
                }
                Ok(Criteria::List(levels))
            }
        }

        deserializer.deserialize_any(CriteriaVisitor)
    }
}

/// Whether a caller-chosen name — a Question id, an Option name, a Level — can be used: non-empty
/// and free of control characters.
fn name_is_usable(name: &str) -> bool {
    !name.trim().is_empty() && !name.chars().any(char::is_control)
}

/// The Options or Levels a Question's Criteria defines, named `noun` in diagnostics: at least two,
/// each a usable name, and none repeated.
fn validate_names(id: &str, kind: QuestionType, noun: &str, names: &[&str]) -> Result<()> {
    if names.len() < 2 {
        return Err(Error::invalid_call(format!(
            "Question `{id}` {kind} Criteria must contain at least two {noun}s"
        )));
    }
    let mut seen = BTreeSet::new();
    for name in names {
        if !name_is_usable(name) {
            return Err(Error::invalid_call(format!(
                "Question `{id}` {noun} name {} must be non-empty and contain no control characters",
                Name(name)
            )));
        }
        if !seen.insert(*name) {
            return Err(Error::invalid_call(format!(
                "Question `{id}` {kind} Criteria must contain distinct {noun}s"
            )));
        }
    }
    Ok(())
}

/// The strings a Criteria array holds, or the error naming the Question whose array it is not.
fn strings<'a>(
    id: &str,
    kind: QuestionType,
    noun: &str,
    values: &'a [Value],
) -> Result<Vec<&'a str>> {
    values
        .iter()
        .map(|value| {
            value.as_str().ok_or_else(|| {
                Error::invalid_call(format!(
                    "Question `{id}` {kind} Criteria array must contain string {noun}s"
                ))
            })
        })
        .collect()
}

/// A caller-chosen name as a diagnostic shows it: quoted, with non-printable characters escaped, so
/// a rejected name is visible and cannot rewrite the terminal it is reported to.
struct Name<'a>(&'a str);

impl fmt::Display for Name<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`", self.0.escape_debug())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_question_ids_and_option_order_are_preserved() {
        let input = r#"{
                "model": "convaiinnovations/laya",
                "state": "evidence",
                "questions": {
                    "first question": {
                        "type": "choice",
                        "instructions": "Which?",
                        "criteria": {"z": "last", "a": "first"}
                    },
                    "second/question": {
                        "type": "score",
                        "instructions": "How much?",
                        "criteria": ["low", "high"]
                    }
                }
            }"#;

        let call = Call::from_bytes(input.as_bytes()).expect("valid call");
        let ids: Vec<_> = call.questions.iter().map(|(id, _)| id).collect();
        assert_eq!(ids, ["first question", "second/question"]);
        let (_, question) = call.questions.iter().next().expect("a question");
        let Some(Criteria::Object(options)) = question.criteria.as_ref() else {
            panic!("choice criteria is an option object");
        };
        assert_eq!(
            options
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            ["z", "a"]
        );
    }
}
