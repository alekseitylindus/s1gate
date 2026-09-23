//! The validated public input to `infer`.

use std::collections::BTreeSet;
use std::fmt;

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

use crate::error::{Error, Result};

/// One System One Call: evidence plus the Questions to judge against it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
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
    /// holds no Questions, or defines Criteria of the wrong shape. The local Backend also requires
    /// answer spaces it can judge and names it can use in a prompt.
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
        if !is_typesafe_value(&self.state) {
            return Err(Error::invalid_call(
                "state must be a string, object, or array",
            ));
        }
        if self.questions.is_empty() {
            return Err(Error::invalid_call("at least one Question is required"));
        }
        let local = !self.model.starts_with("jev-");
        for (id, question) in self.questions.iter() {
            if local && !name_is_usable(id) {
                return Err(Error::invalid_call(format!(
                    "Question id {} must be non-empty and contain no control characters",
                    Name(id)
                )));
            }
            question.validate(id, local)?;
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
pub struct Question {
    /// The kind of Answer this Question expects.
    #[serde(rename = "type")]
    pub kind: QuestionType,
    /// What the Question asks, rendered into its prompt; TypeSafe permits omitting it.
    #[serde(default)]
    pub instructions: Value,
    /// The answer space the Question defines; a `noul` Question may leave it out.
    pub criteria: Option<Criteria>,
}

impl Question {
    fn validate(&self, id: &str, local: bool) -> Result<()> {
        if !self.instructions.is_null() && !is_typesafe_value(&self.instructions) {
            return Err(Error::invalid_call(format!(
                "Question `{id}` instructions must be a string, object, or array"
            )));
        }
        match self.kind {
            QuestionType::Choice => match self.criteria.as_ref() {
                Some(Criteria::Object(options)) => {
                    if local && options.len() > 255 {
                        return Err(Error::invalid_call(format!(
                            "Question `{id}` choice Criteria may contain at most 255 Options"
                        )));
                    }
                    if options
                        .iter()
                        .any(|(_, value)| !value.is_null() && !is_typesafe_value(value))
                    {
                        return Err(Error::invalid_call(format!(
                            "Question `{id}` choice descriptions must be strings, objects, arrays, or null"
                        )));
                    }
                    if local {
                        let names: Vec<&str> =
                            options.iter().map(|(name, _)| name.as_str()).collect();
                        validate_names(id, self.kind, "Option", &names)
                    } else {
                        Ok(())
                    }
                }
                Some(Criteria::List(_)) => Err(Error::invalid_call(format!(
                    "Question `{id}` choice Criteria must be an object of Options"
                ))),
                None => Err(Error::invalid_call(format!(
                    "Question `{id}` choice requires Criteria with at least one Option"
                ))),
            },
            QuestionType::Score => match self.criteria.as_ref() {
                Some(Criteria::List(values)) => {
                    if values.is_empty() {
                        return Err(Error::invalid_call(format!(
                            "Question `{id}` score Criteria must contain at least one Level"
                        )));
                    }
                    if local && values.len() > 10 {
                        return Err(Error::invalid_call(format!(
                            "Question `{id}` score Criteria may contain at most 10 Levels"
                        )));
                    }
                    if values.iter().any(|value| !is_typesafe_value(value)) {
                        return Err(Error::invalid_call(format!(
                            "Question `{id}` score descriptions must be strings, objects, or arrays"
                        )));
                    }
                    Ok(())
                }
                Some(Criteria::Object(_)) => Err(Error::invalid_call(format!(
                    "Question `{id}` score Criteria must be an array of Levels"
                ))),
                None => Err(Error::invalid_call(format!(
                    "Question `{id}` score requires Criteria with at least one Level"
                ))),
            },
            QuestionType::Noul => match self.criteria.as_ref() {
                None => Ok(()),
                Some(Criteria::Object(options)) => {
                    let mut names = BTreeSet::new();
                    for (name, value) in options {
                        if local && !matches!(name.as_str(), "false" | "true") {
                            return Err(Error::invalid_call(format!(
                                "Question `{id}` noul Criteria may contain only `false` and `true`"
                            )));
                        }
                        if matches!(name.as_str(), "false" | "true")
                            && !value.is_null()
                            && !is_typesafe_value(value)
                        {
                            return Err(Error::invalid_call(format!(
                                "Question `{id}` noul descriptions must be strings, objects, or arrays"
                            )));
                        }
                        if !names.insert(name.as_str()) {
                            return Err(Error::invalid_call(format!(
                                "Question `{id}` noul Criteria must not repeat `{name}`"
                            )));
                        }
                    }
                    Ok(())
                }
                Some(Criteria::List(_)) => Err(Error::invalid_call(format!(
                    "Question `{id}` noul Criteria must be an object with false and true Options"
                ))),
            },
        }
    }
}

/// `TypeSafe`'s structured input values are strings, objects, and arrays.
fn is_typesafe_value(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
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

/// The Options or Levels a Question's Criteria defines, named `noun`: at least one,
/// each a usable name, and none repeated.
fn validate_names(id: &str, kind: QuestionType, noun: &str, names: &[&str]) -> Result<()> {
    if names.is_empty() {
        return Err(Error::invalid_call(format!(
            "Question `{id}` {kind} Criteria must contain at least one {noun}"
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
