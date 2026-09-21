//! The validated public input to `infer`.

use std::collections::BTreeSet;
use std::io::Read;

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

use crate::error::{Error, Result};

/// One System One Call: evidence plus the Questions to judge against it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub state: Value,
    pub questions: Questions,
}

impl Call {
    /// Read and validate exactly one JSON value from `reader`.
    pub fn read(reader: &mut impl Read) -> Result<Self> {
        let mut input = String::new();
        reader
            .read_to_string(&mut input)
            .map_err(|error| Error::io("read", "<stdin>", error))?;

        let mut json = serde_json::Deserializer::from_str(&input);
        let call = Self::deserialize(&mut json).map_err(|error| {
            Error::invalid_call(format!("cannot read System One Call: {error}"))
        })?;
        json.end().map_err(|error| {
            Error::invalid_call(format!(
                "cannot read System One Call: trailing input ({error})"
            ))
        })?;
        call.validate()?;
        Ok(call)
    }

    fn validate(&self) -> Result<()> {
        if self.questions.is_empty() {
            return Err(Error::invalid_call(
                "a System One Call must contain at least one Question",
            ));
        }
        for (id, question) in self.questions.iter() {
            validate_id("Question id", id)?;
            question.validate(id)?;
        }
        Ok(())
    }
}

/// Caller-chosen Question ids in their input order.
#[derive(Debug)]
pub struct Questions(Vec<(String, Question)>);

impl Questions {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

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
                let mut ids = BTreeSet::new();
                let mut questions = Vec::new();
                while let Some(id) = map.next_key::<String>()? {
                    if !ids.insert(id.clone()) {
                        return Err(de::Error::custom(format!("duplicate Question id `{id}`")));
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
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    #[serde(rename = "type")]
    pub kind: QuestionType,
    pub instructions: String,
    pub criteria: Option<Criteria>,
}

impl Question {
    fn validate(&self, id: &str) -> Result<()> {
        match self.kind {
            QuestionType::Choice => match self.criteria.as_ref() {
                Some(Criteria::Object(options)) if options.len() >= 2 => {
                    for (option, _) in options {
                        validate_id("Option name", option)?;
                    }
                    Ok(())
                }
                Some(Criteria::List(options)) if options.len() >= 2 => {
                    let mut names = BTreeSet::new();
                    for option in options {
                        let Some(option) = option.as_str() else {
                            return Err(Error::invalid_call(format!(
                                "Question `{id}` choice Criteria array must contain string Options"
                            )));
                        };
                        validate_id("Option name", option)
                            .map_err(|error| Error::invalid_call(error.to_string()))?;
                        if !names.insert(option) {
                            return Err(Error::invalid_call(format!(
                                "Question `{id}` choice Criteria must contain distinct Options"
                            )));
                        }
                    }
                    Ok(())
                }
                Some(Criteria::Object(_)) | Some(Criteria::List(_)) => Err(Error::invalid_call(
                    format!("Question `{id}` choice Criteria must contain at least two Options"),
                )),
                None => Err(Error::invalid_call(format!(
                    "Question `{id}` choice requires Criteria with at least two Options"
                ))),
            },
            QuestionType::Score => match self.criteria.as_ref() {
                Some(Criteria::List(levels)) if levels.len() >= 2 => Ok(()),
                Some(Criteria::List(_)) => Err(Error::invalid_call(format!(
                    "Question `{id}` score Criteria must contain at least two Levels"
                ))),
                Some(Criteria::Object(_)) => Err(Error::invalid_call(format!(
                    "Question `{id}` score Criteria must be an array of Levels"
                ))),
                None => Err(Error::invalid_call(format!(
                    "Question `{id}` score requires Criteria with at least two Levels"
                ))),
            },
            QuestionType::Noul => match self.criteria.as_ref() {
                None => Ok(()),
                Some(Criteria::Object(options)) if options.len() == 2 => {
                    let names: BTreeSet<&str> =
                        options.iter().map(|(name, _)| name.as_str()).collect();
                    if names == BTreeSet::from(["false", "true"]) {
                        Ok(())
                    } else {
                        Err(Error::invalid_call(format!(
                            "Question `{id}` noul Criteria must contain the false and true Options"
                        )))
                    }
                }
                Some(Criteria::Object(_)) => Err(Error::invalid_call(format!(
                    "Question `{id}` noul Criteria must contain the false and true Options"
                ))),
                Some(Criteria::List(_)) => Err(Error::invalid_call(format!(
                    "Question `{id}` noul Criteria must be an object with false and true Options"
                ))),
            },
        }
    }
}

/// The only Question Types supported by the first Backend.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

/// Criteria are either named Options or ordered Levels.
#[derive(Debug)]
pub enum Criteria {
    Object(Vec<(String, Value)>),
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
                let mut names = BTreeSet::new();
                let mut options = Vec::new();
                while let Some(name) = map.next_key::<String>()? {
                    if !names.insert(name.clone()) {
                        return Err(de::Error::custom(format!("duplicate Option `{name}`")));
                    }
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

fn validate_id(kind: &str, id: &str) -> Result<()> {
    if id.trim().is_empty() || id.chars().any(char::is_control) {
        return Err(Error::invalid_call(format!(
            "{kind} `{id}` must be non-empty and contain no control characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn caller_question_ids_and_option_order_are_preserved() {
        let mut input = Cursor::new(
            r#"{
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
            }"#,
        );

        let call = Call::read(&mut input).expect("valid call");
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
