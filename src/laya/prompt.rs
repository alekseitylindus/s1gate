//! Python-runtime-compatible prompt construction and token budgeting.

use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write as _};
use std::path::Path;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::call::{Call, Criteria, Question, QuestionType};
use crate::error::{Error, Result};

use super::SpecialTokens;
use super::config::AgentConfig;

const MAX_OPTION_TOKENS: usize = 48;

#[derive(Debug)]
pub(super) struct Sequence {
    pub(super) ids: Vec<i32>,
    pub(super) markers: Vec<usize>,
}

#[derive(Debug)]
pub(super) struct Prepared {
    pub(super) sequence: Sequence,
    pub(super) kind: QuestionType,
    pub(super) options: Vec<String>,
    pub(super) labels: Vec<String>,
    pub(super) levels: Vec<String>,
}

pub(super) fn special_tokens(directory: &Path, tokenizer: &Tokenizer) -> Result<SpecialTokens> {
    let config: Value = read_json(&directory.join("tokenizer/tokenizer_config.json"))?;
    let token = |name: &str| -> Result<(String, i32)> {
        let value = config.get(name).ok_or_else(|| Error::Inference {
            message: format!("tokenizer config is missing {name}"),
        })?;
        let text = value
            .as_str()
            .map(str::to_string)
            .or_else(|| {
                value
                    .get("content")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .ok_or_else(|| Error::Inference {
                message: format!("tokenizer config has no usable {name}"),
            })?;
        let id = tokenizer
            .token_to_id(&text)
            .ok_or_else(|| Error::Inference {
                message: format!("tokenizer has no {name} token `{text}`"),
            })?;
        let id = i32::try_from(id).map_err(|_| Error::Inference {
            message: format!("tokenizer id for {name} is too large"),
        })?;
        Ok((text, id))
    };
    let (mask_text, mask) = token("mask_token")?;
    Ok(SpecialTokens {
        cls: token("cls_token")?.1,
        sep: token("sep_token")?.1,
        mask,
        pad: token("pad_token")?.1,
        mask_text,
    })
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|error| Error::io("read", path, error))?;
    serde_json::from_slice(&bytes).map_err(|error| Error::json(path.display().to_string(), error))
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|error| Error::Inference {
            message: format!("tokenizing prompt: {error}"),
        })?;
    encoding
        .get_ids()
        .iter()
        .map(|id| {
            i32::try_from(*id).map_err(|_| Error::Inference {
                message: "tokenizer id is too large".to_string(),
            })
        })
        .collect()
}

fn render_value(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| python_repr(value))
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => python_quote(text),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{}: {}", python_quote(key), python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_quote(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push(quote);
    for character in text.chars() {
        match character {
            '\\' => quoted.push_str("\\\\"),
            character if character == quote => {
                quoted.push('\\');
                quoted.push(character);
            }
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            character if character.is_control() => {
                let _ = write!(quoted, "\\x{:02x}", u32::from(character));
            }
            character => quoted.push(character),
        }
    }
    quoted.push(quote);
    quoted
}

fn is_python_falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(value) => !value,
        Value::Number(value) => value.as_f64() == Some(0.0),
        Value::String(value) => value.is_empty(),
        Value::Array(value) => value.is_empty(),
        Value::Object(value) => value.is_empty(),
    }
}

fn render_options(question: &Question) -> (Vec<String>, Vec<String>, Vec<String>) {
    match question.kind {
        QuestionType::Choice => match question.criteria.as_ref() {
            Some(Criteria::Object(options)) => {
                let rendered = options
                    .iter()
                    .map(|(name, value)| {
                        if is_python_falsy(value) {
                            name.clone()
                        } else {
                            format!("{name}: {}", render_value(value))
                        }
                    })
                    .collect();
                (
                    rendered,
                    options.iter().map(|(name, _)| name.clone()).collect(),
                    Vec::new(),
                )
            }
            Some(Criteria::List(options)) => {
                let rendered = options.iter().map(render_value).collect::<Vec<_>>();
                (rendered.clone(), rendered, Vec::new())
            }
            None => (Vec::new(), Vec::new(), Vec::new()),
        },
        QuestionType::Score => match question.criteria.as_ref() {
            Some(Criteria::List(levels)) => {
                let levels = levels.iter().map(render_value).collect::<Vec<_>>();
                let rendered = levels
                    .iter()
                    .enumerate()
                    .map(|(index, level)| format!("level {index}: {level}"))
                    .collect();
                (rendered, Vec::new(), levels)
            }
            _ => (Vec::new(), Vec::new(), Vec::new()),
        },
        QuestionType::Noul => {
            let descriptions = question
                .criteria
                .as_ref()
                .and_then(|criteria| match criteria {
                    Criteria::Object(options) => Some(options),
                    Criteria::List(_) => None,
                });
            let render = |name: &str, fallback: &str| {
                descriptions
                    .and_then(|options| options.iter().find(|(key, _)| key == name))
                    .map(|(_, value)| {
                        if is_python_falsy(value) {
                            fallback.to_string()
                        } else {
                            render_value(value)
                        }
                    })
                    .unwrap_or_else(|| fallback.to_string())
            };
            (
                vec![
                    format!(
                        "false: {}",
                        render("false", "no, the statement does not hold")
                    ),
                    format!("true: {}", render("true", "yes, the statement holds")),
                ],
                Vec::new(),
                Vec::new(),
            )
        }
    }
}

pub(super) fn prepare(
    call: &Call,
    tokenizer: &Tokenizer,
    special: &SpecialTokens,
    config: &AgentConfig,
) -> Result<Vec<Prepared>> {
    call.questions
        .iter()
        .map(|(_, question)| {
            let (options, labels, levels) = render_options(question);
            let mut head = encode(
                tokenizer,
                &format!(
                    "{} question: {}",
                    question.kind,
                    question.instructions.replace(&special.mask_text, " ")
                ),
            )?;
            let mut option_ids = Vec::with_capacity(options.len());
            for option in &options {
                let mut ids = vec![special.mask];
                let mut body = encode(
                    tokenizer,
                    &format!(" {}", option.replace(&special.mask_text, " ")),
                )?;
                body.truncate(MAX_OPTION_TOKENS);
                ids.extend(body);
                option_ids.push(ids);
            }
            let used: usize = option_ids.iter().map(Vec::len).sum();
            let mut budget = config.head_max_len.saturating_sub(used);
            if budget < 16 {
                let per = config
                    .head_max_len
                    .saturating_sub(16)
                    .checked_div(option_ids.len().max(1))
                    .unwrap_or(0)
                    .max(4);
                for ids in &mut option_ids {
                    ids.truncate(per);
                }
                budget = config
                    .head_max_len
                    .saturating_sub(option_ids.iter().map(Vec::len).sum());
            }
            head.truncate(budget.max(8));
            let mut ids = vec![special.cls];
            ids.extend(head);
            ids.push(special.sep);
            let mut markers = Vec::with_capacity(option_ids.len());
            for option in option_ids {
                markers.push(ids.len());
                ids.extend(option);
            }
            ids.push(special.sep);
            let state = serialize_state(&call.state)?.replace(&special.mask_text, " ");
            let room = config.max_len.saturating_sub(ids.len().saturating_add(1));
            let mut state_ids = encode(tokenizer, &state)?;
            state_ids.truncate(room);
            ids.extend(state_ids);
            ids.push(special.sep);
            ids.truncate(config.max_len);
            if markers.iter().any(|marker| *marker >= ids.len()) {
                return Err(Error::Inference {
                    message: "options do not fit in the configured token budget".to_string(),
                });
            }
            Ok(Prepared {
                sequence: Sequence { ids, markers },
                kind: question.kind,
                options,
                labels,
                levels,
            })
        })
        .collect()
}

#[derive(Default)]
struct PythonJsonFormatter;

impl serde_json::ser::Formatter for PythonJsonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

fn serialize_state(state: &Value) -> Result<String> {
    if let Some(state) = state.as_str() {
        return Ok(state.to_string());
    }
    let mut bytes = Vec::new();
    state
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes,
            PythonJsonFormatter,
        ))
        .map_err(|error| Error::Inference {
            message: format!("serializing State: {error}"),
        })?;
    String::from_utf8(bytes).map_err(|error| Error::Inference {
        message: format!("serializing State as UTF-8: {error}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_state_uses_python_json_spacing() {
        let state = serde_json::json!({"ключ": "значение", "nested": [true, null]});
        assert_eq!(
            serialize_state(&state).unwrap(),
            r#"{"ключ": "значение", "nested": [true, null]}"#
        );
    }

    #[test]
    fn criteria_render_like_the_checkpoint_runtime() {
        let call = Call::from_bytes(br#"{"state":"x","questions":{"choice":{"type":"choice","instructions":"x","criteria":{"nil":null,"empty":"","false":false,"zero":0,"array":[],"object":{},"yes":"ok","structured":{"flag":true}}},"score":{"type":"score","instructions":"x","criteria":["low","high"]},"noul":{"type":"noul","instructions":"x","criteria":{"false":false,"true":0}}}}"#).unwrap();
        let mut questions = call.questions.iter();
        let (_, choice) = questions.next().unwrap();
        let (_, score) = questions.next().unwrap();
        let (_, noul) = questions.next().unwrap();
        assert_eq!(
            render_options(choice).0,
            [
                "nil",
                "empty",
                "false",
                "zero",
                "array",
                "object",
                "yes: ok",
                "structured: {'flag': True}"
            ]
        );
        assert_eq!(render_options(score).0, ["level 0: low", "level 1: high"]);
        assert_eq!(
            render_options(noul).0,
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        assert_eq!(python_quote("it's fine"), r#""it's fine""#);
        assert_eq!(python_quote("a\u{1b}b"), r#"'a\x1bb'"#);
    }
}
