//! Python-runtime-compatible prompt construction and token budgeting.

use std::fs;
use std::io;
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
    pub(super) ids: Vec<u32>,
    pub(super) markers: Vec<usize>,
}

/// The response names a Question Type reports, so an Answer can only read the
/// names its own Question Type populates.
#[derive(Debug)]
pub(super) enum Responses {
    Labels(Vec<String>),
    Levels(Vec<String>),
    Unnamed,
}

impl Responses {
    /// The names a `choice` Answer reports; empty for every other Question Type.
    pub(super) fn labels(&self) -> &[String] {
        match self {
            Self::Labels(labels) => labels,
            Self::Levels(_) | Self::Unnamed => &[],
        }
    }

    /// The names a `score` Answer reports; empty for every other Question Type.
    pub(super) fn levels(&self) -> &[String] {
        match self {
            Self::Levels(levels) => levels,
            Self::Labels(_) | Self::Unnamed => &[],
        }
    }
}

#[derive(Debug)]
pub(super) struct Prepared {
    pub(super) sequence: Sequence,
    pub(super) kind: QuestionType,
    pub(super) options: Vec<String>,
    pub(super) responses: Responses,
}

pub(super) fn special_tokens(directory: &Path, tokenizer: &Tokenizer) -> Result<SpecialTokens> {
    let config: Value = read_json(&directory.join("tokenizer/tokenizer_config.json"))?;
    let token = |name: &str| -> Result<(String, u32)> {
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

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|error| Error::Inference {
            message: format!("tokenizing prompt: {error}"),
        })?;
    Ok(encoding.get_ids().to_vec())
}

/// One Criteria value as the original implementation renders it into a prompt: a string as it
/// stands, anything structured as compact JSON.
fn render_criterion(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        value => python_json(value, "a Criteria value"),
    }
}

/// Whether a Criteria value carries a description. Only `null` and the empty string do not, so
/// `0` and `false` describe an Option like any other value does.
fn describes(value: &Value) -> bool {
    !(value.is_null() || value.as_str() == Some(""))
}

/// The rendered Options of a Question and the response names its Question Type reports.
struct Rendered {
    options: Vec<String>,
    responses: Responses,
}

fn render_options(question: &Question) -> Result<Rendered> {
    match question.kind {
        QuestionType::Choice => match question.criteria.as_ref() {
            Some(Criteria::Object(options)) => {
                let mut rendered = Vec::with_capacity(options.len());
                let mut labels = Vec::with_capacity(options.len());
                for (name, value) in options {
                    rendered.push(if describes(value) {
                        format!("{name}: {}", render_criterion(value)?)
                    } else {
                        name.clone()
                    });
                    labels.push(name.clone());
                }
                Ok(Rendered {
                    options: rendered,
                    responses: Responses::Labels(labels),
                })
            }
            // `call.rs` rejects both of these before a prompt is prepared.
            Some(Criteria::List(_)) => Err(Error::Inference {
                message: "choice Criteria must be an object".to_string(),
            }),
            None => Err(Error::Inference {
                message: "choice Question has no Criteria".to_string(),
            }),
        },
        QuestionType::Score => match question.criteria.as_ref() {
            Some(Criteria::List(levels)) => {
                let levels = levels
                    .iter()
                    .map(render_criterion)
                    .collect::<Result<Vec<_>>>()?;
                let rendered = levels
                    .iter()
                    .enumerate()
                    .map(|(index, level)| format!("level {index}: {level}"))
                    .collect();
                Ok(Rendered {
                    options: rendered,
                    responses: Responses::Levels(levels),
                })
            }
            // `call.rs` rejects both of these before a prompt is prepared.
            Some(Criteria::Object(_)) => Err(Error::Inference {
                message: "score Criteria must be an array".to_string(),
            }),
            None => Err(Error::Inference {
                message: "score Question has no Criteria".to_string(),
            }),
        },
        QuestionType::Noul => {
            let descriptions = match question.criteria.as_ref() {
                None => None,
                Some(Criteria::Object(options)) => Some(options),
                // `call.rs` rejects this before a prompt is prepared.
                Some(Criteria::List(_)) => {
                    return Err(Error::Inference {
                        message: "noul Criteria must be an object".to_string(),
                    });
                }
            };
            let render = |name: &str, fallback: &str| -> Result<String> {
                match descriptions.and_then(|options| options.iter().find(|(key, _)| key == name)) {
                    Some((_, value)) if describes(value) => render_criterion(value),
                    _ => Ok(fallback.to_string()),
                }
            };
            Ok(Rendered {
                options: vec![
                    format!(
                        "false: {}",
                        render("false", "no, the statement does not hold")?
                    ),
                    format!("true: {}", render("true", "yes, the statement holds")?),
                ],
                responses: Responses::Unnamed,
            })
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
            let Rendered { options, responses } = render_options(question)?;
            let instructions = match &question.instructions {
                Value::String(text) => text.clone(),
                value => python_json(value, "instructions")?,
            };
            let mut head = encode(
                tokenizer,
                &format!(
                    "{} question: {}",
                    question.kind,
                    instructions.replace(&special.mask_text, " ")
                ),
            )?;
            let mut option_ids = Vec::with_capacity(options.len());
            for option in &options {
                let mut ids = Vec::with_capacity(1 + MAX_OPTION_TOKENS);
                ids.push(special.mask);
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
            let mut ids = Vec::with_capacity(config.max_len);
            ids.push(special.cls);
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
                responses,
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
    python_json(state, "state")
}

/// Serialize `value` the way the original implementation writes it: `", "` between items, `": "`
/// after a key, and non-ASCII left as it stands rather than escaped. `what` names the value in the
/// error.
fn python_json(value: &Value, what: &str) -> Result<String> {
    let mut bytes = Vec::new();
    value
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes,
            PythonJsonFormatter,
        ))
        .map_err(|error| Error::Inference {
            message: format!("serializing {what}: {error}"),
        })?;
    String::from_utf8(bytes).map_err(|error| Error::Inference {
        message: format!("serializing {what} as utf-8: {error}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Options and response names a Question renders to, unwrapped: rendering a value that
    /// came out of JSON cannot fail.
    fn render(question: &Question) -> Rendered {
        render_options(question).unwrap()
    }

    #[test]
    fn structured_state_uses_python_json_spacing() {
        let state = serde_json::json!({"ключ": "значение", "nested": [true, null]});
        assert_eq!(
            serialize_state(&state).unwrap(),
            r#"{"ключ": "значение", "nested": [true, null]}"#
        );
    }

    #[test]
    fn score_levels_render_in_the_callers_order() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{"urgency":{"type":"score","instructions":"x","criteria":["immediate","low","normal","high"]}}}"#).unwrap();
        let (_, question) = call.questions.iter().next().unwrap();

        let rendered = render(question);

        assert_eq!(
            rendered.options,
            [
                "level 0: immediate",
                "level 1: low",
                "level 2: normal",
                "level 3: high"
            ]
        );
        assert_eq!(
            rendered.responses.levels(),
            ["immediate", "low", "normal", "high"]
        );
        assert!(rendered.responses.labels().is_empty());
    }

    /// A `choice` description carries the value the caller wrote. `null` and the empty string mean
    /// there is no description; strings render as written and structured values render as JSON.
    #[test]
    fn criteria_render_their_descriptions() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{"choice":{"type":"choice","instructions":"x","criteria":{"nil":null,"empty":"","list":["a",{"count":2}],"object":{"flag":true,"\u043a\u043b\u044e\u0447":"\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435"},"yes":"ok"}},"score":{"type":"score","instructions":"x","criteria":["low","high"]}}}"#).unwrap();
        let mut questions = call.questions.iter();
        let (_, choice) = questions.next().unwrap();
        let (_, score) = questions.next().unwrap();

        assert_eq!(
            render(choice).options,
            [
                "nil",
                "empty",
                "list: [\"a\", {\"count\": 2}]",
                "object: {\"flag\": true, \"ключ\": \"значение\"}",
                "yes: ok"
            ]
        );
        assert_eq!(render(score).options, ["level 0: low", "level 1: high"]);
    }

    /// A `noul` Question always renders the false Option and then the true one, whatever order
    /// its Criteria wrote them in, with the descriptions its Criteria give and the stock sentences
    /// only where they give none. Its Options are named by nothing, which is why a `noul` Answer
    /// carries neither probabilities nor a legend.
    #[test]
    fn noul_renders_the_false_option_then_the_true_one() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{
            "described":{"type":"noul","instructions":"x","criteria":{"true":"yes","false":"no"}},
            "partial":{"type":"noul","instructions":"x","criteria":{"true":"yes"}},
            "structured":{"type":"noul","instructions":"x","criteria":{"false":{"reason":"stays"},"true":["leaves","churns"]}},
            "blank":{"type":"noul","instructions":"x","criteria":{"false":"","true":""}},
            "plain":{"type":"noul","instructions":"x"}}}"#).unwrap();
        let options = |id: &str| {
            let (_, question) = call.questions.iter().find(|(name, _)| *name == id).unwrap();
            render(question).options
        };

        assert_eq!(options("described"), ["false: no", "true: yes"]);
        assert_eq!(
            options("partial"),
            ["false: no, the statement does not hold", "true: yes"]
        );
        assert_eq!(
            options("structured"),
            [
                "false: {\"reason\": \"stays\"}",
                "true: [\"leaves\", \"churns\"]"
            ]
        );
        assert_eq!(
            options("blank"),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        assert_eq!(
            options("plain"),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );

        let (_, plain) = call
            .questions
            .iter()
            .find(|(name, _)| *name == "plain")
            .unwrap();
        let rendered = render(plain);
        assert!(rendered.responses.labels().is_empty());
        assert!(rendered.responses.levels().is_empty());
    }
}
