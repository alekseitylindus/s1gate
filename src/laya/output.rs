//! Public response rendering and calibrated probability helpers.

use serde_json::{Map, Value};

use crate::call::{Call, QuestionType};
use crate::error::{Error, Result};

use super::config::AgentConfig;
use super::prompt::Prepared;

fn temperature(agent: &AgentConfig, kind: QuestionType, options: usize) -> f32 {
    let size = if options <= 2 {
        "2"
    } else if options <= 5 {
        "3-5"
    } else if options <= 10 {
        "6-10"
    } else {
        "11+"
    };
    agent
        .temperature_by_options
        .get(&format!("{kind}:{size}"))
        .copied()
        .or_else(|| agent.temperature.get(kind.index()).copied())
        .unwrap_or(1.0)
        .max(1e-3)
}

fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let values = logits
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    values.into_iter().map(|value| value / sum).collect()
}

fn round_even(value: f32) -> Value {
    let scaled = f64::from(value) * 10_000.0;
    let lower = scaled.floor();
    let fraction = scaled - lower;
    let units = if fraction < 0.5 {
        lower
    } else if fraction > 0.5 || lower.rem_euclid(2.0) != 0.0 {
        lower + 1.0
    } else {
        lower
    };
    let rounded = units / 10_000.0;
    serde_json::Number::from_f64(rounded)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn confidence(probabilities: &[f32]) -> f32 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy = probabilities
        .iter()
        .map(|value| value * value.max(1e-12).ln())
        .sum::<f32>();
    (1.0 + entropy / (probabilities.len() as f32).ln()).clamp(0.0, 1.0)
}

pub(super) fn format_result(
    call: &Call,
    agent: &AgentConfig,
    prepared: Vec<Prepared>,
    logits: Vec<Vec<f32>>,
    actions: Vec<Vec<f32>>,
) -> Result<Value> {
    let question_count = call.questions.iter().count();
    if prepared.len() != question_count
        || logits.len() != question_count
        || actions.len() != question_count
    {
        return Err(Error::Inference {
            message: "model output does not match the Question batch".to_string(),
        });
    }
    let mut answers = Map::new();
    for (row, (id, _)) in call.questions.iter().enumerate() {
        let item = prepared.get(row).ok_or_else(|| Error::Inference {
            message: format!("missing prepared Question `{id}`"),
        })?;
        let logit_row = logits.get(row).ok_or_else(|| Error::Inference {
            message: format!("missing logits for Question `{id}`"),
        })?;
        let option_logits =
            logit_row
                .get(..item.options.len())
                .ok_or_else(|| Error::Inference {
                    message: format!("not enough logits for Question `{id}`"),
                })?;
        let probabilities = stable_softmax(
            &option_logits
                .iter()
                .map(|value| *value / temperature(agent, item.kind, item.options.len()))
                .collect::<Vec<_>>(),
        );
        let action = actions
            .get(row)
            .and_then(|row| row.first())
            .copied()
            .ok_or_else(|| Error::Inference {
                message: format!("missing action probability for Question `{id}`"),
            })?;
        let mut answer = Map::new();
        answer.insert("type".to_string(), Value::String(item.kind.to_string()));
        answer.insert(
            "rl_agent".to_string(),
            serde_json::json!({ "act_probability": f64::from(action) }),
        );
        match item.kind {
            QuestionType::Choice => {
                let best = probabilities
                    .iter()
                    .enumerate()
                    .reduce(|best, candidate| {
                        if candidate.1 > best.1 {
                            candidate
                        } else {
                            best
                        }
                    })
                    .map(|(index, _)| index)
                    .ok_or_else(|| Error::Inference {
                        message: format!("no probabilities for Question `{id}`"),
                    })?;
                answer.insert(
                    "choice".to_string(),
                    Value::String(
                        item.labels
                            .get(best)
                            .ok_or_else(|| Error::Inference {
                                message: format!("missing choice label for Question `{id}`"),
                            })?
                            .clone(),
                    ),
                );
                let values = item
                    .labels
                    .iter()
                    .zip(&probabilities)
                    .map(|(label, probability)| (label.clone(), round_even(*probability)))
                    .collect();
                answer.insert("probabilities".to_string(), Value::Object(values));
                answer.insert(
                    "confidence".to_string(),
                    round_even(confidence(&probabilities)),
                );
            }
            QuestionType::Score => {
                let score = probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, probability)| index as f32 * probability)
                    .sum::<f32>();
                answer.insert("score".to_string(), round_even(score));
                let legend = item
                    .levels
                    .iter()
                    .enumerate()
                    .map(|(index, level)| (index.to_string(), Value::String(level.clone())))
                    .collect();
                let values = item
                    .levels
                    .iter()
                    .zip(&probabilities)
                    .enumerate()
                    .map(|(index, (_, probability))| (index.to_string(), round_even(*probability)))
                    .collect();
                answer.insert("legend".to_string(), Value::Object(legend));
                answer.insert("probabilities".to_string(), Value::Object(values));
                answer.insert(
                    "confidence".to_string(),
                    round_even(confidence(&probabilities)),
                );
            }
            QuestionType::Noul => {
                let probability =
                    probabilities
                        .get(1)
                        .copied()
                        .ok_or_else(|| Error::Inference {
                            message: format!("missing true probability for Question `{id}`"),
                        })?;
                answer.insert("noul".to_string(), round_even(probability));
            }
        }
        answers.insert(id.to_string(), Value::Object(answer));
    }
    let input_tokens = prepared
        .iter()
        .map(|item| item.sequence.ids.len())
        .sum::<usize>();
    Ok(serde_json::json!({
        "model": "rl-agent", "answers": answers,
        "usage": { "input_tokens": input_tokens, "output_tokens": 0 }
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::call::Call;
    use crate::laya::prompt::{Prepared, Sequence};

    #[test]
    fn stable_softmax_handles_large_logits() {
        let probabilities = stable_softmax(&[1000.0, 999.0]);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probabilities[0] > probabilities[1]);
    }

    #[test]
    fn confidence_is_zero_for_uniform_and_one_for_certain() {
        assert!(confidence(&[0.5, 0.5]).abs() < 1e-6);
        assert!((confidence(&[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reported_values_use_half_even_at_four_decimals() {
        assert_eq!(round_even(0.12344), serde_json::json!(0.1234));
        assert_eq!(round_even(0.12345), serde_json::json!(0.1235));
        assert_eq!(round_even(0.12355), serde_json::json!(0.1235));
        assert_eq!(round_even(0.12356), serde_json::json!(0.1236));
    }

    #[test]
    fn answers_match_the_checkpoint_public_shape() {
        let call = Call::from_bytes(br#"{"state":"x","questions":{"choice":{"type":"choice","instructions":"x","criteria":["no","yes"]},"noul":{"type":"noul","instructions":"x"}}}"#).unwrap();
        let prepared = call
            .questions
            .iter()
            .map(|(_, question)| {
                let (options, labels, levels) = match question.kind {
                    QuestionType::Choice => (
                        vec!["no".to_string(), "yes".to_string()],
                        vec!["no".to_string(), "yes".to_string()],
                        Vec::new(),
                    ),
                    QuestionType::Noul => (
                        vec!["false".to_string(), "true".to_string()],
                        Vec::new(),
                        Vec::new(),
                    ),
                    QuestionType::Score => unreachable!(),
                };
                Prepared {
                    sequence: Sequence {
                        ids: vec![1, 2, 3],
                        markers: vec![1, 2],
                    },
                    kind: question.kind,
                    options,
                    labels,
                    levels,
                }
            })
            .collect();
        let agent = AgentConfig {
            max_len: 512,
            head_max_len: 192,
            head_layers: 2,
            temperature: vec![1.0; 3],
            temperature_by_options: BTreeMap::new(),
            act_costs: BTreeMap::new(),
        };
        let result = format_result(
            &call,
            &agent,
            prepared,
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.25], vec![0.25]],
        )
        .unwrap();
        assert_eq!(
            result,
            serde_json::json!({
                "model": "rl-agent", "answers": {
                    "choice": {"type": "choice", "choice": "no", "probabilities": {"no": 0.5, "yes": 0.5}, "confidence": 0.0, "rl_agent": {"act_probability": 0.25}},
                    "noul": {"type": "noul", "noul": 0.5, "rl_agent": {"act_probability": 0.25}}
                }, "usage": {"input_tokens": 6, "output_tokens": 0}
            })
        );
    }
}
