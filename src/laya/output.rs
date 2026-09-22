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
    let mut values = logits
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    for value in &mut values {
        *value /= sum;
    }
    values
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

/// Render the Answers for a Call from the model output over its prepared Questions.
///
/// # Errors
///
/// Fails when the output row count does not match the Question batch, when a logits or action row
/// is missing, when a Question's Options outrun its logits, or when a choice label or `true`
/// probability is missing.
pub(super) fn format_result(
    call: &Call,
    agent: &AgentConfig,
    prepared: &[Prepared],
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
            "action".to_string(),
            serde_json::json!({ "act_probability": round_even(action) }),
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
                        item.responses
                            .labels()
                            .get(best)
                            .ok_or_else(|| Error::Inference {
                                message: format!("missing choice label for Question `{id}`"),
                            })?
                            .clone(),
                    ),
                );
                let values = item
                    .responses
                    .labels()
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
                    .responses
                    .levels()
                    .iter()
                    .enumerate()
                    .map(|(index, level)| (index.to_string(), Value::String(level.clone())))
                    .collect();
                let values = item
                    .responses
                    .levels()
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
                answer.insert(
                    "confidence".to_string(),
                    round_even(probability.max(1.0 - probability)),
                );
            }
        }
        answers.insert(id.to_string(), Value::Object(answer));
    }
    let input_tokens = prepared
        .iter()
        .map(|item| item.sequence.ids.len())
        .sum::<usize>();
    Ok(serde_json::json!({
        "model": call.model, "answers": answers,
        "usage": { "input_tokens": input_tokens, "output_tokens": 0 }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call::Call;
    use crate::laya::prompt::{Prepared, Responses, Sequence};

    /// The per-Question-Type fallback temperatures a Checkpoint agent carries in
    /// `rl_agent_config.json`, chosen distinct so a test tells them apart.
    const FALLBACK: [f32; 3] = [1.3, 0.7, 1.1];

    /// The agent configuration `sizes` describes: each `(kind:size, temperature)` pair is one
    /// entry of `temperature_by_options`, and every Question Type falls back to [`FALLBACK`].
    fn agent(sizes: &[(&str, f32)]) -> AgentConfig {
        AgentConfig {
            max_len: 512,
            head_max_len: 192,
            head_layers: 2,
            temperature: FALLBACK.to_vec(),
            temperature_by_options: sizes
                .iter()
                .map(|(key, value)| ((*key).to_string(), *value))
                .collect(),
        }
    }

    /// One prepared Question reporting `names` as its Options or Levels, in a Sequence of
    /// `tokens` ids, so a test can tell the whole-call token total from one Question's own.
    fn prepared(kind: QuestionType, names: &[&str], tokens: usize) -> Prepared {
        let names = names
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
        let responses = match kind {
            QuestionType::Choice => Responses::Labels(names.clone()),
            QuestionType::Score => Responses::Levels(names.clone()),
            QuestionType::Noul => Responses::Unnamed,
        };
        Prepared {
            sequence: Sequence {
                ids: vec![1; tokens],
                markers: (0..names.len()).collect(),
            },
            kind,
            options: names,
            responses,
        }
    }

    /// Logits that put all their weight on the first Option or Level.
    fn leading(count: usize) -> Vec<f32> {
        let mut row = vec![0.0; count];
        row[0] = 2.0;
        row
    }

    /// The probabilities the answers of `result` report, in the order they are keyed.
    fn reported(result: &Value, id: &str) -> Vec<f64> {
        result["answers"][id]["probabilities"]
            .as_object()
            .expect("an Answer with probabilities")
            .values()
            .map(|value| value.as_f64().expect("a numeric probability"))
            .collect()
    }

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
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{"choice":{"type":"choice","instructions":"x","criteria":["no","yes"]},"noul":{"type":"noul","instructions":"x"}}}"#).unwrap();
        let prepared = vec![
            prepared(QuestionType::Choice, &["no", "yes"], 3),
            prepared(QuestionType::Noul, &["false", "true"], 3),
        ];
        let result = format_result(
            &call,
            &agent(&[]),
            &prepared,
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.25], vec![0.25]],
        )
        .unwrap();
        assert_eq!(
            result,
            serde_json::json!({
                "model": "convaiinnovations/laya", "answers": {
                    "choice": {"type": "choice", "choice": "no", "probabilities": {"no": 0.5, "yes": 0.5}, "confidence": 0.0, "action": {"act_probability": 0.25}},
                    "noul": {"type": "noul", "noul": 0.5, "confidence": 0.5, "action": {"act_probability": 0.25}}
                }, "usage": {"input_tokens": 6, "output_tokens": 0}
            })
        );
    }

    /// A `noul` Answer reports the true side's probability against the Confidence of the stronger
    /// side, so a true side weaker than a half reports the false side's probability as its
    /// confidence — the oracle's `max(p[1], 1 - p[1])`. Both come from the unrounded calibrated
    /// probabilities: a third on the true side reports `0.3333` and `0.6667`.
    #[test]
    fn a_noul_answer_reports_the_stronger_sides_probability() {
        let call = Call::from_bytes(
            br#"{"model":"convaiinnovations/laya","state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}"#,
        )
        .unwrap();
        // `temperature` falls back to 1.1 for `noul`, so a logit of `ln(3) * 1.1` leaves one side
        // three times the other's odds.
        let odds = (3.0f32).ln() * FALLBACK[2];
        for (logits, true_side, confidence) in [
            (vec![0.0, odds], 0.75, 0.75),
            (vec![odds, 0.0], 0.25, 0.75),
            (vec![0.0, (0.5f32).ln() * FALLBACK[2]], 0.3333, 0.6667),
        ] {
            let result = format_result(
                &call,
                &agent(&[]),
                &[prepared(QuestionType::Noul, &["false", "true"], 3)],
                vec![logits],
                vec![vec![0.5]],
            )
            .unwrap();

            assert_eq!(result["answers"]["q"]["noul"], serde_json::json!(true_side));
            assert_eq!(
                result["answers"]["q"]["confidence"],
                serde_json::json!(confidence)
            );
        }
    }

    /// A `noul` Question calibrates by its own Type and its two Options: `noul:2` is the entry the
    /// released Checkpoint fits, and the per-Type `temperature` is only what a Checkpoint without
    /// that entry falls back to. A true-side logit of `-2` is worth `e^-1` of the false side's odds
    /// at a Calibration Temperature of two, and `e^-1.8181` of them at the fallback of 1.1.
    #[test]
    fn a_noul_question_takes_its_own_two_option_calibration_temperature() {
        let call = Call::from_bytes(
            br#"{"model":"convaiinnovations/laya","state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}"#,
        )
        .unwrap();
        let judge = |agent: &AgentConfig| {
            format_result(
                &call,
                agent,
                &[prepared(QuestionType::Noul, &["false", "true"], 3)],
                vec![vec![0.0, -2.0]],
                vec![vec![0.5]],
            )
            .unwrap()
        };

        let fitted = judge(&agent(&[("noul:2", 2.0)]));
        assert_eq!(fitted["answers"]["q"]["noul"], serde_json::json!(0.2689));
        assert_eq!(
            fitted["answers"]["q"]["confidence"],
            serde_json::json!(0.7311)
        );

        let fallback = judge(&agent(&[]));
        assert_eq!(fallback["answers"]["q"]["noul"], serde_json::json!(0.1397));
        assert_eq!(
            fallback["answers"]["q"]["confidence"],
            serde_json::json!(0.8603)
        );
    }

    /// The `6-10` band of `temperature_by_options` is looked up, not skipped, and the bands meet
    /// where they say they do: a `score` of five Options takes `3-5`, six and ten take `6-10`, and
    /// eleven takes `11+`. The released Checkpoint fits a `choice:6-10` temperature, so a Question
    /// of six to ten Options would otherwise take the per-Type fallback in silence.
    #[test]
    fn calibration_covers_the_six_to_ten_bucket() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{
            "five":{"type":"score","instructions":"x","criteria":["a","b","c","d","e"]},
            "six":{"type":"score","instructions":"x","criteria":["a","b","c","d","e","f"]},
            "ten":{"type":"score","instructions":"x","criteria":["a","b","c","d","e","f","g","h","i","j"]},
            "eleven":{"type":"score","instructions":"x","criteria":["a","b","c","d","e","f","g","h","i","j","k"]}}}"#).unwrap();
        let five = ["a", "b", "c", "d", "e"];
        let six = ["a", "b", "c", "d", "e", "f"];
        let ten = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
        let eleven = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"];
        let prepared = vec![
            prepared(QuestionType::Score, &five, 3),
            prepared(QuestionType::Score, &six, 3),
            prepared(QuestionType::Score, &ten, 3),
            prepared(QuestionType::Score, &eleven, 3),
        ];
        let result = format_result(
            &call,
            &agent(&[("score:3-5", 0.25), ("score:6-10", 2.0), ("score:11+", 4.0)]),
            &prepared,
            vec![leading(5), leading(6), leading(10), leading(11)],
            vec![vec![0.5]; 4],
        )
        .unwrap();

        // Leading probabilities, `1 / (1 + (k - 1) * exp(-2 / T))` at each band's temperature.
        assert_eq!(reported(&result, "five")[0], 0.9987);
        assert_eq!(reported(&result, "six")[0], 0.3522);
        assert_eq!(reported(&result, "ten")[0], 0.232);
        assert_eq!(reported(&result, "eleven")[0], 0.1415);
    }

    /// A `score` Answer reports the Levels the caller gave, in the caller's order, beside the
    /// calibrated distribution over that order, its Confidence and the Action.
    #[test]
    fn a_score_answer_reports_the_callers_levels_in_order() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{"urgency":{"type":"score","instructions":"How urgent?","criteria":["immediate","low","normal","high"]}}}"#).unwrap();
        let result = format_result(
            &call,
            &agent(&[]),
            &[prepared(
                QuestionType::Score,
                &["immediate", "low", "normal", "high"],
                3,
            )],
            vec![vec![0.0; 4]],
            // A number only half-even rounding at four decimals turns into 0.1235.
            vec![vec![0.123_456_78]],
        )
        .unwrap();

        assert_eq!(
            result,
            serde_json::json!({
                "model": "convaiinnovations/laya", "answers": {
                    "urgency": {
                        "type": "score",
                        "score": 1.5,
                        "legend": {"0": "immediate", "1": "low", "2": "normal", "3": "high"},
                        "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
                        "confidence": 0.0,
                        "action": {"act_probability": 0.1235}
                    }
                }, "usage": {"input_tokens": 3, "output_tokens": 0}
            })
        );
    }

    /// The Calibration Temperature comes from `temperature_by_options` under the Question's own
    /// Type and option count, falling back to `temperature` for that Type.
    #[test]
    fn calibration_follows_the_question_type_and_option_count() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{
            "score2":{"type":"score","instructions":"x","criteria":["a","b"]},
            "choice2":{"type":"choice","instructions":"x","criteria":["a","b"]},
            "score4":{"type":"score","instructions":"x","criteria":["a","b","c","d"]},
            "choice4":{"type":"choice","instructions":"x","criteria":["a","b","c","d"]},
            "score9":{"type":"score","instructions":"x","criteria":["a","b","c","d","e","f","g","h","i"]},
            "noul":{"type":"noul","instructions":"x"}}}"#).unwrap();
        let prepared = vec![
            prepared(QuestionType::Score, &["a", "b"], 3),
            prepared(QuestionType::Choice, &["a", "b"], 3),
            prepared(QuestionType::Score, &["a", "b", "c", "d"], 3),
            prepared(QuestionType::Choice, &["a", "b", "c", "d"], 3),
            prepared(
                QuestionType::Score,
                &["a", "b", "c", "d", "e", "f", "g", "h", "i"],
                3,
            ),
            prepared(QuestionType::Noul, &["false", "true"], 3),
        ];
        let result = format_result(
            &call,
            &agent(&[("score:2", 2.0), ("choice:2", 0.5), ("score:3-5", 0.25)]),
            &prepared,
            vec![
                leading(2),
                leading(2),
                leading(4),
                leading(4),
                leading(9),
                leading(2),
            ],
            vec![vec![0.5]; 6],
        )
        .unwrap();

        // Leading probabilities, `1 / (1 + exp(-2 / T))`: `score:2` at 2.0, `choice:2` at 0.5,
        // `score:3-5` at 0.25, and the fallbacks 1.3, 0.7 and 1.1 for a `choice` of four, a
        // `score` of nine and a `noul`.
        assert_eq!(reported(&result, "score2"), [0.7311, 0.2689]);
        assert_eq!(reported(&result, "choice2"), [0.982, 0.018]);
        assert_eq!(reported(&result, "score4"), [0.999, 0.0003, 0.0003, 0.0003]);
        assert_eq!(
            reported(&result, "choice4"),
            [0.6082, 0.1306, 0.1306, 0.1306]
        );
        assert_eq!(reported(&result, "score9")[0], 0.6852);
        assert_eq!(result["answers"]["noul"]["noul"], serde_json::json!(0.1397));
    }

    /// A fitted temperature below the clamp is floored at `1e-3`, so the `score:2` temperature
    /// of `0.0001` scales the logits by a thousandth, not by a ten-thousandth.
    #[test]
    fn a_calibration_temperature_below_the_clamp_is_floored() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{"q":{"type":"score","instructions":"x","criteria":["low","high"]}}}"#).unwrap();
        let result = format_result(
            &call,
            &agent(&[("score:2", 0.000_1)]),
            &[prepared(QuestionType::Score, &["low", "high"], 3)],
            vec![vec![0.000_5, 0.0]],
            vec![vec![0.5]],
        )
        .unwrap();

        assert_eq!(reported(&result, "q"), [0.6225, 0.3775]);
        assert_eq!(result["answers"]["q"]["score"], serde_json::json!(0.3775));
    }

    /// A `score` is the expectation over the unrounded calibrated distribution: rounding the
    /// probabilities first would move it to 4.5.
    #[test]
    fn a_score_is_the_expectation_over_unrounded_probabilities() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{"q":{"type":"score","instructions":"x","criteria":["L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11"]}}}"#).unwrap();
        // Eleven Levels whose last carries 0.00002 of the calibrated mass and whose other ten
        // carry 0.099998 each, at a calibration temperature of one: the logit that says so is the
        // `z` with `e^z = 0.00002 * 10 / (1 - 0.00002)`.
        let mut logits = vec![0.0f32; 11];
        logits[10] = (2.000_04e-4f32).ln();
        let names = [
            "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11",
        ];
        let result = format_result(
            &call,
            &agent(&[("score:11+", 1.0)]),
            &[prepared(QuestionType::Score, &names, 3)],
            vec![logits],
            vec![vec![0.5]],
        )
        .unwrap();

        assert_eq!(result["answers"]["q"]["score"], serde_json::json!(4.5001));
        assert_eq!(reported(&result, "q")[0], 0.1);
        assert_eq!(reported(&result, "q")[10], 0.0);
    }

    /// Reported probabilities are each rounded as they are; they are not renormalized back to a
    /// distribution that sums to one.
    #[test]
    fn rounded_probabilities_are_not_renormalized() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{"q":{"type":"choice","instructions":"x","criteria":["a","b","c"]}}}"#).unwrap();
        let result = format_result(
            &call,
            &agent(&[]),
            &[prepared(QuestionType::Choice, &["a", "b", "c"], 3)],
            vec![vec![0.0; 3]],
            vec![vec![0.5]],
        )
        .unwrap();

        let probabilities = reported(&result, "q");
        assert_eq!(probabilities, [0.3333, 0.3333, 0.3333]);
        assert!(
            probabilities.iter().sum::<f64>() < 1.0,
            "{probabilities:?} should stay as rounded, not be renormalized"
        );
    }

    /// One Call of all three Question Types is judged as a whole: every Question's Answer appears
    /// under its own id, a `noul` Answer carries the true side's probability and its Confidence in
    /// place of a distribution and a legend, and `usage.input_tokens` is the total the whole call
    /// took.
    #[test]
    fn a_mixed_call_reports_one_whole_call_token_total() {
        let call = Call::from_bytes(br#"{"model":"convaiinnovations/laya","state":"x","questions":{
            "route":{"type":"choice","instructions":"x","criteria":["billing","shipping","account"]},
            "urgency":{"type":"score","instructions":"x","criteria":["low","high","normal"]},
            "churn_risk":{"type":"noul","instructions":"x"}}}"#).unwrap();
        let result = format_result(
            &call,
            &agent(&[]),
            &[
                prepared(QuestionType::Choice, &["billing", "shipping", "account"], 4),
                prepared(QuestionType::Score, &["low", "high", "normal"], 5),
                prepared(QuestionType::Noul, &["false", "true"], 3),
            ],
            vec![vec![0.0; 3], vec![0.0; 3], vec![0.0; 2]],
            vec![vec![0.25], vec![0.75], vec![0.5]],
        )
        .unwrap();

        assert_eq!(result["answers"]["route"]["type"], "choice");
        assert_eq!(result["answers"]["route"]["choice"], "billing");
        assert_eq!(
            result["answers"]["urgency"]["legend"],
            serde_json::json!({"0": "low", "1": "high", "2": "normal"})
        );
        assert_eq!(
            result["answers"]["urgency"]["action"]["act_probability"],
            0.75
        );
        assert_eq!(
            result["answers"]["churn_risk"],
            serde_json::json!({
                "type": "noul", "noul": 0.5, "confidence": 0.5,
                "action": {"act_probability": 0.5}
            }),
            "a noul Answer reports no distribution and no legend"
        );
        assert_eq!(
            result["usage"],
            serde_json::json!({"input_tokens": 12, "output_tokens": 0})
        );
    }
}
