//! The native Laya backend.  This module owns prompt construction, checkpoint weight materializing,
//! and the one MLX forward pass used by `infer`.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use half::f16;
use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops::{self, *};
use serde::Deserialize;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::call::{Call, Criteria, Question, QuestionType};
use crate::checkpoint;
use crate::error::{Error, Result};
use crate::model_source::ModelSource;
use crate::store::Store;

const MAX_OPTION_TOKENS: usize = 48;
const MASK_FILL: f32 = -1e4;
const LN_EPS: f32 = 1e-5;

#[derive(Debug, Deserialize)]
struct EncoderConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    #[serde(default = "one")]
    num_attention_heads: usize,
    #[serde(default = "default_local_attention")]
    local_attention: usize,
    #[serde(default = "default_global_interval")]
    global_attn_every_n_layers: usize,
    #[serde(default = "default_global_rope")]
    global_rope_theta: f32,
    #[serde(default = "default_local_rope")]
    local_rope_theta: f32,
    #[serde(default = "default_max_position")]
    max_position_embeddings: usize,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    rope_parameters: Option<BTreeMap<String, RopeParameters>>,
    #[serde(default = "default_norm_eps")]
    norm_eps: f32,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default = "default_rope_type")]
    rope_type: String,
    #[serde(default)]
    rope_theta: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    #[serde(default = "default_max_len")]
    max_len: usize,
    #[serde(default = "default_head_max_len")]
    head_max_len: usize,
    #[serde(default)]
    head_layers: usize,
    #[serde(default = "default_temperatures")]
    temperature: Vec<f32>,
    #[serde(default)]
    temperature_by_options: BTreeMap<String, f32>,
    #[serde(default)]
    act_costs: BTreeMap<String, Value>,
}

fn one() -> usize {
    1
}
fn default_local_attention() -> usize {
    128
}
fn default_global_interval() -> usize {
    3
}
fn default_global_rope() -> f32 {
    160_000.0
}
fn default_local_rope() -> f32 {
    10_000.0
}
fn default_max_position() -> usize {
    8192
}
fn default_norm_eps() -> f32 {
    LN_EPS
}
fn default_rope_type() -> String {
    "default".to_string()
}
fn default_max_len() -> usize {
    512
}
fn default_head_max_len() -> usize {
    192
}
fn default_temperatures() -> Vec<f32> {
    vec![1.0, 1.0, 1.0]
}

impl EncoderConfig {
    fn layer_type(&self, index: usize) -> &str {
        self.layer_types
            .as_ref()
            .and_then(|types| types.get(index).map(String::as_str))
            .unwrap_or_else(|| {
                if self.global_attn_every_n_layers != 0
                    && index.is_multiple_of(self.global_attn_every_n_layers)
                {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
    }

    fn rope_base(&self, kind: &str) -> Result<f32> {
        if let Some(params) = self.rope_parameters.as_ref().and_then(|p| p.get(kind)) {
            if params.rope_type != "default" {
                return Err(Error::Inference {
                    message: format!("unsupported RoPE type `{}`", params.rope_type),
                });
            }
            if let Some(theta) = params.rope_theta {
                return Ok(theta);
            }
        }
        Ok(if kind == "full_attention" {
            self.global_rope_theta
        } else {
            self.local_rope_theta
        })
    }
}

#[derive(Debug, Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
struct SpecialTokens {
    cls: i32,
    sep: i32,
    mask: i32,
    pad: i32,
    mask_text: String,
}

#[derive(Debug)]
struct Sequence {
    ids: Vec<i32>,
    markers: Vec<usize>,
}

#[derive(Debug)]
struct Prepared {
    sequence: Sequence,
    kind: QuestionType,
    options: Vec<String>,
    labels: Vec<String>,
    levels: Vec<String>,
}

#[derive(Debug)]
struct Weights {
    values: HashMap<String, Array>,
}

impl Weights {
    fn get(&self, name: &str) -> MlxResult<&Array> {
        self.values
            .get(name)
            .ok_or_else(|| Exception::custom(format!("missing materialized parameter `{name}`")))
    }
}

type MlxResult<T> = std::result::Result<T, Exception>;

/// Run one validated System One Call against a local Checkpoint.
pub fn run(store: &Store, source: &ModelSource, call: &Call) -> Result<Value> {
    let directory = store.checkpoint_dir(source.repo)?;
    checkpoint::verify(store, source)?;
    let agent: AgentConfig = read_json(&directory.join("rl_agent_config.json"))?;
    let encoder: EncoderConfig = read_json(&directory.join("encoder/config.json"))?;
    validate_config(&encoder, &agent)?;
    let tokenizer =
        Tokenizer::from_file(directory.join("tokenizer/tokenizer.json")).map_err(|error| {
            Error::Inference {
                message: format!("loading tokenizer: {error}"),
            }
        })?;
    let special = special_tokens(&directory, &tokenizer)?;
    let prepared = prepare(call, &tokenizer, &special, &agent)?;
    let weights = load_weights(&directory.join("model.safetensors"))?;
    let (logits, actions) = forward(&encoder, &agent, &weights, &prepared, &special)?;
    format_result(call, &agent, prepared, logits, actions)
}

fn validate_config(encoder: &EncoderConfig, agent: &AgentConfig) -> Result<()> {
    let valid_temperatures = agent.temperature.len() == 3
        && agent
            .temperature
            .iter()
            .chain(agent.temperature_by_options.values())
            .all(|value| value.is_finite() && *value > 0.0);
    if !valid_temperatures {
        return Err(Error::Inference {
            message: "calibration temperatures must be three finite positive values".to_string(),
        });
    }
    if encoder.hidden_size == 0
        || encoder.num_attention_heads == 0
        || encoder.hidden_size % encoder.num_attention_heads != 0
        || encoder.num_hidden_layers == 0
        || encoder.local_attention == 0
        || encoder.max_position_embeddings == 0
    {
        return Err(Error::Inference {
            message: "encoder configuration has invalid dimensions".to_string(),
        });
    }
    if !(4 < agent.head_max_len
        && agent.head_max_len < agent.max_len
        && agent.max_len <= encoder.max_position_embeddings)
    {
        return Err(Error::Inference {
            message: "expected 4 < head_max_len < max_len <= max_position_embeddings".to_string(),
        });
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|error| Error::io("read", path, error))?;
    serde_json::from_slice(&bytes).map_err(|error| Error::json(path.display().to_string(), error))
}

fn special_tokens(directory: &Path, tokenizer: &Tokenizer) -> Result<SpecialTokens> {
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

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|error| Error::Inference {
            message: format!("tokenization failed: {error}"),
        })?;
    encoding
        .get_ids()
        .iter()
        .map(|id| {
            i32::try_from(*id).map_err(|_| Error::Inference {
                message: "token id is too large for MLX".to_string(),
            })
        })
        .collect()
}

fn render_value(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn render_options(question: &Question) -> (Vec<String>, Vec<String>, Vec<String>) {
    match question.kind {
        QuestionType::Choice => match question.criteria.as_ref() {
            Some(Criteria::Object(options)) => {
                let labels = options
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect::<Vec<_>>();
                let rendered = options
                    .iter()
                    .map(|(name, value)| {
                        if value.is_null() || value.as_str() == Some("") {
                            name.clone()
                        } else {
                            format!("{name}: {}", render_value(value))
                        }
                    })
                    .collect();
                (rendered, labels, Vec::new())
            }
            Some(Criteria::List(values)) => {
                let labels = values.iter().map(render_value).collect::<Vec<_>>();
                (labels.clone(), labels, Vec::new())
            }
            None => (Vec::new(), Vec::new(), Vec::new()),
        },
        QuestionType::Score => match question.criteria.as_ref() {
            Some(Criteria::List(values)) => {
                let levels = values.iter().map(render_value).collect::<Vec<_>>();
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
            let object = match question.criteria.as_ref() {
                Some(Criteria::Object(values)) => values,
                _ => {
                    return (
                        vec![
                            "false: no, the statement does not hold".to_string(),
                            "true: yes, the statement holds".to_string(),
                        ],
                        Vec::new(),
                        Vec::new(),
                    );
                }
            };
            let description = |name: &str, fallback: &str| {
                object
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|(_, value)| {
                        if value.is_null() || value.as_str() == Some("") {
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
                        description("false", "no, the statement does not hold")
                    ),
                    format!("true: {}", description("true", "yes, the statement holds")),
                ],
                Vec::new(),
                Vec::new(),
            )
        }
    }
}

fn prepare(
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
            let mut budget = config.head_max_len as isize - used as isize;
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
                budget = config.head_max_len as isize
                    - option_ids.iter().map(Vec::len).sum::<usize>() as isize;
            }
            head.truncate(budget.max(8) as usize);
            let mut ids = vec![special.cls];
            ids.extend(head);
            ids.push(special.sep);
            let mut markers = Vec::with_capacity(option_ids.len());
            for option in option_ids {
                markers.push(ids.len());
                ids.extend(option);
            }
            ids.push(special.sep);
            let state = serialize_state(&call.state).replace(&special.mask_text, " ");
            let room = config.max_len.saturating_sub(ids.len() + 1);
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

fn serialize_state(state: &Value) -> String {
    state
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| state.to_string())
}

fn load_weights(path: &Path) -> Result<Weights> {
    let bytes = fs::read(path).map_err(|error| Error::io("read", path, error))?;
    if bytes.len() < 8 {
        return Err(Error::InvalidCheckpoint {
            path: path.to_path_buf(),
            message: "safetensors file has no header".to_string(),
        });
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let header_end = 8usize
        .checked_add(
            usize::try_from(header_len).map_err(|_| Error::InvalidCheckpoint {
                path: path.to_path_buf(),
                message: "safetensors header is too large".to_string(),
            })?,
        )
        .ok_or_else(|| Error::InvalidCheckpoint {
            path: path.to_path_buf(),
            message: "safetensors header is too large".to_string(),
        })?;
    if header_end > bytes.len() {
        return Err(Error::InvalidCheckpoint {
            path: path.to_path_buf(),
            message: "safetensors header is truncated".to_string(),
        });
    }
    let header: BTreeMap<String, Value> =
        serde_json::from_slice(&bytes[8..header_end]).map_err(|error| {
            Error::InvalidCheckpoint {
                path: path.to_path_buf(),
                message: format!("safetensors header is not JSON: {error}"),
            }
        })?;
    let data = &bytes[header_end..];
    let mut values = HashMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let entry: TensorHeader =
            serde_json::from_value(value).map_err(|error| Error::InvalidCheckpoint {
                path: path.to_path_buf(),
                message: format!("parameter `{name}` is invalid: {error}"),
            })?;
        let start =
            usize::try_from(entry.data_offsets[0]).map_err(|_| Error::InvalidCheckpoint {
                path: path.to_path_buf(),
                message: "tensor offset is too large".to_string(),
            })?;
        let end = usize::try_from(entry.data_offsets[1]).map_err(|_| Error::InvalidCheckpoint {
            path: path.to_path_buf(),
            message: "tensor offset is too large".to_string(),
        })?;
        if end > data.len() || start > end {
            return Err(Error::InvalidCheckpoint {
                path: path.to_path_buf(),
                message: format!("parameter `{name}` has invalid data offsets"),
            });
        }
        let shape = entry
            .shape
            .iter()
            .map(|size| {
                i32::try_from(*size).map_err(|_| Error::InvalidCheckpoint {
                    path: path.to_path_buf(),
                    message: format!("parameter `{name}` shape is too large"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let raw = &data[start..end];
        let array = match entry.dtype.as_str() {
            "F16" => Array::from_slice(
                &raw.chunks_exact(2)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect::<Vec<_>>(),
                &shape,
            ),
            "F32" => Array::from_slice(
                &raw.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<_>>(),
                &shape,
            ),
            dtype => {
                return Err(Error::InvalidCheckpoint {
                    path: path.to_path_buf(),
                    message: format!("parameter `{name}` has unsupported runtime dtype {dtype}"),
                });
            }
        };
        values.insert(name, array);
    }
    Ok(Weights { values })
}

fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> MlxResult<Array> {
    let y = x.matmul(weight.t())?;
    match bias {
        Some(bias) => y.add(bias),
        None => Ok(y),
    }
}

fn layer_norm(x: &Array, weight: &Array, bias: Option<&Array>, eps: f32) -> MlxResult<Array> {
    mlx_rs::fast::layer_norm(x, Some(weight), bias, eps)
}

fn attention(
    x: &Array,
    qkv_weight: &Array,
    qkv_bias: Option<&Array>,
    out_weight: &Array,
    out_bias: Option<&Array>,
    heads: usize,
    rope_base: Option<f32>,
    mask: &Array,
) -> MlxResult<Array> {
    let shape = x.shape();
    let batch = shape[0];
    let length = shape[1];
    let hidden = shape[2];
    let head_dim = hidden / i32::try_from(heads).unwrap();
    let qkv = linear(x, qkv_weight, qkv_bias)?.reshape(&[
        batch,
        length,
        3,
        i32::try_from(heads).unwrap(),
        head_dim,
    ])?;
    let axis_indices = [0i32];
    let take = |index: i32| -> MlxResult<Array> {
        qkv.take_axis(Array::from_slice(&[index], &axis_indices), 2)?
            .squeeze_axes(&[2])?
            .transpose_axes(&[0, 2, 1, 3])
    };
    let mut q = take(0)?;
    let mut k = take(1)?;
    let v = take(2)?;
    if let Some(base) = rope_base {
        q = mlx_rs::fast::rope(&q, head_dim, false, Some(base), 1.0, 0, None::<&Array>)?;
        k = mlx_rs::fast::rope(&k, head_dim, false, Some(base), 1.0, 0, None::<&Array>)?;
    }
    let attended = mlx_rs::fast::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        (head_dim as f32).powf(-0.5),
        mask,
        None::<&Array>,
    )?;
    let attended = attended
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, length, hidden])?;
    linear(&attended, out_weight, out_bias)
}

fn encoder_mask(attention_mask: &Array, window: usize) -> MlxResult<(Array, Array)> {
    let shape = attention_mask.shape();
    if shape.len() != 2 {
        return Err(Exception::from("attention mask must be rank two"));
    }
    let batch = shape[0];
    let length = shape[1];
    let full = attention_mask.reshape(&[batch, 1, 1, length])?;
    let mut local = vec![false; usize::try_from(length * length).unwrap()];
    let radius = i32::try_from(window / 2).unwrap();
    for row in 0..length {
        for col in 0..length {
            local[usize::try_from(row * length + col).unwrap()] = (row - col).abs() <= radius;
        }
    }
    let local = Array::from_slice(&local, &[1, 1, length, length]);
    let valid_queries = attention_mask
        .logical_not()?
        .reshape(&[batch, 1, length, 1])?;
    let sliding = local.logical_or(&valid_queries)?.logical_and(&full)?;
    Ok((full, sliding))
}

fn encoder(
    config: &EncoderConfig,
    weights: &Weights,
    input_ids: &Array,
    attention_mask: &Array,
) -> MlxResult<Array> {
    let embeddings = weights.get("encoder.embeddings.tok_embeddings.weight")?;
    let mut x = embeddings.take_axis(input_ids, 0)?;
    x = layer_norm(
        &x,
        weights.get("encoder.embeddings.norm.weight")?,
        None,
        config.norm_eps,
    )?;
    let (full_mask, sliding_mask) = encoder_mask(attention_mask, config.local_attention)?;
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("encoder.layers.{layer}");
        let normalized = if layer == 0 {
            x.clone()
        } else {
            layer_norm(
                &x,
                weights.get(&format!("{prefix}.attn_norm.weight"))?,
                None,
                config.norm_eps,
            )?
        };
        let kind = config.layer_type(layer);
        let mask = if kind == "full_attention" {
            &full_mask
        } else {
            &sliding_mask
        };
        let attended = attention(
            &normalized,
            weights.get(&format!("{prefix}.attn.Wqkv.weight"))?,
            None,
            weights.get(&format!("{prefix}.attn.Wo.weight"))?,
            None,
            config.num_attention_heads,
            Some(config.rope_base(kind)?),
            mask,
        )?;
        x = x.add(&attended)?;
        let normalized = layer_norm(
            &x,
            weights.get(&format!("{prefix}.mlp_norm.weight"))?,
            None,
            config.norm_eps,
        )?;
        let gated = linear(
            &normalized,
            weights.get(&format!("{prefix}.mlp.Wi.weight"))?,
            None,
        )?;
        let halves = gated.split_equal(2, Some(-1))?;
        let activated = mlx_rs::nn::gelu(&halves[0])?.multiply(&halves[1])?;
        let projected = linear(
            &activated,
            weights.get(&format!("{prefix}.mlp.Wo.weight"))?,
            None,
        )?;
        x = x.add(&projected)?;
    }
    layer_norm(
        &x,
        weights.get("encoder.final_norm.weight")?,
        None,
        config.norm_eps,
    )
}

fn head_layer(
    x: &Array,
    prefix: &str,
    weights: &Weights,
    hidden: i32,
    mask: &Array,
) -> MlxResult<Array> {
    let normalized = layer_norm(
        x,
        weights.get(&format!("{prefix}.norm1.weight"))?,
        Some(weights.get(&format!("{prefix}.norm1.bias"))?),
        LN_EPS,
    )?;
    let attended = attention(
        &normalized,
        weights.get(&format!("{prefix}.self_attn.in_proj_weight"))?,
        Some(weights.get(&format!("{prefix}.self_attn.in_proj_bias"))?),
        weights.get(&format!("{prefix}.self_attn.out_proj.weight"))?,
        Some(weights.get(&format!("{prefix}.self_attn.out_proj.bias"))?),
        usize::try_from((hidden / 64).max(1)).unwrap(),
        None,
        mask,
    )?;
    let x = x.add(&attended)?;
    let normalized = layer_norm(
        &x,
        weights.get(&format!("{prefix}.norm2.weight"))?,
        Some(weights.get(&format!("{prefix}.norm2.bias"))?),
        LN_EPS,
    )?;
    let feed = linear(
        &normalized,
        weights.get(&format!("{prefix}.linear1.weight"))?,
        Some(weights.get(&format!("{prefix}.linear1.bias"))?),
    )?;
    let feed = mlx_rs::nn::relu(&feed)?;
    let feed = linear(
        &feed,
        weights.get(&format!("{prefix}.linear2.weight"))?,
        Some(weights.get(&format!("{prefix}.linear2.bias"))?),
    )?;
    x.add(&feed)
}

fn forward(
    config: &EncoderConfig,
    agent: &AgentConfig,
    weights: &Weights,
    prepared: &[Prepared],
    special: &SpecialTokens,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let count = i32::try_from(prepared.len()).map_err(|_| Error::Inference {
        message: "too many Questions".to_string(),
    })?;
    let length = i32::try_from(
        prepared
            .iter()
            .map(|item| item.sequence.ids.len())
            .max()
            .unwrap_or(0),
    )
    .map_err(|_| Error::Inference {
        message: "input is too long".to_string(),
    })?;
    let markers = i32::try_from(
        prepared
            .iter()
            .map(|item| item.sequence.markers.len())
            .max()
            .unwrap_or(0)
            .max(2),
    )
    .map_err(|_| Error::Inference {
        message: "too many Options".to_string(),
    })?;
    let mut ids = vec![special.pad; usize::try_from(count * length).unwrap()];
    let mut mask = vec![false; usize::try_from(count * length).unwrap()];
    let mut marker_pos = vec![0i32; usize::try_from(count * markers).unwrap()];
    let mut marker_mask = vec![false; usize::try_from(count * markers).unwrap()];
    let mut qtypes = Vec::with_capacity(prepared.len());
    for (row, item) in prepared.iter().enumerate() {
        let base = row * usize::try_from(length).unwrap();
        ids[base..base + item.sequence.ids.len()].copy_from_slice(&item.sequence.ids);
        mask[base..base + item.sequence.ids.len()].fill(true);
        for (index, marker) in item.sequence.markers.iter().enumerate() {
            marker_pos[row * usize::try_from(markers).unwrap() + index] =
                i32::try_from(*marker).unwrap();
            marker_mask[row * usize::try_from(markers).unwrap() + index] = true;
        }
        qtypes.push(item.kind.index());
    }
    let result = (|| -> MlxResult<(Array, Array)> {
        let input_ids = Array::from_slice(&ids, &[count, length]);
        let attention_mask = Array::from_slice(&mask, &[count, length]);
        let marker_pos = Array::from_slice(&marker_pos, &[count, markers]);
        let marker_mask = Array::from_slice(&marker_mask, &[count, markers]);
        let qtype = Array::from_slice(&qtypes, &[count]);
        let mut hidden = encoder(config, weights, &input_ids, &attention_mask)?;
        let type_emb = weights
            .get("type_emb.weight")?
            .take_axis(&qtype, 0)?
            .expand_dims(1)?;
        hidden = hidden.add(&type_emb)?;
        let head_mask = attention_mask.reshape(&[count, 1, 1, length])?;
        for layer in 0..agent.head_layers {
            hidden = head_layer(
                &hidden,
                &format!("head.layers.{layer}"),
                weights,
                config.hidden_size as i32,
                &head_mask,
            )?;
        }
        let marker_indices = marker_pos.expand_dims(2)?.broadcast_to(&[
            count,
            markers,
            config.hidden_size as i32,
        ])?;
        let marker_hidden = hidden.take_along_axis(&marker_indices, 1)?;
        let scorer = layer_norm(
            &marker_hidden,
            weights.get("scorer.0.weight")?,
            Some(weights.get("scorer.0.bias")?),
            LN_EPS,
        )?;
        let scorer = linear(
            &scorer,
            weights.get("scorer.1.weight")?,
            Some(weights.get("scorer.1.bias")?),
        )?;
        let scorer = mlx_rs::nn::gelu(&scorer)?;
        let scorer = linear(
            &scorer,
            weights.get("scorer.3.weight")?,
            Some(weights.get("scorer.3.bias")?),
        )?
        .squeeze_axes(&[2])?;
        let logits = ops::select(&marker_mask, &scorer, &Array::from_f32(MASK_FILL))?;
        let probabilities = ops::softmax_axis(&logits, -1, false)?;
        let sorted = ops::sort_axis(&probabilities, -1)?;
        let top = sorted.take_axis(Array::from_slice(&[markers - 2, markers - 1], &[2]), -1)?;
        let top_one = top
            .take_axis(Array::from_slice(&[1], &[1]), -1)?
            .squeeze_axes(&[1])?;
        let top_two = top
            .take_axis(Array::from_slice(&[0], &[1]), -1)?
            .squeeze_axes(&[1])?;
        let p_log_p = probabilities
            .multiply(&ops::maximum(&probabilities, &Array::from_f32(1e-9))?.log()?)?;
        let entropy = p_log_p.negative()?.sum_axis(-1, false)?
            / ops::maximum(
                &marker_mask
                    .count_nonzero(mlx_rs::ops::CountNonzeroOptions {
                        axes: mlx_rs::Axes::Axis(-1),
                        keep_dims: false,
                    })?
                    .as_type::<f32>()?,
                &Array::from_f32(2.0),
            )?
            .log()?;
        let features = ops::stack(
            &[
                &top_one,
                &top_one.subtract(&top_two)?,
                &entropy,
                &(marker_mask
                    .count_nonzero(mlx_rs::ops::CountNonzeroOptions {
                        axes: mlx_rs::Axes::Axis(-1),
                        keep_dims: false,
                    })?
                    .as_type::<f32>()?
                    / 255.0),
            ],
            -1,
        )?;
        let pooled = hidden
            .take_axis(Array::from_slice(&[0], &[1]), 1)?
            .squeeze_axes(&[1])?;
        let pooled = ops::concatenate(&[&pooled, &features], 1)?;
        let action = linear(
            &pooled,
            weights.get("act_head.0.weight")?,
            Some(weights.get("act_head.0.bias")?),
        )?;
        let action = mlx_rs::nn::gelu(&action)?;
        let action = linear(
            &action,
            weights.get("act_head.2.weight")?,
            Some(weights.get("act_head.2.bias")?),
        )?;
        Ok((logits, ops::softmax_axis(&action, -1, false)?))
    })()
    .map_err(|error| Error::Inference {
        message: error.to_string(),
    })?;
    mlx_rs::transforms::eval([&result.0, &result.1]).map_err(|error| Error::Inference {
        message: error.to_string(),
    })?;
    let logits_slice = result.0.as_slice::<f32>();
    let actions_slice = result.1.as_slice::<f32>();
    if logits_slice
        .iter()
        .chain(actions_slice)
        .any(|value| !value.is_finite())
    {
        return Err(Error::Inference {
            message: "model produced non-finite output".to_string(),
        });
    }
    let logits = logits_slice
        .chunks(markers as usize)
        .map(|row| row.to_vec())
        .collect();
    let actions = actions_slice
        .chunks(agent.act_costs.len() + 1)
        .map(|row| row.to_vec())
        .collect();
    Ok((logits, actions))
}

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
    } else if fraction > 0.5 || (lower as i64) % 2 != 0 {
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

fn format_result(
    call: &Call,
    agent: &AgentConfig,
    prepared: Vec<Prepared>,
    logits: Vec<Vec<f32>>,
    actions: Vec<Vec<f32>>,
) -> Result<Value> {
    let mut answers = Map::new();
    for (row, (id, _)) in call.questions.iter().enumerate() {
        let item = &prepared[row];
        let probabilities = stable_softmax(
            &logits[row][..item.options.len()]
                .iter()
                .map(|value| *value / temperature(agent, item.kind, item.options.len()))
                .collect::<Vec<_>>(),
        );
        let action = actions[row].first().copied().unwrap_or(0.0);
        let mut answer = Map::new();
        answer.insert("type".to_string(), Value::String(item.kind.to_string()));
        answer.insert(
            "confidence".to_string(),
            round_even(confidence(&probabilities)),
        );
        answer.insert(
            "action".to_string(),
            serde_json::json!({ "act_probability": round_even(action) }),
        );
        match item.kind {
            QuestionType::Choice => {
                let best = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
                    .map(|(index, _)| index)
                    .unwrap_or(0);
                answer.insert(
                    "choice".to_string(),
                    Value::String(item.labels[best].clone()),
                );
                let mut values = Map::new();
                for (label, probability) in item.labels.iter().zip(&probabilities) {
                    values.insert(label.clone(), round_even(*probability));
                }
                answer.insert("probabilities".to_string(), Value::Object(values));
            }
            QuestionType::Score => {
                let score = probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, probability)| index as f32 * probability)
                    .sum::<f32>();
                answer.insert("score".to_string(), round_even(score));
                let mut legend = Map::new();
                let mut values = Map::new();
                for (index, (level, probability)) in
                    item.levels.iter().zip(&probabilities).enumerate()
                {
                    legend.insert(index.to_string(), Value::String(level.clone()));
                    values.insert(index.to_string(), round_even(*probability));
                }
                answer.insert("legend".to_string(), Value::Object(legend));
                answer.insert("probabilities".to_string(), Value::Object(values));
            }
            QuestionType::Noul => {
                answer.insert("noul".to_string(), round_even(probabilities[1]));
                answer.insert(
                    "confidence".to_string(),
                    round_even(probabilities[1].max(1.0 - probabilities[1])),
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
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": { "input_tokens": input_tokens, "output_tokens": 0 }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(round_even(0.12345), serde_json::json!(0.1234));
        assert_eq!(round_even(0.12355), serde_json::json!(0.1236));
    }
}
