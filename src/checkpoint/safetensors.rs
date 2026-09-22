//! A Checkpoint's weights header against the Parameter Manifest (ADR-0005): every expected
//! parameter present with the expected dtype and shape, no parameter beyond them, every data range
//! inside the file, and no tensor read.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

use super::invalid;
use super::manifest::{Dtype, Manifest};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderEntry {
    /// The dtype as the header spells it; parsed into a [`Dtype`] before the entry is used.
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// A header entry once its dtype is parsed: what the weights file states a parameter is.
#[derive(Debug)]
struct ParsedEntry {
    dtype: Dtype,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Verify the header of `path`, a Checkpoint's weights, against `manifest`.
pub(super) fn verify(path: &Path, manifest: &Manifest) -> Result<()> {
    let length = std::fs::metadata(path)
        .map_err(|error| Error::io("inspect", path, error))?
        .len();
    let mut file = File::open(path).map_err(|error| Error::io("open", path, error))?;
    let mut size = [0u8; 8];
    file.read_exact(&mut size)
        .map_err(|error| invalid(path, format!("cannot read header length: {error}")))?;
    let header_len = u64::from_le_bytes(size);
    // The file may have been replaced since it was measured, so a length below the prefix leaves
    // no header and no data rather than subtracting past zero.
    let available = length.saturating_sub(8);
    if header_len > available {
        return Err(invalid(
            path,
            format!("header is {header_len} bytes, but only {available} are available"),
        ));
    }
    let data_len = available - header_len;
    let header_len_usize =
        usize::try_from(header_len).map_err(|_| invalid(path, "header is too large"))?;
    let mut bytes = vec![0u8; header_len_usize];
    file.read_exact(&mut bytes)
        .map_err(|error| invalid(path, format!("cannot read header: {error}")))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| invalid(path, format!("header is not JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid(path, "header is not an object"))?;
    let mut entries = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            if !value.is_object() {
                return Err(invalid(path, "__metadata__ is not an object"));
            }
            continue;
        }
        let entry: HeaderEntry = HeaderEntry::deserialize(value)
            .map_err(|error| invalid(path, format!("parameter `{name}` is invalid: {error}")))?;
        if entry.data_offsets[0] > entry.data_offsets[1] || entry.data_offsets[1] > data_len {
            return Err(invalid(
                path,
                format!("parameter `{name}` has invalid data offsets"),
            ));
        }
        let dtype = entry.dtype.parse::<Dtype>().map_err(|_| {
            invalid(
                path,
                format!(
                    "parameter `{name}` uses an unsupported dtype `{}`",
                    entry.dtype
                ),
            )
        })?;
        let expected_bytes = entry
            .shape
            .iter()
            .try_fold(1u64, |elements, size| elements.checked_mul(*size))
            .and_then(|elements| elements.checked_mul(dtype.bytes()))
            .ok_or_else(|| invalid(path, format!("parameter `{name}` shape is too large")))?;
        if entry.data_offsets[1] - entry.data_offsets[0] != expected_bytes {
            return Err(invalid(
                path,
                format!("parameter `{name}` data length does not match its dtype and shape"),
            ));
        }
        // The name is stored once; the data ranges borrow it back out of the map to be sorted.
        entries.insert(
            name.clone(),
            ParsedEntry {
                dtype,
                shape: entry.shape,
                data_offsets: entry.data_offsets,
            },
        );
    }
    let mut ranges: Vec<(u64, u64, &str)> = entries
        .iter()
        .map(|(name, entry)| (entry.data_offsets[0], entry.data_offsets[1], name.as_str()))
        .collect();
    ranges.sort_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(invalid(
                path,
                format!("parameters `{}` and `{}` overlap", pair[0].2, pair[1].2),
            ));
        }
    }

    for (name, expected) in &manifest.parameters {
        let Some(actual) = entries.get(name) else {
            return Err(Error::MissingParameter { name: name.clone() });
        };
        if actual.dtype != expected.dtype || actual.shape != expected.shape {
            return Err(Error::ParameterMismatch {
                name: name.clone(),
                expected_dtype: expected.dtype.to_string(),
                actual_dtype: actual.dtype.to_string(),
                expected_shape: expected.shape.clone(),
                actual_shape: actual.shape.clone(),
            });
        }
    }
    if let Some(name) = entries
        .keys()
        .find(|name| !manifest.parameters.contains_key(*name))
    {
        return Err(Error::UnexpectedParameter { name: name.clone() });
    }
    Ok(())
}
