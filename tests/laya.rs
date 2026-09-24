//! The local Backend as a library: one stored Checkpoint judged in-process, through the interface
//! the Model Router judges a local Call with.

mod support;

use serde_json::Value;
use support::TempDir;
use support::fixture;

use s1gate::call::Call;
use s1gate::laya::Loaded;
use s1gate::model_source;
use s1gate::store::Store;

/// A System One Call with one Question of each Question Type, whose Criteria the caller sets.
const CALL: &str = r#"{
    "model": "convaiinnovations/laya",
    "state": "Ticket #4821: the customer was charged twice for order A-5512.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payments, invoices, refunds.", "account": "Sign-in and profile changes."}
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgently should this be answered?",
            "criteria": ["low", "normal", "high"]
        },
        "risk": {
            "type": "noul",
            "instructions": "Is the customer at risk of churning?"
        }
    }
}"#;

#[test]
fn a_stored_checkpoint_judges_every_question_type() {
    let loaded = loaded("laya-judged");
    let call = Call::from_bytes(CALL.as_bytes()).expect("a valid System One Call");

    let response = loaded.run(&call).expect("the Call is judged");

    assert_eq!(
        response["model"],
        fixture::NAME,
        "the Model Identifier is the Checkpoint's"
    );
    let answers = response["answers"].as_object().expect("the Answers");
    assert_eq!(
        answers
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>(),
        ["route", "urgency", "risk"].into_iter().collect(),
        "one Answer per Question, under the id the caller chose"
    );

    let route = &answers["route"];
    assert_eq!(route["type"], "choice");
    assert!(
        matches!(route["choice"].as_str(), Some("billing" | "account")),
        "the chosen Option is one the Question defines: {route}"
    );
    let probabilities = route["probabilities"]
        .as_object()
        .expect("a probability per Option");
    assert_eq!(probabilities.len(), 2);
    assert_eq!(
        probabilities.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["billing", "account"],
        "the distribution is over the caller's Option names"
    );
    assert!(
        (total(probabilities) - 1.0).abs() < 0.001,
        "the distribution sums to one: {route}"
    );
    assert!(route["confidence"].is_number(), "{route}");

    let urgency = &answers["urgency"];
    assert_eq!(urgency["type"], "score");
    let score = urgency["score"].as_f64().expect("a score");
    assert!(
        (0.0..=2.0).contains(&score),
        "the score is an index over the caller's Levels: {urgency}"
    );
    assert_eq!(urgency["legend"]["0"], "low");
    assert_eq!(urgency["legend"]["2"], "high");
    assert!(
        (total(
            urgency["probabilities"]
                .as_object()
                .expect("a distribution")
        ) - 1.0)
            .abs()
            < 0.001,
        "the distribution sums to one: {urgency}"
    );

    let risk = &answers["risk"];
    assert_eq!(risk["type"], "noul");
    let probability = risk["noul"].as_f64().expect("the probability of true");
    assert!((0.0..=1.0).contains(&probability), "{risk}");
    assert!(
        risk.get("confidence").is_none(),
        "a noul Answer reports only the probability of true (ADR-0012): {risk}"
    );

    assert_eq!(response["usage"]["output_tokens"], 0);
    assert!(
        response["usage"]["input_tokens"]
            .as_u64()
            .is_some_and(|tokens| tokens > 0),
        "the Call reports the tokens it read: {}",
        response["usage"]
    );
}

#[test]
fn one_loaded_backend_judges_the_same_call_the_same_way_twice() {
    let loaded = loaded("laya-repeated");
    let call = Call::from_bytes(CALL.as_bytes()).expect("a valid System One Call");

    assert_eq!(
        loaded.run(&call).expect("the Call is judged"),
        loaded.run(&call).expect("the Call is judged again"),
        "a Backend loaded once judges a repeated Call the same way"
    );
}

/// The store's Checkpoint of the curated Model Source, loaded for judging.
fn loaded(case: &str) -> Loaded {
    let root = TempDir::new(case).path().join("s1gate/models");
    fixture::write_inferable(&root);
    let checkpoint = Store::at(&root)
        .checkpoint(&model_source::LAYA)
        .expect("the Model Store is readable")
        .expect("the fixture Checkpoint is stored");
    Loaded::load(&checkpoint).expect("the Checkpoint loads")
}

/// The sum of one Answer's probabilities.
fn total(probabilities: &serde_json::Map<String, Value>) -> f64 {
    probabilities
        .values()
        .map(|probability| probability.as_f64().expect("a probability"))
        .sum()
}
