use pyo3::prelude::*;

mod evaluator;
mod io;
mod io_full;
mod river;
mod game_full;
mod tabular;
mod trainer;
mod parallel_trainer;
mod parallel_trainer_full;
mod flop_abstraction;
mod solver_db;  // Future: Pre-computed strategy database

use evaluator::{evaluate_7_cards as eval_rust, get_hand_category, card_to_string};
use river::RustRiverHoldem;
use game_full::RustFullHoldem;
use tabular::TabularCFR;
use trainer::{RustTrainer, StepResult};
use parallel_trainer::{ParallelTrainer, StepResultPar};
use parallel_trainer_full::{ParallelTrainerFull, StepResultFull};
use flop_abstraction::{PyFlopBucketing, py_generate_canonical_flops, py_num_canonical_flops, py_create_texture_buckets, py_get_flop_bucket};

/// Evaluate a 7-card poker hand
///
/// Args:
///     cards: List of 7 card indices (0-51)
///
/// Returns:
///     Hand rank (1-7462, lower is better)
#[pyfunction]
fn evaluate_7_cards(cards: Vec<u8>) -> PyResult<u32> {
    if cards.len() != 7 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Expected 7 cards, got {}", cards.len())
        ));
    }

    let card_array: [u8; 7] = cards.try_into()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid cards"))?;

    Ok(eval_rust(&card_array))
}

/// Get hand category from rank
///
/// Args:
///     rank: Hand rank (1-7462)
///
/// Returns:
///     Category (0-9): 0=High Card, 1=Pair, ..., 9=Royal Flush
#[pyfunction]
fn get_category(rank: u32) -> u8 {
    get_hand_category(rank)
}

/// Convert card index to string
///
/// Args:
///     card: Card index (0-51)
///
/// Returns:
///     Card string (e.g., "A♠")
#[pyfunction]
fn card_str(card: u8) -> String {
    card_to_string(card)
}

/// Aion-26 Rust Extension
///
/// Provides fast implementations of:
/// - Hand evaluation (50-100x faster than treys)
/// - River Hold'em game state (50x faster than Python)
/// - Tabular CFR solver (exact Nash equilibrium)
/// - Deep CFR Trainer (20-50x faster via Rust driver architecture)
#[pymodule]
fn aion26_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_7_cards, m)?)?;
    m.add_function(wrap_pyfunction!(get_category, m)?)?;
    m.add_function(wrap_pyfunction!(card_str, m)?)?;
    m.add_class::<RustRiverHoldem>()?;
    m.add_class::<RustFullHoldem>()?;
    m.add_class::<TabularCFR>()?;
    m.add_class::<RustTrainer>()?;
    m.add_class::<StepResult>()?;
    m.add_class::<ParallelTrainer>()?;
    m.add_class::<StepResultPar>()?;
    m.add_class::<ParallelTrainerFull>()?;
    m.add_class::<StepResultFull>()?;
    // Flop abstraction
    m.add_class::<PyFlopBucketing>()?;
    m.add_function(wrap_pyfunction!(py_generate_canonical_flops, m)?)?;
    m.add_function(wrap_pyfunction!(py_num_canonical_flops, m)?)?;
    m.add_function(wrap_pyfunction!(py_create_texture_buckets, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_flop_bucket, m)?)?;
    Ok(())
}
