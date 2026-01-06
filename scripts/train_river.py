#!/usr/bin/env python3
"""Train Deep CFR on Texas Hold'em River.

This script trains a Deep PDCFR+ agent on the River subgame of Texas Hold'em.
Since NashConv is computationally infeasible for 52-card games, we use
head-to-head evaluation against baseline bots.

Validation Strategy:
- Every 1,000 iterations, play 1,000 hands vs baseline bots
- Measure win rate in milli-big-blinds per hand (mbb/h)
- Baselines: RandomBot, CallingStation, HonestBot

Expected Performance:
- vs RandomBot: +2000-3000 mbb/h (should dominate random)
- vs CallingStation: +1000-2000 mbb/h (exploit passivity)
- vs HonestBot: +500-1500 mbb/h (exploit honesty, no bluffs)

Hyperparameters:
- Buffer: 100,000 samples (Hold'em needs more data)
- Batch: 1,024 (larger batches for stability)
- Iterations: 10,000 (endgame solving requires fewer iterations)
- Hidden: 128 units (larger network for 31-dim input)
- Layers: 3 (standard depth)
"""

import sys
from pathlib import Path
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch

from aion26.games.river_holdem import new_river_holdem_game
from aion26.deep_cfr.networks import HoldemEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.learner.discounting import PDCFRScheduler, LinearScheduler
from aion26.baselines import RandomBot, CallingStation, HonestBot
from aion26.metrics.evaluator import HeadToHeadEvaluator

# ============================================================================
# Configuration
# ============================================================================

ITERATIONS = 10000
BUFFER_CAPACITY = 100000  # Large buffer for Hold'em
BATCH_SIZE = 1024
HIDDEN_SIZE = 128
NUM_LAYERS = 3
LEARNING_RATE = 0.001

EVAL_EVERY = 1000
EVAL_HANDS = 1000

# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    print("="*80)
    print("TEXAS HOLD'EM RIVER - DEEP PDCFR+ TRAINING")
    print("="*80)
    print()

    print("Configuration:")
    print(f"  Game: Texas Hold'em River (52 cards)")
    print(f"  Iterations: {ITERATIONS:,}")
    print(f"  Buffer: {BUFFER_CAPACITY:,} samples")
    print(f"  Batch: {BATCH_SIZE}")
    print(f"  Network: {HIDDEN_SIZE}×{NUM_LAYERS}")
    print(f"  Eval: Every {EVAL_EVERY} iters, {EVAL_HANDS} hands")
    print()

    # Create game and encoder
    print("Initializing...")
    game = new_river_holdem_game()
    encoder = HoldemEncoder()

    print(f"  Game: TexasHoldemRiver")
    print(f"  Encoder: HoldemEncoder ({encoder.input_size} dims)")
    print(f"  Actions: 4 (Fold, Check/Call, Bet Pot, All-In)")
    print()

    # Create trainer
    trainer = DeepCFRTrainer(
        initial_state=game,
        encoder=encoder,
        input_size=encoder.input_size,
        output_size=4,  # 4 actions
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        regret_scheduler=PDCFRScheduler(alpha=2.0, beta=0.5),
        strategy_scheduler=LinearScheduler(),
        use_vr=True,  # Variance reduction
    )

    print(f"Trainer initialized:")
    print(f"  Regret scheduler: PDCFR (α=2.0, β=0.5)")
    print(f"  Strategy scheduler: Linear")
    print(f"  Variance reduction: Enabled")
    print()

    # Create baseline bots
    random_bot = RandomBot()
    calling_bot = CallingStation()
    honest_bot = HonestBot()

    print("Baseline bots:")
    print(f"  RandomBot: Uniform random policy")
    print(f"  CallingStation: Always checks/calls")
    print(f"  HonestBot: Strength-based (>0.8: bet, >0.5: call, else: fold)")
    print()

    # Create evaluator
    evaluator = HeadToHeadEvaluator(big_blind=2.0)

    print("="*80)
    print("TRAINING")
    print("="*80)
    print()

    start_time = time.time()

    for iteration in range(1, ITERATIONS + 1):
        # Run one iteration
        metrics = trainer.run_iteration()

        # Print progress
        if iteration % 100 == 0 or iteration == 1:
            elapsed = time.time() - start_time
            iter_per_sec = iteration / elapsed if elapsed > 0 else 0

            print(
                f"Iter {iteration:5d} | "
                f"Loss: {metrics.get('loss', 0):.4f} | "
                f"Buffer: {len(trainer.buffer):6d}/{BUFFER_CAPACITY} "
                f"({trainer.buffer.fill_percentage:5.1f}%) | "
                f"{iter_per_sec:.1f} it/s"
            )

        # Evaluate against baseline bots
        if iteration % EVAL_EVERY == 0:
            print()
            print(f"{'='*80}")
            print(f"EVALUATION AT ITERATION {iteration}")
            print(f"{'='*80}")
            print()

            # Get current strategy
            strategy = trainer.get_all_average_strategies()

            if len(strategy) == 0:
                print("⚠️  No strategy learned yet, skipping evaluation")
                print()
                continue

            print(f"Strategy size: {len(strategy)} information states")
            print()

            # Evaluate vs each bot
            opponents = {
                "RandomBot": random_bot,
                "CallingStation": calling_bot,
                "HonestBot": honest_bot,
            }

            for name, bot in opponents.items():
                print(f"Playing {EVAL_HANDS} hands vs {name}...")

                result = evaluator.evaluate(
                    initial_state=game,
                    strategy=strategy,
                    opponent=bot,
                    num_hands=EVAL_HANDS
                )

                print(f"  Win rate: {result.avg_mbb_per_hand:+7.0f} mbb/h "
                      f"± {result.confidence_95:.0f} (95% CI)")
                print(f"  Total: {result.agent_winnings:+.1f} BB "
                      f"over {result.num_hands} hands")
                print()

            print(f"{'='*80}")
            print()

    # Final summary
    elapsed_total = time.time() - start_time
    print()
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print()
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"Final buffer: {len(trainer.buffer)}/{BUFFER_CAPACITY} "
          f"({trainer.buffer.fill_percentage:.1f}%)")
    print()

    # Final evaluation
    print("Final Evaluation:")
    print()

    strategy = trainer.get_all_average_strategies()
    print(f"Strategy size: {len(strategy)} information states")
    print()

    opponents = {
        "RandomBot": random_bot,
        "CallingStation": calling_bot,
        "HonestBot": honest_bot,
    }

    for name, bot in opponents.items():
        print(f"{name}:")
        result = evaluator.evaluate(
            initial_state=game,
            strategy=strategy,
            opponent=bot,
            num_hands=EVAL_HANDS
        )
        print(f"  {result.avg_mbb_per_hand:+7.0f} mbb/h ± {result.confidence_95:.0f}")

    print()
    print("Training complete! 🚀")
    print()


if __name__ == "__main__":
    main()
