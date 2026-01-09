#!/usr/bin/env python3
"""Train Deep CFR on Texas Hold'em River - TURBO MODE with Uniform Strategy Weighting."""

import sys
from pathlib import Path
import time
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from aion26.games.rust_wrapper import new_rust_river_game
from aion26.deep_cfr.networks import HoldemEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.learner.discounting import PDCFRScheduler, UniformScheduler
from aion26.baselines import RandomBot
from aion26.metrics.evaluator import HeadToHeadEvaluator

# TURBO MODE: Accumulate N traversals before each training step
TRAVERSALS_PER_STEP = 50

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100000, help='Training steps (×50 = total traversals)')
parser.add_argument('--buffer', type=int, default=4000000, help='Reservoir buffer capacity (4M for stability)')
parser.add_argument('--batch', type=int, default=4096)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--use-rust', action='store_true')
parser.add_argument('--save-every', type=int, default=10000, help='Save every N training steps')
parser.add_argument('--fixed-board', action='store_true')
args = parser.parse_args()

def main():
    total_traversals = args.iterations * TRAVERSALS_PER_STEP

    print("="*80, flush=True)
    print("TEXAS HOLD'EM RIVER - TURBO MODE (UNIFORM WEIGHTING FIX)", flush=True)
    print("="*80, flush=True)
    print(f"Training Steps: {args.iterations:,} × {TRAVERSALS_PER_STEP} = {total_traversals:,} traversals", flush=True)
    print(f"Buffer: {args.buffer:,} (TENSOR, O(1)) | Batch: {args.batch} | LR: {args.lr}", flush=True)
    print(f"Save: every {args.save_every:,} steps ({args.save_every * TRAVERSALS_PER_STEP:,} traversals)", flush=True)
    print(flush=True)

    if args.fixed_board:
        game = new_rust_river_game(fixed_board=[12, 11, 10, 9, 13])
    else:
        game = new_rust_river_game()

    encoder = HoldemEncoder()
    trainer = DeepCFRTrainer(
        initial_state=game,
        encoder=encoder,
        input_size=encoder.input_size,
        output_size=4,
        hidden_size=512,
        num_hidden_layers=3,
        buffer_capacity=args.buffer,
        batch_size=args.batch,
        learning_rate=args.lr,
        regret_scheduler=PDCFRScheduler(alpha=2.0, beta=0.5),
        strategy_scheduler=UniformScheduler(),  # FIXED: Uniform weighting for stability
        device="cpu"
    )

    random_bot = RandomBot()
    evaluator = HeadToHeadEvaluator(big_blind=2.0)
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    print("TRAINING (TURBO MODE)", flush=True)
    print("="*80, flush=True)
    start_time = time.time()
    traversal_count = 0

    for step in range(1, args.iterations + 1):
        # === 1. BURST GENERATION (Rust/CPU Speed) ===
        # Accumulate N traversals without GPU overhead
        trainer.collect_experience(num_traversals=TRAVERSALS_PER_STEP)
        traversal_count += TRAVERSALS_PER_STEP

        # === 2. SINGLE TRAINING UPDATE (GPU) ===
        metrics = trainer.train_step()

        # Logging
        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            trav_per_sec = traversal_count / elapsed if elapsed > 0 else 0
            print(f"Step {step:7d} ({traversal_count:,} trav) | Loss: {metrics.get('loss', 0):.4f} | "
                  f"Buffer: {len(trainer.buffer):7d}/{args.buffer} "
                  f"({trainer.buffer.fill_percentage:5.1f}%) | {trav_per_sec:.0f} trav/s", flush=True)

        if step % args.save_every == 0:
            print(f"\nCHECKPOINT {traversal_count:,} traversals (Step {step:,})", flush=True)
            strategy = trainer.get_all_average_strategies()
            print(f"Strategy: {len(strategy):,} states", flush=True)

            cp_path = models_dir / f"river_checkpoint_{traversal_count}.pt"
            torch.save({
                'traversals': traversal_count,
                'training_steps': step,
                'advantage_network': trainer.advantage_net.state_dict(),
                'target_network': trainer.target_net.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'strategy_sum': trainer.strategy_sum,
                'config': {'hidden_size': 512, 'num_layers': 3, 'learning_rate': args.lr,
                          'buffer_capacity': args.buffer, 'traversals_per_step': TRAVERSALS_PER_STEP}
            }, cp_path)
            print(f"Saved: {cp_path.name}", flush=True)

            result = evaluator.evaluate(game, strategy, random_bot, 5000)
            print(f"vs RandomBot: {result.avg_mbb_per_hand:+.0f} mbb/h\n", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min ({traversal_count:,} traversals)", flush=True)

    final_path = models_dir / "river_v1_turbo.pt"
    strategy = trainer.get_all_average_strategies()
    torch.save({
        'traversals': traversal_count,
        'training_steps': args.iterations,
        'advantage_network': trainer.advantage_net.state_dict(),
        'final_strategy': strategy,
        'config': {'total_traversals': traversal_count, 'traversals_per_step': TRAVERSALS_PER_STEP}
    }, final_path)
    print(f"Final: {final_path}", flush=True)

if __name__ == "__main__":
    main()
