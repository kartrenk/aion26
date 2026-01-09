#!/usr/bin/env python3
"""Automatic Checkpoint Evaluator - Watch models/ folder and evaluate new checkpoints.

As soon as a checkpoint appears (e.g., river_checkpoint_500000.pt), this script:
1. Detects the new file
2. Loads the checkpoint
3. Runs 10,000 hand evaluation vs RandomBot, CallingStation, HonestBot
4. Logs results to evaluation_results.csv
5. Resumes watching

Critical Success Criteria:
- vs RandomBot: > +2,500 mbb/h at 1M checkpoint
- Loss: Stable and low
- No crashes

Usage:
    python scripts/auto_evaluate.py --watch

Or evaluate a specific checkpoint:
    python scripts/auto_evaluate.py --checkpoint models/river_checkpoint_500000.pt
"""

import sys
from pathlib import Path
import time
import argparse
import csv
from typing import Optional, Dict, Any
from datetime import datetime
import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from aion26.games.rust_wrapper import new_rust_river_game
from aion26.deep_cfr.networks import DeepCFRNetwork, HoldemEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.baselines import RandomBot, CallingStation, HonestBot
from aion26.metrics.evaluator import HeadToHeadEvaluator
from aion26.cfr.regret_matching import regret_matching


class CheckpointEvaluator:
    """Evaluate a single checkpoint."""

    def __init__(self, eval_hands: int = 10_000):
        self.eval_hands = eval_hands

        # Create baseline bots
        self.random_bot = RandomBot()
        self.calling_bot = CallingStation()
        self.honest_bot = HonestBot()

        # Create evaluator
        self.evaluator = HeadToHeadEvaluator(big_blind=2.0)

        # Create game (for evaluation)
        self.game = new_rust_river_game()  # Random boards for evaluation

        print(f"✅ Evaluator initialized ({eval_hands:,} hands per bot)")

    def load_checkpoint(self, checkpoint_path: Path) -> tuple[dict, dict]:
        """Load checkpoint and reconstruct trainer.

        Returns:
            (checkpoint_data, strategy)
        """
        print(f"\n📂 Loading checkpoint: {checkpoint_path.name}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract config
        config = checkpoint.get('config', {})
        hidden_size = config.get('hidden_size', 512)
        num_layers = config.get('num_layers', 3)

        print(f"   Config: hidden={hidden_size}, layers={num_layers}")

        # Reconstruct advantage network
        encoder = HoldemEncoder()
        advantage_network = DeepCFRNetwork(
            input_size=encoder.input_size,
            output_size=4,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            zero_init_output=True,
        )

        # Load weights
        advantage_network.load_state_dict(checkpoint['advantage_network'])
        advantage_network.eval()

        # Get strategy (from checkpoint if available, otherwise compute)
        if 'final_strategy' in checkpoint:
            strategy = checkpoint['final_strategy']
        elif 'strategy_sum' in checkpoint:
            # Compute average strategy from strategy_sum
            from aion26.cfr.regret_matching import regret_matching
            strategy_sum = checkpoint['strategy_sum']
            strategy = {}
            for info_state, action_sum in strategy_sum.items():
                strategy[info_state] = regret_matching(action_sum)
        else:
            print("   ⚠️  WARNING: No strategy found in checkpoint")
            strategy = {}

        print(f"   Strategy size: {len(strategy):,} states")

        return checkpoint, strategy

    def evaluate_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Evaluate a checkpoint against all bots.

        Returns:
            results: {
                'checkpoint': checkpoint_path,
                'iteration': iteration,
                'strategy_size': size,
                'random_bot_mbb': win_rate,
                'random_bot_ci': confidence_interval,
                'calling_station_mbb': ...,
                'honest_bot_mbb': ...,
                'timestamp': time,
            }
        """
        start_time = time.time()

        # Load checkpoint
        checkpoint, strategy = self.load_checkpoint(checkpoint_path)

        if len(strategy) == 0:
            print("   ❌ ERROR: No strategy to evaluate")
            return None

        # Extract metadata
        iteration = checkpoint.get('iteration', 0)

        # Evaluate vs each bot
        results = {
            'checkpoint': checkpoint_path.name,
            'iteration': iteration,
            'strategy_size': len(strategy),
            'timestamp': datetime.now().isoformat(),
        }

        opponents = {
            'random_bot': self.random_bot,
            'calling_station': self.calling_bot,
            'honest_bot': self.honest_bot,
        }

        print(f"\n{'='*80}")
        print(f"EVALUATING CHECKPOINT: {checkpoint_path.name} (Iter {iteration:,})")
        print(f"{'='*80}\n")

        for name, bot in opponents.items():
            print(f"Playing {self.eval_hands:,} hands vs {name.replace('_', ' ').title()}...")

            result = self.evaluator.evaluate(
                initial_state=self.game,
                strategy=strategy,
                opponent=bot,
                num_hands=self.eval_hands
            )

            win_rate = result.avg_mbb_per_hand
            ci = result.confidence_95

            print(f"  Win rate: {win_rate:+7.0f} mbb/h ± {ci:.0f} (95% CI)")
            print()

            # Store results
            results[f'{name}_mbb'] = win_rate
            results[f'{name}_ci'] = ci

        # Check success criteria
        random_mbb = results['random_bot_mbb']
        print(f"{'='*80}")
        print(f"SUCCESS CRITERIA CHECK")
        print(f"{'='*80}\n")

        success = random_mbb > 2500
        status = "✅ PASS" if success else "❌ FAIL"

        print(f"vs RandomBot: {random_mbb:+.0f} mbb/h (target: > +2,500) {status}")
        print()

        if success:
            print("🎉 CHECKPOINT PASSES! River model is healthy.")
        else:
            print("⚠️  CHECKPOINT BELOW TARGET. Need more training or investigate issues.")

        print()

        elapsed = time.time() - start_time
        print(f"Evaluation complete in {elapsed:.1f}s")
        print(f"{'='*80}\n")

        results['success'] = success
        results['eval_time_seconds'] = elapsed

        return results


class CheckpointWatcher:
    """Watch models/ folder for new checkpoints and auto-evaluate."""

    def __init__(self, models_dir: Path, results_csv: Path, eval_hands: int = 10_000):
        self.models_dir = models_dir
        self.results_csv = results_csv
        self.eval_hands = eval_hands

        # Track evaluated checkpoints
        self.evaluated = set()

        # Create evaluator
        self.evaluator = CheckpointEvaluator(eval_hands)

        # Initialize CSV
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not self.results_csv.exists():
            with open(self.results_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp',
                    'checkpoint',
                    'iteration',
                    'strategy_size',
                    'random_bot_mbb',
                    'random_bot_ci',
                    'calling_station_mbb',
                    'calling_station_ci',
                    'honest_bot_mbb',
                    'honest_bot_ci',
                    'success',
                    'eval_time_seconds',
                ])
                writer.writeheader()
            print(f"✅ Created results CSV: {self.results_csv}")

    def _log_milestone(self, results: dict):
        """Log major milestone to logs/major_milestones.log."""
        logs_dir = self.models_dir.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        milestone_log = logs_dir / "major_milestones.log"

        with open(milestone_log, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CHECKPOINT EVALUATION: {results['checkpoint']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Iteration: {results['iteration']:,}\n")
            f.write(f"Strategy Size: {results['strategy_size']:,} states\n")
            f.write(f"\n")
            f.write(f"Results:\n")
            f.write(f"  RandomBot:       {results['random_bot_mbb']:+7.0f} mbb/h ± {results['random_bot_ci']:.0f}\n")
            f.write(f"  CallingStation:  {results['calling_station_mbb']:+7.0f} mbb/h ± {results['calling_station_ci']:.0f}\n")
            f.write(f"  HonestBot:       {results['honest_bot_mbb']:+7.0f} mbb/h ± {results['honest_bot_ci']:.0f}\n")
            f.write(f"\n")
            f.write(f"Success: {'✅ PASS' if results['success'] else '❌ FAIL'}\n")
            f.write(f"Eval Time: {results['eval_time_seconds']:.1f}s\n")
            f.write(f"{'='*80}\n\n")

        print(f"✅ Milestone logged to {milestone_log}")

    def scan_for_checkpoints(self) -> list[Path]:
        """Scan models/ folder for new checkpoints.

        Returns:
            List of new checkpoint paths
        """
        if not self.models_dir.exists():
            return []

        # Find all checkpoint files
        checkpoints = list(self.models_dir.glob("river_checkpoint_*.pt"))

        # Filter out already evaluated
        new_checkpoints = [cp for cp in checkpoints if cp not in self.evaluated]

        # Sort by iteration number
        def extract_iteration(path):
            try:
                # Extract number from "river_checkpoint_500000.pt"
                return int(path.stem.split('_')[-1])
            except:
                return 0

        new_checkpoints.sort(key=extract_iteration)

        return new_checkpoints

    def evaluate_and_log(self, checkpoint_path: Path):
        """Evaluate checkpoint and log results."""
        # Evaluate
        results = self.evaluator.evaluate_checkpoint(checkpoint_path)

        if results is None:
            print("⚠️  Evaluation failed, skipping logging")
            return

        # Log to CSV
        with open(self.results_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp',
                'checkpoint',
                'iteration',
                'strategy_size',
                'random_bot_mbb',
                'random_bot_ci',
                'calling_station_mbb',
                'calling_station_ci',
                'honest_bot_mbb',
                'honest_bot_ci',
                'success',
                'eval_time_seconds',
            ])
            writer.writerow(results)

        print(f"✅ Results logged to {self.results_csv}")

        # Log to major milestones
        self._log_milestone(results)

        # Mark as evaluated
        self.evaluated.add(checkpoint_path)

    def watch(self, check_interval: int = 60):
        """Watch for new checkpoints and evaluate.

        Args:
            check_interval: Seconds between scans (default: 60)
        """
        print("="*80)
        print("AUTOMATIC CHECKPOINT EVALUATOR - WATCHING")
        print("="*80)
        print()
        print(f"Watching: {self.models_dir}")
        print(f"Results: {self.results_csv}")
        print(f"Check interval: {check_interval}s")
        print(f"Eval hands: {self.eval_hands:,} per bot")
        print()
        print("Waiting for checkpoints...")
        print("Press Ctrl+C to exit")
        print("="*80)
        print()

        try:
            while True:
                # Scan for new checkpoints
                new_checkpoints = self.scan_for_checkpoints()

                if new_checkpoints:
                    print(f"\n🔔 Detected {len(new_checkpoints)} new checkpoint(s)")

                    for cp in new_checkpoints:
                        print(f"\n{'='*80}")
                        print(f"NEW CHECKPOINT DETECTED: {cp.name}")
                        print(f"{'='*80}")

                        self.evaluate_and_log(cp)

                        print()

                # Wait before next scan
                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\n✅ Watcher stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Automatic Checkpoint Evaluator')
    parser.add_argument('--watch', action='store_true',
                       help='Watch models/ folder for new checkpoints')
    parser.add_argument('--checkpoint', type=str,
                       help='Evaluate a specific checkpoint file')
    parser.add_argument('--hands', type=int, default=10_000,
                       help='Number of hands per bot (default: 10,000)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds (default: 60)')
    args = parser.parse_args()

    # Paths
    models_dir = project_root / "models"
    results_csv = project_root / "evaluation_results.csv"

    if args.checkpoint:
        # Evaluate specific checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
            return 1

        evaluator = CheckpointEvaluator(eval_hands=args.hands)
        results = evaluator.evaluate_checkpoint(checkpoint_path)

        if results and results['success']:
            return 0
        else:
            return 1

    elif args.watch:
        # Watch mode
        models_dir.mkdir(exist_ok=True)

        watcher = CheckpointWatcher(
            models_dir=models_dir,
            results_csv=results_csv,
            eval_hands=args.hands,
        )

        watcher.watch(check_interval=args.interval)
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
