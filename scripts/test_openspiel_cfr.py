"""Test OpenSpiel's built-in CFR solver on Leduc.

This establishes a baseline to confirm:
1. OpenSpiel's CFR works
2. What exploitability we should expect
3. How many iterations are needed
"""

import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability

def main():
    print("=" * 70)
    print("OpenSpiel Built-in CFR on Leduc Poker (Baseline)")
    print("=" * 70)
    print()

    # Load game
    game = pyspiel.load_game("leduc_poker")
    print(f"Game: {game.get_type().long_name}")
    print()

    # Create CFR solver
    cfr_solver = cfr.CFRSolver(game)

    print("Training with OpenSpiel's CFR...")
    print()
    print("  Iter    NashConv")
    print("  ----    --------")

    for i in range(1, 2001):
        cfr_solver.evaluate_and_update_policy()

        if i % 500 == 0 or i == 1:
            nash_conv = exploitability.nash_conv(game, cfr_solver.average_policy())
            print(f"  {i:4d}    {nash_conv:8.4f}")

    print()
    final_nash_conv = exploitability.nash_conv(game, cfr_solver.average_policy())
    print(f"Final NashConv: {final_nash_conv:.4f}")
    print()

    if final_nash_conv < 0.1:
        print("✅ OpenSpiel CFR works - converged to < 0.1")
    else:
        print("⚠️  OpenSpiel CFR didn't fully converge in 2000 iterations")

    print("=" * 70)

if __name__ == "__main__":
    main()
