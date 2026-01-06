"""Tkinter GUI frontend for Deep PDCFR+ training visualization.

Adapted from poker_solver-main with simplified interface for Deep RL experiments.
"""

from __future__ import annotations

# Silence verbose logging from matplotlib and PIL before they get imported
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import queue
import threading
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('TkAgg')  # Backend for embedding in Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from aion26.config import AionConfig, GameConfig, TrainingConfig, ModelConfig, AlgorithmConfig
from aion26.gui.model import TrainingThread, MetricsUpdate
import numpy as np


def _convert_strategy_to_heatmap(strategy_dict: dict, game_name: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert strategy dictionary to 2D array for heatmap visualization.

    Args:
        strategy_dict: Dictionary mapping info_state string to action probabilities
        game_name: Name of the game ("kuhn" or "leduc")

    Returns:
        Tuple of (heatmap_data, row_labels, col_labels)
    """
    if not strategy_dict:
        return np.array([[]]), [], []

    # Get action labels based on game
    if game_name == "kuhn":
        action_labels = ["Check", "Bet"]
    elif game_name == "leduc":
        action_labels = ["Fold", "Call", "Raise"]
    else:
        # Generic labels
        num_actions = len(next(iter(strategy_dict.values())))
        action_labels = [f"Action {i}" for i in range(num_actions)]

    # Sort info states for consistent ordering
    sorted_states = sorted(strategy_dict.keys())

    # For large state spaces, sample representative states
    if len(sorted_states) > 50:
        # For Leduc, prioritize states with cards
        if game_name == "leduc":
            # Take states from different categories
            round1_states = [s for s in sorted_states if len(s.split()) == 1][:15]
            round2_states = [s for s in sorted_states if len(s.split()) > 1][:35]
            sorted_states = round1_states + round2_states
        else:
            # Take evenly spaced sample
            step = len(sorted_states) // 50
            sorted_states = sorted_states[::step][:50]

    # Build 2D array: rows = info states, cols = actions
    num_states = len(sorted_states)
    num_actions = len(action_labels)
    heatmap_data = np.zeros((num_states, num_actions))

    for i, info_state in enumerate(sorted_states):
        strategy = strategy_dict[info_state]
        heatmap_data[i, :] = strategy[:num_actions]

    return heatmap_data, sorted_states, action_labels


def _convert_strategy_to_matrix(strategy_dict: dict, game_name: str) -> dict:
    """Convert strategy dictionary to matrix format for tree-like visualization.

    For Leduc: Creates a 3×3 matrix (private card × board card) for Round 2 states.
    For Kuhn: Shows betting tree structure.

    Args:
        strategy_dict: Dictionary mapping info_state string to action probabilities
        game_name: Name of the game ("kuhn" or "leduc")

    Returns:
        Dictionary with matrix data structure
    """
    if not strategy_dict:
        return {"game": game_name, "matrix": None}

    if game_name == "leduc":
        # Create 3×3 matrix for Leduc Round 2 (private × board)
        ranks = ["J", "Q", "K"]
        suits = ["s", "h"]  # Spades and hearts

        # Initialize 3×3 matrix
        matrix = {}
        for private_rank in ranks:
            for board_rank in ranks:
                key = (private_rank, board_rank)
                matrix[key] = {"fold": [], "call": [], "raise": []}

        # Populate matrix from strategy dict
        for info_state, strategy in strategy_dict.items():
            # Parse Round 2 states (format: "Js Qh" or "Ks Jh p")
            parts = info_state.split()
            if len(parts) < 2:
                continue  # Skip Round 1 states

            # Extract private and board cards
            private_card = parts[0]  # e.g., "Js"
            board_card = parts[1]    # e.g., "Qh"

            if len(private_card) < 2 or len(board_card) < 2:
                continue

            private_rank = private_card[0]  # "J", "Q", or "K"
            board_rank = board_card[0]

            if private_rank in ranks and board_rank in ranks:
                key = (private_rank, board_rank)
                if key in matrix:
                    # Store strategy (fold, call, raise)
                    if len(strategy) >= 3:
                        matrix[key]["fold"].append(strategy[0])
                        matrix[key]["call"].append(strategy[1])
                        matrix[key]["raise"].append(strategy[2])

        # Average strategies for each cell
        matrix_avg = {}
        for key, actions in matrix.items():
            if actions["fold"]:  # If we have data for this cell
                matrix_avg[key] = {
                    "fold": np.mean(actions["fold"]),
                    "call": np.mean(actions["call"]),
                    "raise": np.mean(actions["raise"]),
                }

        return {
            "game": "leduc",
            "ranks": ranks,
            "matrix": matrix_avg,
        }

    elif game_name == "kuhn":
        # For Kuhn, show betting tree
        # Round 1: Initial card (J, Q, K)
        # Round 2: After opponent action (p = pass/check, b = bet)

        tree = {"J": {}, "Q": {}, "K": {}}

        for info_state, strategy in strategy_dict.items():
            # Parse Kuhn states
            if " " in info_state:
                card, history = info_state.split(" ", 1)
            else:
                card = info_state
                history = ""

            if card in tree:
                if len(strategy) >= 2:
                    tree[card][history] = {
                        "check": strategy[0],
                        "bet": strategy[1],
                    }

        return {
            "game": "kuhn",
            "tree": tree,
        }

    return {"game": game_name, "matrix": None}


class DeepCFRVisualizer:
    """Main GUI application for Deep PDCFR+ training.

    Layout:
    - Left Panel: Configuration inputs (game, algorithm, hyperparameters)
    - Right Panel:
        - Top: Real-time NashConv plot
        - Bottom: Strategy inspector (text display)
    - Bottom: Control buttons (Start, Stop, Save, Load)
    """

    def __init__(self, master: tk.Tk):
        """Initialize the visualizer.

        Args:
            master: Tkinter root window
        """
        self.master = master
        self.master.title("Aion-26 Deep PDCFR+ Visualizer")
        self.master.geometry("1200x800")

        # Training state
        self.training_thread: Optional[TrainingThread] = None
        self.metrics_queue: queue.Queue[MetricsUpdate] = queue.Queue()
        self.stop_event = threading.Event()
        self.is_training = False

        # Metrics history for plotting
        self.iterations = []
        self.nash_convs = []
        self.losses = []

        # Heatmap colorbar reference
        self.heatmap_colorbar = None

        # Build UI
        self._build_ui()

        # Start polling for metrics
        self._poll_metrics()

    def _build_ui(self):
        """Build the user interface."""
        # Main container
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # Left panel: Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        self._build_config_panel(config_frame)

        # Right panel: Visualization
        viz_frame = ttk.Frame(main_frame)
        viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.rowconfigure(1, weight=1)
        self._build_visualization_panel(viz_frame)

        # Bottom: Control buttons
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self._build_control_buttons(button_frame)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

    def _build_config_panel(self, parent: ttk.LabelFrame):
        """Build configuration input panel."""
        row = 0

        # Game selection
        ttk.Label(parent, text="Game:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.game_var = tk.StringVar(value="leduc")
        game_combo = ttk.Combobox(
            parent, textvariable=self.game_var, values=["kuhn", "leduc", "river_holdem"], state="readonly"
        )
        game_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Algorithm selection
        ttk.Label(parent, text="Algorithm:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.algo_var = tk.StringVar(value="ddcfr")
        algo_combo = ttk.Combobox(
            parent, textvariable=self.algo_var,
            values=["uniform", "linear", "pdcfr", "ddcfr"], state="readonly"
        )
        algo_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Variance Reduction
        self.use_vr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Use Variance Reduction", variable=self.use_vr_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5
        )
        row += 1

        # Separator
        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        # Training hyperparameters
        ttk.Label(parent, text="Iterations:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.iterations_var = tk.StringVar(value="2000")  # Demo-friendly: enough to fill buffer
        ttk.Entry(parent, textvariable=self.iterations_var).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        ttk.Label(parent, text="Batch Size:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.batch_size_var = tk.StringVar(value="128")
        ttk.Entry(parent, textvariable=self.batch_size_var).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        ttk.Label(parent, text="Buffer Capacity:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.buffer_capacity_var = tk.StringVar(value="1000")  # Demo-friendly: fills quickly
        ttk.Entry(parent, textvariable=self.buffer_capacity_var).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        # Separator
        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        # Model hyperparameters
        ttk.Label(parent, text="Hidden Size:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.hidden_size_var = tk.StringVar(value="128")
        ttk.Entry(parent, textvariable=self.hidden_size_var).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        ttk.Label(parent, text="Num Layers:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.num_layers_var = tk.StringVar(value="4")
        ttk.Entry(parent, textvariable=self.num_layers_var).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        ttk.Label(parent, text="Learning Rate:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.learning_rate_var = tk.StringVar(value="0.001")
        ttk.Entry(parent, textvariable=self.learning_rate_var).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        # Separator
        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        # Discounting parameters
        ttk.Label(parent, text="Alpha (regret):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.alpha_var = tk.StringVar(value="1.5")
        ttk.Entry(parent, textvariable=self.alpha_var).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        ttk.Label(parent, text="Beta (negative):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.beta_var = tk.StringVar(value="0.0")
        ttk.Entry(parent, textvariable=self.beta_var).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

        ttk.Label(parent, text="Gamma (strategy):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.gamma_var = tk.StringVar(value="2.0")
        ttk.Entry(parent, textvariable=self.gamma_var).grid(
            row=row, column=1, sticky=(tk.W, tk.E), pady=5
        )
        row += 1

    def _build_visualization_panel(self, parent: ttk.Frame):
        """Build visualization panel with plot and strategy inspector."""
        # Top: NashConv plot
        plot_frame = ttk.LabelFrame(parent, text="NashConv Progress", padding="10")
        plot_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("NashConv")
        self.ax.set_title("Convergence to Nash Equilibrium")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Bottom: Strategy inspector with tabs
        strategy_frame = ttk.LabelFrame(parent, text="Strategy Inspector", padding="10")
        strategy_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        strategy_frame.columnconfigure(0, weight=1)
        strategy_frame.rowconfigure(0, weight=1)

        # Create notebook (tabbed interface)
        self.strategy_notebook = ttk.Notebook(strategy_frame)
        self.strategy_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tab 1: Text view
        text_tab = ttk.Frame(self.strategy_notebook)
        self.strategy_notebook.add(text_tab, text="Text View")
        text_tab.columnconfigure(0, weight=1)
        text_tab.rowconfigure(0, weight=1)

        self.strategy_text = scrolledtext.ScrolledText(
            text_tab, wrap=tk.WORD, height=15, font=("Courier", 10)
        )
        self.strategy_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.strategy_text.insert("1.0", "Strategy will be displayed here after training starts...\n")
        self.strategy_text.config(state=tk.DISABLED)

        # Tab 2: Heatmap view
        heatmap_tab = ttk.Frame(self.strategy_notebook)
        self.strategy_notebook.add(heatmap_tab, text="Heatmap View")
        heatmap_tab.columnconfigure(0, weight=1)
        heatmap_tab.rowconfigure(0, weight=1)

        # Create heatmap figure
        self.heatmap_fig = Figure(figsize=(8, 6), dpi=100)
        self.heatmap_ax = self.heatmap_fig.add_subplot(111)
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=heatmap_tab)
        self.heatmap_canvas.draw()
        self.heatmap_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initialize with placeholder
        self.heatmap_ax.text(0.5, 0.5, "Strategy heatmap will appear here after training starts...",
                            horizontalalignment='center', verticalalignment='center',
                            transform=self.heatmap_ax.transAxes, fontsize=12)
        self.heatmap_ax.axis('off')

        # Tab 3: Matrix view (tree-like for Leduc)
        matrix_tab = ttk.Frame(self.strategy_notebook)
        self.strategy_notebook.add(matrix_tab, text="Matrix View")
        matrix_tab.columnconfigure(0, weight=1)
        matrix_tab.rowconfigure(0, weight=1)

        # Create matrix figure
        self.matrix_fig = Figure(figsize=(8, 6), dpi=100)
        self.matrix_ax = self.matrix_fig.add_subplot(111)
        self.matrix_canvas = FigureCanvasTkAgg(self.matrix_fig, master=matrix_tab)
        self.matrix_canvas.draw()
        self.matrix_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initialize with placeholder
        self.matrix_ax.text(0.5, 0.5, "Strategy matrix will appear here after training starts...",
                           horizontalalignment='center', verticalalignment='center',
                           transform=self.matrix_ax.transAxes, fontsize=12)
        self.matrix_ax.axis('off')

    def _build_control_buttons(self, parent: ttk.Frame):
        """Build control buttons."""
        self.start_button = ttk.Button(parent, text="Start Training", command=self._start_training)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(
            parent, text="Stop Training", command=self._stop_training, state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, padx=5)

        ttk.Button(parent, text="Save Config", command=self._save_config).grid(
            row=0, column=2, padx=5
        )
        ttk.Button(parent, text="Load Config", command=self._load_config).grid(
            row=0, column=3, padx=5
        )

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_label.grid(row=0, column=4, padx=20, sticky=(tk.W, tk.E))
        parent.columnconfigure(4, weight=1)

    def _get_config_from_ui(self) -> AionConfig:
        """Build AionConfig from UI inputs."""
        game = GameConfig(name=self.game_var.get())
        training = TrainingConfig(
            iterations=int(self.iterations_var.get()),
            batch_size=int(self.batch_size_var.get()),
            buffer_capacity=int(self.buffer_capacity_var.get()),
        )
        model = ModelConfig(
            hidden_size=int(self.hidden_size_var.get()),
            num_hidden_layers=int(self.num_layers_var.get()),
            learning_rate=float(self.learning_rate_var.get()),
        )
        algorithm = AlgorithmConfig(
            use_vr=self.use_vr_var.get(),
            scheduler_type=self.algo_var.get(),
            alpha=float(self.alpha_var.get()),
            beta=float(self.beta_var.get()),
            gamma=float(self.gamma_var.get()),
        )

        return AionConfig(
            game=game,
            training=training,
            model=model,
            algorithm=algorithm,
            name=f"{game.name}_{algorithm.scheduler_type}_gui",
        )

    def _start_training(self):
        """Start training in background thread."""
        if self.is_training:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        try:
            # Get config from UI
            config = self._get_config_from_ui()

            # Clear previous data
            self.iterations.clear()
            self.nash_convs.clear()
            self.losses.clear()
            self.ax.clear()
            self.ax.set_xlabel("Iteration")
            self.ax.set_ylabel("NashConv")
            self.ax.set_title("Convergence to Nash Equilibrium")
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()

            # Clear heatmap
            self.heatmap_ax.clear()
            self.heatmap_ax.text(0.5, 0.5, "Strategy heatmap will appear here after training starts...",
                                horizontalalignment='center', verticalalignment='center',
                                transform=self.heatmap_ax.transAxes, fontsize=12)
            self.heatmap_ax.axis('off')
            self.heatmap_colorbar = None
            self.heatmap_canvas.draw()

            # Clear matrix
            self.matrix_ax.clear()
            self.matrix_ax.text(0.5, 0.5, "Strategy matrix will appear here after training starts...",
                               horizontalalignment='center', verticalalignment='center',
                               transform=self.matrix_ax.transAxes, fontsize=12)
            self.matrix_ax.axis('off')
            self.matrix_canvas.draw()

            # Start training thread
            self.stop_event.clear()
            self.training_thread = TrainingThread(config, self.metrics_queue, self.stop_event)
            self.training_thread.start()

            # Update UI state
            self.is_training = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set(f"Training {config.name}...")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")

    def _stop_training(self):
        """Stop training."""
        if not self.is_training:
            return

        self.stop_event.set()
        self.status_var.set("Stopping training...")

    def _save_config(self):
        """Save current configuration to YAML."""
        try:
            config = self._get_config_from_ui()
            filepath = filedialog.asksaveasfilename(
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            )
            if filepath:
                config.to_yaml(filepath)
                messagebox.showinfo("Success", f"Configuration saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {str(e)}")

    def _load_config(self):
        """Load configuration from YAML."""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
            )
            if filepath:
                config = AionConfig.from_yaml(filepath)
                # Update UI
                self.game_var.set(config.game.name)
                self.algo_var.set(config.algorithm.scheduler_type)
                self.use_vr_var.set(config.algorithm.use_vr)
                self.iterations_var.set(str(config.training.iterations))
                self.batch_size_var.set(str(config.training.batch_size))
                self.buffer_capacity_var.set(str(config.training.buffer_capacity))
                self.hidden_size_var.set(str(config.model.hidden_size))
                self.num_layers_var.set(str(config.model.num_hidden_layers))
                self.learning_rate_var.set(str(config.model.learning_rate))
                self.alpha_var.set(str(config.algorithm.alpha))
                self.beta_var.set(str(config.algorithm.beta))
                self.gamma_var.set(str(config.algorithm.gamma))
                messagebox.showinfo("Success", f"Configuration loaded from {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {str(e)}")

    def _poll_metrics(self):
        """Poll metrics queue and update UI (runs in main thread)."""
        try:
            while True:
                metrics = self.metrics_queue.get_nowait()
                self._update_ui_with_metrics(metrics)

                if metrics.status == "completed":
                    self._training_completed()
                elif metrics.status == "error":
                    self._training_error(metrics.error_message or "Unknown error")

        except queue.Empty:
            pass

        # Schedule next poll
        self.master.after(100, self._poll_metrics)

    def _update_ui_with_metrics(self, metrics: MetricsUpdate):
        """Update UI with new metrics."""
        # Update plot
        if metrics.nash_conv is not None:
            self.iterations.append(metrics.iteration)
            self.nash_convs.append(metrics.nash_conv)

            self.ax.clear()
            self.ax.plot(self.iterations, self.nash_convs, 'b-', linewidth=2, label="NashConv")
            self.ax.set_xlabel("Iteration")
            self.ax.set_ylabel("NashConv")
            self.ax.set_title("Convergence to Nash Equilibrium")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            self.canvas.draw()

        # Update strategy inspector
        if metrics.strategy is not None:
            # Update text view
            self.strategy_text.config(state=tk.NORMAL)
            self.strategy_text.delete("1.0", tk.END)
            self.strategy_text.insert("1.0", f"Iteration {metrics.iteration}\n")
            self.strategy_text.insert(tk.END, f"NashConv: {metrics.nash_conv:.6f}\n")
            self.strategy_text.insert(tk.END, f"Loss: {metrics.loss:.6f}\n")
            self.strategy_text.insert(tk.END, f"Value Loss: {metrics.value_loss:.6f}\n")
            self.strategy_text.insert(tk.END, f"Buffer: {metrics.buffer_size} ({metrics.buffer_fill_pct:.1f}%)\n")
            self.strategy_text.insert(tk.END, "\n" + "="*60 + "\n\n")

            # Display strategy for each information set
            for info_state, strategy in sorted(metrics.strategy.items()):
                self.strategy_text.insert(tk.END, f"{info_state}:\n")
                for i, prob in enumerate(strategy):
                    self.strategy_text.insert(tk.END, f"  Action {i}: {prob:.4f}\n")
                self.strategy_text.insert(tk.END, "\n")

            self.strategy_text.config(state=tk.DISABLED)

            # Update heatmap view
            self._update_strategy_heatmap(metrics.strategy)

            # Update matrix view
            self._update_strategy_matrix(metrics.strategy)

        # Update status
        nashconv_str = f"{metrics.nash_conv:.6f}" if metrics.nash_conv is not None else "N/A"
        self.status_var.set(
            f"Iteration {metrics.iteration} | Loss: {metrics.loss:.6f} | "
            f"NashConv: {nashconv_str}"
        )

    def _update_strategy_heatmap(self, strategy_dict: dict):
        """Update the strategy heatmap visualization.

        Args:
            strategy_dict: Dictionary mapping info_state to action probabilities
        """
        try:
            # Get game name from current config
            game_name = self.game_var.get()

            # Convert strategy to heatmap format
            heatmap_data, row_labels, col_labels = _convert_strategy_to_heatmap(
                strategy_dict, game_name
            )

            if heatmap_data.size == 0:
                return

            # Clear previous plot
            self.heatmap_ax.clear()

            # Create heatmap
            im = self.heatmap_ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            # Set ticks and labels
            self.heatmap_ax.set_xticks(np.arange(len(col_labels)))
            self.heatmap_ax.set_yticks(np.arange(len(row_labels)))
            self.heatmap_ax.set_xticklabels(col_labels)
            self.heatmap_ax.set_yticklabels(row_labels, fontsize=8)

            # Rotate the tick labels for better readability
            plt.setp(self.heatmap_ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Add colorbar
            if not hasattr(self, 'heatmap_colorbar') or self.heatmap_colorbar is None:
                self.heatmap_colorbar = self.heatmap_fig.colorbar(im, ax=self.heatmap_ax)
                self.heatmap_colorbar.set_label('Action Probability', rotation=270, labelpad=15)

            # Add title
            self.heatmap_ax.set_title(f'Strategy Heatmap ({game_name.capitalize()} Poker)')
            self.heatmap_ax.set_xlabel('Actions')
            self.heatmap_ax.set_ylabel('Information States')

            # Annotate cells with probability values (for small grids)
            if len(row_labels) <= 20 and len(col_labels) <= 5:
                for i in range(len(row_labels)):
                    for j in range(len(col_labels)):
                        text = self.heatmap_ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                                   ha="center", va="center", color="black",
                                                   fontsize=8)

            # Adjust layout
            self.heatmap_fig.tight_layout()

            # Redraw canvas
            self.heatmap_canvas.draw()

        except Exception as e:
            logger.error(f"Error updating strategy heatmap: {e}")
            # Show error message in heatmap
            self.heatmap_ax.clear()
            self.heatmap_ax.text(0.5, 0.5, f"Error: {str(e)}",
                                horizontalalignment='center', verticalalignment='center',
                                transform=self.heatmap_ax.transAxes, fontsize=10, color='red')
            self.heatmap_ax.axis('off')
            self.heatmap_canvas.draw()

    def _update_strategy_matrix(self, strategy_dict: dict):
        """Update the strategy matrix visualization.

        For Leduc: 3×3 grid showing private card × board card strategies.
        For Kuhn: Tree structure showing betting sequences.

        Args:
            strategy_dict: Dictionary mapping info_state to action probabilities
        """
        try:
            # Get game name from current config
            game_name = self.game_var.get()

            # Convert strategy to matrix format
            matrix_data = _convert_strategy_to_matrix(strategy_dict, game_name)

            if matrix_data["matrix"] is None:
                return

            # Clear previous plot
            self.matrix_ax.clear()

            if game_name == "leduc":
                self._draw_leduc_matrix(matrix_data)
            elif game_name == "kuhn":
                self._draw_kuhn_tree(matrix_data)

            # Adjust layout
            self.matrix_fig.tight_layout()

            # Redraw canvas
            self.matrix_canvas.draw()

        except Exception as e:
            logger.error(f"Error updating strategy matrix: {e}")
            # Show error message in matrix
            self.matrix_ax.clear()
            self.matrix_ax.text(0.5, 0.5, f"Error: {str(e)}",
                               horizontalalignment='center', verticalalignment='center',
                               transform=self.matrix_ax.transAxes, fontsize=10, color='red')
            self.matrix_ax.axis('off')
            self.matrix_canvas.draw()

    def _draw_leduc_matrix(self, matrix_data: dict):
        """Draw 3×3 matrix for Leduc poker strategies.

        Args:
            matrix_data: Dictionary with ranks and matrix data
        """
        ranks = matrix_data["ranks"]
        matrix = matrix_data["matrix"]

        # Create 3×3 grid
        n = len(ranks)

        # Draw grid lines
        for i in range(n + 1):
            self.matrix_ax.axhline(i, color='black', linewidth=2)
            self.matrix_ax.axvline(i, color='black', linewidth=2)

        # Add labels
        for i, rank in enumerate(ranks):
            # Row labels (private card)
            self.matrix_ax.text(-0.3, i + 0.5, f"{rank}♠/♥",
                               ha='right', va='center', fontsize=12, fontweight='bold')
            # Column labels (board card)
            self.matrix_ax.text(i + 0.5, n + 0.2, f"{rank}♠/♥",
                               ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Add axis labels
        self.matrix_ax.text(-0.8, n/2, "Private Card",
                           ha='center', va='center', fontsize=14, fontweight='bold',
                           rotation=90)
        self.matrix_ax.text(n/2, n + 0.6, "Board Card",
                           ha='center', va='top', fontsize=14, fontweight='bold')

        # Fill cells with strategy data
        for i, private_rank in enumerate(ranks):
            for j, board_rank in enumerate(ranks):
                key = (private_rank, board_rank)

                if key in matrix:
                    strategy = matrix[key]
                    fold_prob = strategy["fold"]
                    call_prob = strategy["call"]
                    raise_prob = strategy["raise"]

                    # Draw pie chart in cell
                    self._draw_pie_in_cell(i, j, fold_prob, call_prob, raise_prob)

                    # Add text with probabilities
                    text = f"F:{fold_prob:.2f}\nC:{call_prob:.2f}\nR:{raise_prob:.2f}"
                    self.matrix_ax.text(j + 0.5, i + 0.5, text,
                                       ha='center', va='center', fontsize=8)
                else:
                    # No data for this cell
                    self.matrix_ax.text(j + 0.5, i + 0.5, "N/A",
                                       ha='center', va='center', fontsize=10, color='gray')

        # Set limits and aspect
        self.matrix_ax.set_xlim(-0.5, n)
        self.matrix_ax.set_ylim(-0.5, n + 0.5)
        self.matrix_ax.set_aspect('equal')
        self.matrix_ax.axis('off')
        self.matrix_ax.set_title("Leduc Strategy Matrix (Round 2)", fontsize=14, fontweight='bold', pad=20)

    def _draw_pie_in_cell(self, row: int, col: int, fold: float, call: float, raise_prob: float):
        """Draw a small pie chart in a matrix cell.

        Args:
            row: Row index
            col: Column index
            fold: Fold probability
            call: Call probability
            raise_prob: Raise probability
        """
        # Only draw if probabilities are meaningful
        if fold + call + raise_prob < 0.01:
            return

        # Create mini pie chart
        sizes = [fold, call, raise_prob]
        colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']  # Red, teal, light green
        explode = (0.05, 0.05, 0.05)

        # Position in cell (background)
        center_x = col + 0.5
        center_y = row + 0.5

        # Draw semi-transparent background pie
        wedges, texts = self.matrix_ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            radius=0.35,
            center=(center_x, center_y),
            wedgeprops=dict(alpha=0.3, edgecolor='white', linewidth=1)
        )

    def _draw_kuhn_tree(self, matrix_data: dict):
        """Draw betting tree for Kuhn poker.

        Args:
            matrix_data: Dictionary with tree structure
        """
        tree = matrix_data["tree"]

        # Simple tree visualization
        # Level 1: Initial cards (J, Q, K)
        # Level 2: After opponent action

        self.matrix_ax.text(0.5, 0.95, "Kuhn Poker Strategy Tree",
                           ha='center', va='top', fontsize=14, fontweight='bold',
                           transform=self.matrix_ax.transAxes)

        y_positions = {"J": 0.7, "Q": 0.45, "K": 0.2}
        x_root = 0.1

        for card, y_pos in y_positions.items():
            # Draw card node
            self.matrix_ax.text(x_root, y_pos, f"{card}",
                               ha='center', va='center',
                               bbox=dict(boxstyle='circle', facecolor='lightblue', edgecolor='black'),
                               fontsize=12, fontweight='bold',
                               transform=self.matrix_ax.transAxes)

            # Draw strategy for this card
            if card in tree and "" in tree[card]:
                strategy = tree[card][""]
                check_prob = strategy["check"]
                bet_prob = strategy["bet"]

                text = f"Check: {check_prob:.2f}\nBet: {bet_prob:.2f}"
                self.matrix_ax.text(x_root + 0.15, y_pos, text,
                                   ha='left', va='center', fontsize=9,
                                   transform=self.matrix_ax.transAxes)

        self.matrix_ax.set_xlim(0, 1)
        self.matrix_ax.set_ylim(0, 1)
        self.matrix_ax.axis('off')

    def _training_completed(self):
        """Handle training completion."""
        self.is_training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Training completed!")
        messagebox.showinfo("Training Complete", "Training has finished successfully!")

    def _training_error(self, error_message: str):
        """Handle training error."""
        self.is_training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Error occurred during training")
        messagebox.showerror("Training Error", f"An error occurred:\n\n{error_message}")


def launch_gui():
    """Launch the Deep PDCFR+ Visualizer GUI."""
    root = tk.Tk()
    app = DeepCFRVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
