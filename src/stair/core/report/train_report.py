import json
import re
from pathlib import Path

import plotext


def find_latest_checkpoint_state(run_dir):
    run_dir_path = Path(run_dir)
    checkpoint_dir_pattern = re.compile(r"^checkpoint-(\d+)$")

    numbered_checkpoints = []
    for candidate_dir in run_dir_path.glob("checkpoint-*"):
        match = checkpoint_dir_pattern.match(candidate_dir.name)
        if match is None:
            continue
        step_number = int(match.group(1))
        state_path = candidate_dir / "trainer_state.json"
        if state_path.exists():
            numbered_checkpoints.append((step_number, state_path))

    if not numbered_checkpoints:
        raise FileNotFoundError(f"no checkpoints with trainer_state.json found under {run_dir}")

    numbered_checkpoints.sort(key=lambda entry: entry[0])
    highest_step_path = numbered_checkpoints[-1][1]
    return str(highest_step_path)


def load_log_history(trainer_state_path):
    with open(trainer_state_path, "r") as trainer_state_file:
        trainer_state = json.load(trainer_state_file)
    return trainer_state["log_history"]


def split_log_history(log_history):
    train_entries = []
    eval_entries = []
    for entry in log_history:
        if "eval_loss" in entry:
            eval_entries.append(entry)
        elif "loss" in entry:
            train_entries.append(entry)
    return train_entries, eval_entries


def diagnose(train_entries, eval_entries, report_config, num_train_epochs):
    diagnostics = []

    if not train_entries:
        diagnostics.append("No training data found in this run's log history.")
        return diagnostics

    if not eval_entries:
        diagnostics.append(
            "No eval data found in this run's log history — cannot assess overfitting "
            "or plateauing. Check eval_strategy/eval_interval in your config."
        )
        return diagnostics

    if len(eval_entries) < 2:
        diagnostics.append(
            "Not enough eval data points yet (only 1 so far) to assess a trend. "
            "Check back after more evals have run."
        )
        return diagnostics

    overfitting_patience_evals = report_config["overfitting_patience_evals"]
    rising_streak = 0
    for entry_index in range(1, len(eval_entries)):
        if eval_entries[entry_index]["eval_loss"] > eval_entries[entry_index - 1]["eval_loss"]:
            rising_streak += 1
        else:
            rising_streak = 0
        if rising_streak >= overfitting_patience_evals:
            diagnostics.append(
                f"Overfitting: eval loss has risen for {rising_streak} consecutive evals "
                f"while train loss continues to fall. Try a lower learning rate, fewer "
                f"epochs, or more training data."
            )
            break

    plateau_patience_evals = report_config["plateau_patience_evals"]
    plateau_relative_improvement = report_config["plateau_relative_improvement"]
    if len(eval_entries) >= plateau_patience_evals:
        recent_eval_losses = []
        for entry in eval_entries[-plateau_patience_evals:]:
            recent_eval_losses.append(entry["eval_loss"])
        best_recent_loss = min(recent_eval_losses)
        earliest_recent_loss = recent_eval_losses[0]
        relative_improvement = (earliest_recent_loss - best_recent_loss) / earliest_recent_loss
        if relative_improvement < plateau_relative_improvement:
            diagnostics.append(
                f"Plateaued: eval loss has improved by less than "
                f"{plateau_relative_improvement * 100:.1f}% over the last "
                f"{plateau_patience_evals} evals. More epochs are unlikely to help much "
                f"without other changes."
            )

    last_train_epoch = train_entries[-1]["epoch"]
    if last_train_epoch < num_train_epochs:
        epochs_completed_fraction = last_train_epoch / num_train_epochs
        if epochs_completed_fraction < 0.9:
            diagnostics.append(
                f"Training stopped early at epoch {last_train_epoch:.1f} of "
                f"{num_train_epochs} configured — likely triggered by early stopping."
            )

    if not diagnostics:
        diagnostics.append("No issues detected: eval loss is trending down without a plateau.")

    return diagnostics


def render_curves(train_entries, eval_entries):
    if not train_entries:
        print("No training data yet.")
    else:
        train_steps = []
        train_losses = []
        for entry in train_entries:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        plotext.plot(train_steps, train_losses, label="train loss")

    if not eval_entries:
        print("No eval data yet.")
    else:
        eval_steps = []
        eval_losses = []
        for entry in eval_entries:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])
        plotext.plot(eval_steps, eval_losses, label="eval loss")

    if train_entries or eval_entries:
        plotext.title("Training curves")
        plotext.xlabel("step")
        plotext.ylabel("loss")
        plotext.show()
