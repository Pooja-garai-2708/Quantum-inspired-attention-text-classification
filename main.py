"""Command-line runner for the Quantum-Inspired Attention project.

The project is organised as independent scripts by phase.  This module keeps
those scripts independent while providing one predictable entry point to run a
single phase or the full experiment in the required order.

Examples
--------
List the available phases::

    python main.py --list

Run preprocessing only::

    python main.py --phase preprocessing

Run every phase (downloads data and retrains models)::

    python main.py --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


# Each command is deliberately executed in a separate Python process.  The
# phase scripts contain their own imports and paths, so this avoids import-name
# clashes between similarly named modules (for example, both model.py files).
PHASES: dict[str, tuple[str, tuple[Path, ...]]] = {
    "dataset": (
        "Download, explore, and describe the IMDB dataset.",
        (
            Path("phase_1_dataset/download_dataset.py"),
            Path("phase_1_dataset/explore_dataset.py"),
            Path("phase_1_dataset/dataset_statistics.py"),
        ),
    ),
    "preprocessing": (
        "Clean, tokenize, encode, and pad the reviews.",
        (
            Path("phase_2_preprocessing/clean_text.py"),
            Path("phase_2_preprocessing/text_tokenization.py"),
            Path("phase_2_preprocessing/build_vocabulary.py"),
            Path("phase_2_preprocessing/encode_text.py"),
            Path("phase_2_preprocessing/create_padding.py"),
        ),
    ),
    "baseline": (
        "Train and evaluate the BiLSTM standard-attention baseline.",
        (
            Path("phase_3_baseline_model/train.py"),
            Path("phase_3_baseline_model/evaluate.py"),
        ),
    ),
    "quantum": (
        "Train and evaluate the BiLSTM quantum-inspired-attention model.",
        (
            Path("phase_4_quantum_inspired_attention/train.py"),
            Path("phase_4_quantum_inspired_attention/evaluate.py"),
        ),
    ),
    "comparison": (
        "Compare models and run statistical and computational analyses.",
        (
            Path("phase_5_comparison/compare_models.py"),
            Path("phase_5_comparison/statistical_analysis.py"),
            Path("phase_5_comparison/computational_analysis.py"),
        ),
    ),
    "visualization": (
        "Extract and visualize standard and quantum attention weights.",
        (
            Path("phase_6_attention_visualization/extract_standard_attention.py"),
            Path("phase_6_attention_visualization/extract_quantum_attention.py"),
            Path("phase_6_attention_visualization/visualize_attention.py"),
        ),
    ),
    "ablation": (
        "Run the mean-pooling ablation experiment.",
        (
            Path("phase_7_final_analysis/ablation/train_ablation.py"),
            Path("phase_7_final_analysis/ablation/evaluate_ablation.py"),
        ),
    ),
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Run the Quantum-Inspired Attention experiment pipeline."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--phase",
        choices=tuple(PHASES),
        help="Run one named phase.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every phase in order (may download data and retrain models).",
    )
    selection.add_argument(
        "--list",
        action="store_true",
        help="Display available phases and exit.",
    )
    return parser.parse_args()


def print_available_phases() -> None:
    """Print phase names, descriptions, and their scripts."""

    print("\nAvailable phases:")
    for name, (description, scripts) in PHASES.items():
        print(f"\n  {name:<14} {description}")
        for script in scripts:
            print(f"    - {script.as_posix()}")


def run_scripts(scripts: Iterable[Path]) -> None:
    """Run project scripts one by one and stop immediately on an error."""

    for relative_path in scripts:
        script = PROJECT_ROOT / relative_path
        if not script.is_file():
            raise FileNotFoundError(f"Required script was not found: {script}")

        print("\n" + "=" * 72)
        print(f"Running: {relative_path.as_posix()}")
        print("=" * 72)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            check=True,
        )


def main() -> None:
    """Select and run the requested project phase."""

    args = parse_arguments()

    if args.list or (not args.phase and not args.all):
        print("Quantum-Inspired Attention project runner")
        print_available_phases()
        print("\nUse: python main.py --phase <name>  or  python main.py --all")
        return

    selected_phases = tuple(PHASES) if args.all else (args.phase,)

    try:
        for phase_name in selected_phases:
            description, scripts = PHASES[phase_name]
            print(f"\nStarting phase: {phase_name} — {description}")
            run_scripts(scripts)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"\nPipeline stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("\n" + "=" * 72)
    print("Requested pipeline completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
