#!/usr/bin/env python3
"""Evaluate POMO checkpoints on fixed TSP100 data-farm instances.

Primary purpose: paired generalization comparison between
  * checkpoint-200 baseline,
  * ordinary uniform checkpoint-300,
  * STD-PGD checkpoint-300.

The script evaluates every model on exactly the same instances and reports
no-augmentation and 8-fold-augmentation optimality gaps, per-instance CSVs,
family summaries, and paired bootstrap confidence intervals versus a chosen
reference model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


FAMILIES = ("uniform", "clustered1", "clustered2", "explosion", "implosion")
OOD_FAMILIES = ("clustered1", "clustered2", "explosion", "implosion")
MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}
ENV_PARAMS = {"problem_size": 100, "pomo_size": 100}


@dataclass(frozen=True)
class FamilyData:
    coordinates: np.ndarray  # (N, 100, 2), float32
    optimum: np.ndarray      # (N,), float64
    optimum_runtime: np.ndarray  # (N,), float64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired generalization evaluation for POMO TSP100 checkpoints."
    )
    parser.add_argument(
        "--pomo-dir",
        type=Path,
        default=Path.home() / "projects/pomo/NEW_py_ver/TSP/POMO",
        help="Directory containing TSPModel.py, TSPEnv.py, and TSProblemDef.py.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/data_farm",
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/solution_farm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for each model, e.g. --checkpoint uniform300=/path/checkpoint-300.pt",
    )
    parser.add_argument(
        "--reference-model",
        default="uniform300",
        help="Model used as the paired-comparison reference.",
    )
    parser.add_argument("--start-row", type=int, default=0, help="Zero-based first row.")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=0,
        help="Rows per family; 0 means all rows from --start-row onward.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument(
        "--solution-tolerance",
        type=float,
        default=1e-3,
        help="Maximum allowed absolute difference between supplied optimum and recomputed tour length.",
    )
    return parser.parse_args()


def parse_checkpoint_specs(specs: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --checkpoint value {spec!r}; expected NAME=PATH")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if not name:
            raise ValueError(f"Empty checkpoint name in {spec!r}")
        if name in result:
            raise ValueError(f"Duplicate checkpoint name: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        result[name] = path
    return result


def parse_coordinate_line(line: str, *, source: Path, line_no: int) -> np.ndarray:
    tokens = line.strip().split()
    if len(tokens) != 100:
        raise ValueError(
            f"{source}:{line_no}: expected 100 coordinate pairs, found {len(tokens)}"
        )
    coords = np.empty((100, 2), dtype=np.float32)
    for index, token in enumerate(tokens):
        try:
            x_text, y_text = token.split(",", 1)
            coords[index, 0] = float(x_text)
            coords[index, 1] = float(y_text)
        except Exception as exc:
            raise ValueError(
                f"{source}:{line_no}: malformed coordinate token {token!r}"
            ) from exc
    if not np.isfinite(coords).all():
        raise ValueError(f"{source}:{line_no}: non-finite coordinate")
    return coords


def tour_length(coords: np.ndarray, tour: Sequence[int]) -> float:
    order = np.asarray(tour, dtype=np.int64)
    selected = coords[order].astype(np.float64, copy=False)
    shifted = np.roll(selected, shift=-1, axis=0)
    return float(np.linalg.norm(selected - shifted, axis=1).sum())


def parse_solution_line(
    line: str,
    coords: np.ndarray,
    *,
    source: Path,
    line_no: int,
    tolerance: float,
) -> Tuple[float, float]:
    parts = line.strip().rsplit(maxsplit=2)
    if len(parts) != 3:
        raise ValueError(
            f"{source}:{line_no}: expected TOUR OPTIMUM RUNTIME, found {line.strip()!r}"
        )
    tour_text, optimum_text, runtime_text = parts
    try:
        tour = [int(token) for token in tour_text.split(",")]
        optimum = float(optimum_text)
        runtime = float(runtime_text)
    except Exception as exc:
        raise ValueError(f"{source}:{line_no}: malformed solution row") from exc

    if len(tour) != 100 or set(tour) != set(range(100)):
        raise ValueError(
            f"{source}:{line_no}: tour must be a permutation of 0..99; got length {len(tour)}"
        )
    recomputed = tour_length(coords, tour)
    difference = abs(recomputed - optimum)
    if difference > tolerance:
        raise ValueError(
            f"{source}:{line_no}: supplied optimum {optimum:.9f} disagrees with "
            f"tour length {recomputed:.9f} by {difference:.3e}"
        )
    return optimum, runtime


def load_family(
    family: str,
    data_dir: Path,
    solution_dir: Path,
    *,
    start_row: int,
    num_instances: int,
    solution_tolerance: float,
) -> FamilyData:
    data_path = data_dir / f"tsp100_{family}.txt"
    solution_path = solution_dir / f"gurobi_tsp100_{family}.txt"
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not solution_path.is_file():
        raise FileNotFoundError(solution_path)

    data_lines = data_path.read_text(encoding="utf-8").splitlines()
    solution_lines = solution_path.read_text(encoding="utf-8").splitlines()
    if len(data_lines) != len(solution_lines):
        raise ValueError(
            f"{family}: data rows ({len(data_lines)}) != solution rows ({len(solution_lines)})"
        )
    if start_row < 0 or start_row >= len(data_lines):
        raise ValueError(
            f"{family}: --start-row={start_row} outside 0..{len(data_lines)-1}"
        )
    stop = len(data_lines) if num_instances == 0 else min(len(data_lines), start_row + num_instances)
    if stop <= start_row:
        raise ValueError(f"{family}: empty selected row range")

    coordinates: List[np.ndarray] = []
    optima: List[float] = []
    runtimes: List[float] = []
    for zero_based in range(start_row, stop):
        line_no = zero_based + 1
        coords = parse_coordinate_line(data_lines[zero_based], source=data_path, line_no=line_no)
        optimum, runtime = parse_solution_line(
            solution_lines[zero_based],
            coords,
            source=solution_path,
            line_no=line_no,
            tolerance=solution_tolerance,
        )
        coordinates.append(coords)
        optima.append(optimum)
        runtimes.append(runtime)

    return FamilyData(
        coordinates=np.stack(coordinates, axis=0),
        optimum=np.asarray(optima, dtype=np.float64),
        optimum_runtime=np.asarray(runtimes, dtype=np.float64),
    )


def configure_torch(device: torch.device, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    # TSPEnv creates tensors without explicit device arguments. Setting the
    # default device keeps those tensors colocated with the model and inputs.
    if hasattr(torch, "set_default_device"):
        torch.set_default_device(device)
    elif device.type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)


def load_checkpoint(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def inject_problems(env, problems: torch.Tensor, aug_factor: int, augment_fn) -> None:
    if problems.ndim != 3 or tuple(problems.shape[1:]) != (100, 2):
        raise ValueError(f"Expected problems shaped (B,100,2), got {tuple(problems.shape)}")
    base_batch = int(problems.shape[0])
    if aug_factor == 1:
        augmented = problems
    elif aug_factor == 8:
        augmented = augment_fn(problems)
    else:
        raise ValueError(f"Unsupported augmentation factor: {aug_factor}")

    env.batch_size = base_batch * aug_factor
    env.problems = augmented
    device = augmented.device
    env.BATCH_IDX = torch.arange(env.batch_size, device=device)[:, None].expand(
        env.batch_size, env.pomo_size
    )
    env.POMO_IDX = torch.arange(env.pomo_size, device=device)[None, :].expand(
        env.batch_size, env.pomo_size
    )


def rollout_batch(model, env, problems: torch.Tensor, augment_fn) -> Tuple[np.ndarray, np.ndarray]:
    base_batch = int(problems.shape[0])
    aug_factor = 8
    model.eval()
    with torch.inference_mode():
        inject_problems(env, problems, aug_factor=aug_factor, augment_fn=augment_fn)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

        aug_reward = reward.reshape(aug_factor, base_batch, env.pomo_size)
        max_pomo_reward = aug_reward.max(dim=2).values
        no_aug_length = -max_pomo_reward[0]
        aug8_length = -max_pomo_reward.max(dim=0).values

    return (
        no_aug_length.detach().cpu().numpy().astype(np.float64, copy=False),
        aug8_length.detach().cpu().numpy().astype(np.float64, copy=False),
    )


def evaluate_model(
    model_name: str,
    checkpoint_path: Path,
    families: Mapping[str, FamilyData],
    *,
    batch_size: int,
    device: torch.device,
    Model,
    Env,
    augment_fn,
) -> Dict[str, Dict[str, np.ndarray]]:
    print(f"\n=== Loading {model_name}: {checkpoint_path} ===", flush=True)
    model = Model(**MODEL_PARAMS)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path}: missing model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    env = Env(**ENV_PARAMS)

    model_results: Dict[str, Dict[str, np.ndarray]] = {}
    for family_name, family_data in families.items():
        n = len(family_data.optimum)
        noaug_parts: List[np.ndarray] = []
        aug8_parts: List[np.ndarray] = []
        start_time = time.perf_counter()
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            problems = torch.as_tensor(
                family_data.coordinates[start:stop], dtype=torch.float32, device=device
            )
            noaug, aug8 = rollout_batch(model, env, problems, augment_fn)
            noaug_parts.append(noaug)
            aug8_parts.append(aug8)
            print(
                f"{model_name:>12s} | {family_name:>10s} | {stop:4d}/{n:4d}",
                end="\r",
                flush=True,
            )
        elapsed = time.perf_counter() - start_time
        print(
            f"{model_name:>12s} | {family_name:>10s} | {n:4d}/{n:4d} | {elapsed:7.2f}s",
            flush=True,
        )
        model_results[family_name] = {
            "length_noaug": np.concatenate(noaug_parts),
            "length_aug8": np.concatenate(aug8_parts),
        }

    del env
    del model
    del checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model_results


def optimality_gap_pct(length: np.ndarray, optimum: np.ndarray) -> np.ndarray:
    return 100.0 * (length - optimum) / optimum


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap_mean_ci requires a nonempty 1-D array")
    if values.size == 1 or reps <= 0:
        value = float(values.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    # Chunk the bootstrap to avoid allocating reps x N at once for large banks.
    chunk = 2000
    written = 0
    while written < reps:
        current = min(chunk, reps - written)
        indices = rng.integers(0, values.size, size=(current, values.size))
        means[written : written + current] = values[indices].mean(axis=1)
        written += current
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def stratified_ood_macro_bootstrap_ci(
    family_deltas: Mapping[str, np.ndarray],
    *,
    reps: int,
    seed: int,
) -> Tuple[float, float]:
    if reps <= 0:
        macro = float(np.mean([np.mean(family_deltas[name]) for name in OOD_FAMILIES]))
        return macro, macro
    rng = np.random.default_rng(seed)
    bootstrap_macro = np.zeros(reps, dtype=np.float64)
    for family in OOD_FAMILIES:
        values = np.asarray(family_deltas[family], dtype=np.float64)
        indices = rng.integers(0, values.size, size=(reps, values.size))
        bootstrap_macro += values[indices].mean(axis=1) / len(OOD_FAMILIES)
    low, high = np.percentile(bootstrap_macro, [2.5, 97.5])
    return float(low), float(high)


def write_per_instance_csv(
    path: Path,
    families: Mapping[str, FamilyData],
    results: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    start_row: int,
) -> None:
    model_names = list(results.keys())
    fieldnames = ["family", "source_row_1based", "optimum", "optimum_runtime"]
    for model in model_names:
        fieldnames.extend(
            [
                f"{model}_length_noaug",
                f"{model}_gap_noaug_pct",
                f"{model}_length_aug8",
                f"{model}_gap_aug8_pct",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family_name, family_data in families.items():
            for index in range(len(family_data.optimum)):
                row = {
                    "family": family_name,
                    "source_row_1based": start_row + index + 1,
                    "optimum": f"{family_data.optimum[index]:.12f}",
                    "optimum_runtime": f"{family_data.optimum_runtime[index]:.12f}",
                }
                for model in model_names:
                    noaug = results[model][family_name]["length_noaug"][index]
                    aug8 = results[model][family_name]["length_aug8"][index]
                    optimum = family_data.optimum[index]
                    row[f"{model}_length_noaug"] = f"{noaug:.12f}"
                    row[f"{model}_gap_noaug_pct"] = f"{100.0*(noaug-optimum)/optimum:.12f}"
                    row[f"{model}_length_aug8"] = f"{aug8:.12f}"
                    row[f"{model}_gap_aug8_pct"] = f"{100.0*(aug8-optimum)/optimum:.12f}"
                writer.writerow(row)


def build_summary_rows(
    families: Mapping[str, FamilyData],
    results: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name, model_results in results.items():
        family_means: Dict[str, Dict[str, float]] = {}
        for family_name, family_data in families.items():
            family_means[family_name] = {}
            for metric in ("noaug", "aug8"):
                length = model_results[family_name][f"length_{metric}"]
                gap = optimality_gap_pct(length, family_data.optimum)
                family_means[family_name][metric] = float(gap.mean())
                rows.append(
                    {
                        "model": model_name,
                        "family": family_name,
                        "metric": metric,
                        "n": int(gap.size),
                        "mean_length": float(length.mean()),
                        "mean_gap_pct": float(gap.mean()),
                        "std_gap_pct": float(gap.std(ddof=1)) if gap.size > 1 else 0.0,
                        "median_gap_pct": float(np.median(gap)),
                        "p90_gap_pct": float(np.percentile(gap, 90)),
                    }
                )
        for metric in ("noaug", "aug8"):
            macro = float(np.mean([family_means[name][metric] for name in OOD_FAMILIES]))
            rows.append(
                {
                    "model": model_name,
                    "family": "OOD_macro",
                    "metric": metric,
                    "n": sum(len(families[name].optimum) for name in OOD_FAMILIES),
                    "mean_length": math.nan,
                    "mean_gap_pct": macro,
                    "std_gap_pct": math.nan,
                    "median_gap_pct": math.nan,
                    "p90_gap_pct": math.nan,
                }
            )
    return rows


def write_dict_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_paired_rows(
    families: Mapping[str, FamilyData],
    results: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    reference_model: str,
    bootstrap_reps: int,
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    tolerance = 1e-9
    for model_index, (model_name, model_results) in enumerate(results.items()):
        if model_name == reference_model:
            continue
        for metric_index, metric in enumerate(("noaug", "aug8")):
            family_deltas: Dict[str, np.ndarray] = {}
            for family_index, (family_name, family_data) in enumerate(families.items()):
                reference_gap = optimality_gap_pct(
                    results[reference_model][family_name][f"length_{metric}"],
                    family_data.optimum,
                )
                candidate_gap = optimality_gap_pct(
                    model_results[family_name][f"length_{metric}"],
                    family_data.optimum,
                )
                delta = candidate_gap - reference_gap
                family_deltas[family_name] = delta
                ci_low, ci_high = bootstrap_mean_ci(
                    delta,
                    reps=bootstrap_reps,
                    seed=seed + 10000 * model_index + 1000 * metric_index + family_index,
                )
                rows.append(
                    {
                        "candidate_model": model_name,
                        "reference_model": reference_model,
                        "family": family_name,
                        "metric": metric,
                        "n": int(delta.size),
                        "mean_delta_gap_pp": float(delta.mean()),
                        "ci95_low_pp": ci_low,
                        "ci95_high_pp": ci_high,
                        "wins": int(np.sum(delta < -tolerance)),
                        "ties": int(np.sum(np.abs(delta) <= tolerance)),
                        "losses": int(np.sum(delta > tolerance)),
                    }
                )
            macro_mean = float(np.mean([family_deltas[name].mean() for name in OOD_FAMILIES]))
            macro_low, macro_high = stratified_ood_macro_bootstrap_ci(
                family_deltas,
                reps=bootstrap_reps,
                seed=seed + 900000 + 10000 * model_index + metric_index,
            )
            rows.append(
                {
                    "candidate_model": model_name,
                    "reference_model": reference_model,
                    "family": "OOD_macro",
                    "metric": metric,
                    "n": sum(len(family_deltas[name]) for name in OOD_FAMILIES),
                    "mean_delta_gap_pp": macro_mean,
                    "ci95_low_pp": macro_low,
                    "ci95_high_pp": macro_high,
                    "wins": sum(int(np.sum(family_deltas[name] < -tolerance)) for name in OOD_FAMILIES),
                    "ties": sum(int(np.sum(np.abs(family_deltas[name]) <= tolerance)) for name in OOD_FAMILIES),
                    "losses": sum(int(np.sum(family_deltas[name] > tolerance)) for name in OOD_FAMILIES),
                }
            )
    return rows


def print_primary_table(summary_rows: Sequence[Mapping[str, object]]) -> None:
    primary = {
        (str(row["model"]), str(row["family"])): float(row["mean_gap_pct"])
        for row in summary_rows
        if row["metric"] == "aug8"
    }
    models = []
    for row in summary_rows:
        model = str(row["model"])
        if model not in models:
            models.append(model)
    columns = list(FAMILIES) + ["OOD_macro"]
    print("\nPrimary result: mean optimality gap (%) with 8-fold augmentation")
    header = "model".ljust(14) + "".join(name.rjust(14) for name in columns)
    print(header)
    print("-" * len(header))
    for model in models:
        line = model.ljust(14)
        for family in columns:
            line += f"{primary[(model, family)]:14.6f}"
        print(line)


def print_paired_table(paired_rows: Sequence[Mapping[str, object]], candidate: str) -> None:
    selected = [
        row
        for row in paired_rows
        if row["candidate_model"] == candidate and row["metric"] == "aug8"
    ]
    if not selected:
        return
    print(f"\nPaired aug8 gap difference: {candidate} - reference (negative is better)")
    print("family".ljust(14) + "mean(pp)".rjust(12) + "95% CI".rjust(28) + "  W/T/L")
    print("-" * 72)
    for row in selected:
        ci = f"[{float(row['ci95_low_pp']):.6f}, {float(row['ci95_high_pp']):.6f}]"
        wtl = f"{row['wins']}/{row['ties']}/{row['losses']}"
        print(
            str(row["family"]).ljust(14)
            + f"{float(row['mean_delta_gap_pp']):12.6f}"
            + ci.rjust(28)
            + "  "
            + wtl
        )


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoint_specs(args.checkpoint)
    if args.reference_model not in checkpoints:
        raise ValueError(
            f"--reference-model={args.reference_model!r} not in checkpoints {list(checkpoints)}"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_instances < 0:
        raise ValueError("--num-instances must be nonnegative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    configure_torch(device, args.seed)

    pomo_dir = args.pomo_dir.expanduser().resolve()
    if not pomo_dir.is_dir():
        raise FileNotFoundError(pomo_dir)
    sys.path.insert(0, str(pomo_dir))
    from TSPEnv import TSPEnv as Env  # type: ignore
    from TSPModel import TSPModel as Model  # type: ignore
    from TSProblemDef import augment_xy_data_by_8_fold  # type: ignore

    print("Loading fixed evaluation banks and validating supplied optimum tours...", flush=True)
    families: Dict[str, FamilyData] = {}
    for family in FAMILIES:
        family_data = load_family(
            family,
            args.data_dir.expanduser().resolve(),
            args.solution_dir.expanduser().resolve(),
            start_row=args.start_row,
            num_instances=args.num_instances,
            solution_tolerance=args.solution_tolerance,
        )
        families[family] = family_data
        print(
            f"  {family:>10s}: {len(family_data.optimum)} instances, "
            f"rows {args.start_row + 1}..{args.start_row + len(family_data.optimum)}",
            flush=True,
        )

    results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    wall_start = time.perf_counter()
    for model_name, checkpoint_path in checkpoints.items():
        results[model_name] = evaluate_model(
            model_name,
            checkpoint_path,
            families,
            batch_size=args.batch_size,
            device=device,
            Model=Model,
            Env=Env,
            augment_fn=augment_xy_data_by_8_fold,
        )

    per_instance_path = args.output_dir / "per_instance.csv"
    summary_path = args.output_dir / "summary.csv"
    paired_path = args.output_dir / "paired_vs_reference.csv"
    metadata_path = args.output_dir / "metadata.json"

    write_per_instance_csv(
        per_instance_path,
        families,
        results,
        start_row=args.start_row,
    )
    summary_rows = build_summary_rows(families, results)
    paired_rows = build_paired_rows(
        families,
        results,
        reference_model=args.reference_model,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    write_dict_rows(summary_path, summary_rows)
    write_dict_rows(paired_path, paired_rows)

    metadata = {
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "reference_model": args.reference_model,
        "families": list(FAMILIES),
        "ood_families": list(OOD_FAMILIES),
        "start_row_zero_based": args.start_row,
        "num_instances_per_family": {
            name: len(data.optimum) for name, data in families.items()
        },
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
        "bootstrap_reps": args.bootstrap_reps,
        "primary_metric": "aug8 optimality gap (%)",
        "elapsed_seconds": time.perf_counter() - wall_start,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print_primary_table(summary_rows)
    for candidate in checkpoints:
        if candidate != args.reference_model:
            print_paired_table(paired_rows, candidate)

    print("\nSaved:")
    for path in (per_instance_path, summary_path, paired_path, metadata_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
