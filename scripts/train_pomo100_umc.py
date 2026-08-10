#!/usr/bin/env python3
"""Batch-wise max-policy-gradient continuation training for POMO TSP100.

At every optimizer update:
  1. Generate M=8 independent candidate batches, each containing 64 uniform TSP100 instances.
  2. Score each complete batch by the L2 norm of the standard POMO policy gradient.
  3. Select the batch with the largest norm.
  4. Reuse the winning scoring gradient directly for optimizer.step().

The script restores model, Adam, and scheduler states from the source POMO checkpoint.
It does not override the checkpoint learning rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rbf_soft_clustered_generator import generate_rbf_soft_contracted_batch


def configure_pomo_imports(pomo_tsp_root: Path) -> None:
    pomo_tsp_root = pomo_tsp_root.resolve()
    for path in (str(pomo_tsp_root / "POMO"), str(pomo_tsp_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


ENV_PARAMS = {"problem_size": 100, "pomo_size": 100}
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

# Constructor values are overwritten by the restored optimizer/scheduler states.
OPTIMIZER_CONSTRUCTOR_PARAMS = {"lr": 1e-4, "weight_decay": 1e-6}
SCHEDULER_CONSTRUCTOR_PARAMS = {"milestones": [3001], "gamma": 0.1}


@dataclass
class RunConfig:
    mode: str
    source_checkpoint: str
    output_dir: str
    pomo_tsp_root: str
    schedule_origin_epoch: int
    cycle_pattern: str
    additional_epochs: int
    train_episodes: int
    train_batch_size: int
    candidate_batches: int
    cluster_k_min: int
    cluster_k_max: int
    cluster_alpha: float
    rbf_rho: float
    cluster_seed_offset: int
    save_interval_epochs: int
    log_interval_updates: int
    seed: int
    expected_source_epoch: int
    expected_initial_lr: float
    device: str


def torch_load_full(path: Path, map_location: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mixed_maxgrad_rbf_soft_uniform")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output_dir / "train.log", mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(
    data_generator: torch.Generator,
    cluster_generator: torch.Generator,
) -> Dict:
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "data_generator_state": data_generator.get_state(),
        "cluster_generator_state": cluster_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    checkpoint: Dict,
    data_generator: torch.Generator,
    cluster_generator: torch.Generator,
    *,
    require_cluster_state: bool,
) -> None:
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_cpu_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
    if "torch_cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])
    if "data_generator_state" in checkpoint:
        data_generator.set_state(checkpoint["data_generator_state"])
    if "cluster_generator_state" in checkpoint:
        cluster_generator.set_state(checkpoint["cluster_generator_state"])
    elif require_cluster_state:
        raise KeyError(
            "Resume checkpoint is missing cluster_generator_state; exact "
            "cluster-generator replay is impossible."
        )


def set_rollout_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def generate_uniform_candidate_pool(
    candidate_batches: int,
    batch_size: int,
    problem_size: int,
    data_generator: torch.Generator,
) -> torch.Tensor:
    return torch.rand(
        (candidate_batches, batch_size, problem_size, 2),
        generator=data_generator,
        device="cpu",
        dtype=torch.float32,
    )


def load_problems_into_env(env, problems: torch.Tensor) -> None:
    batch_size = int(problems.size(0))
    env.batch_size = batch_size
    env.problems = problems
    env.BATCH_IDX = torch.arange(batch_size, device=problems.device)[:, None].expand(
        batch_size, env.pomo_size
    )
    env.POMO_IDX = torch.arange(env.pomo_size, device=problems.device)[None, :].expand(
        batch_size, env.pomo_size
    )


def pomo_forward_loss(model, env, problems: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    model.train()
    load_problems_into_env(env, problems)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    batch_size = int(problems.size(0))
    prob_list = torch.zeros((batch_size, env.pomo_size, 0), device=problems.device)
    state, reward, done = env.pre_step()
    while not done:
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

    advantage = reward - reward.float().mean(dim=1, keepdim=True)
    log_prob = prob_list.log().sum(dim=2)
    loss_mean = (-advantage * log_prob).mean()
    max_pomo_reward, _ = reward.max(dim=1)
    score_mean = -max_pomo_reward.float().mean()
    return loss_mean, score_mean


def gradient_l2_norm(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().pow(2).sum()
        squared = value if squared is None else squared + value
    if squared is None:
        raise RuntimeError("No gradients were produced.")
    return squared.sqrt()


def clone_gradients(parameters: Sequence[torch.nn.Parameter]) -> List[Optional[torch.Tensor]]:
    return [None if p.grad is None else p.grad.detach().clone() for p in parameters]


def restore_gradients(
    parameters: Sequence[torch.nn.Parameter], gradients: Sequence[Optional[torch.Tensor]]
) -> None:
    if len(parameters) != len(gradients):
        raise ValueError("Parameter/gradient length mismatch.")
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient


def norm_statistics(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=0))
    maximum = float(array.max())
    return {
        "norm_min": float(array.min()),
        "norm_q25": float(np.quantile(array, 0.25)),
        "norm_median": float(np.quantile(array, 0.50)),
        "norm_q75": float(np.quantile(array, 0.75)),
        "norm_max": maximum,
        "norm_mean": mean,
        "norm_std": std,
        "norm_max_over_mean": maximum / max(mean, 1e-12),
        "norm_max_z": (maximum - mean) / max(std, 1e-12),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    global_update: int,
    model,
    optimizer,
    scheduler,
    data_generator: torch.Generator,
    cluster_generator: torch.Generator,
    config: RunConfig,
) -> None:
    payload = {
        "epoch": epoch,
        "global_update": global_update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(config),
        **capture_rng_state(data_generator, cluster_generator),
    }
    torch.save(payload, path)


def append_history_row(csv_path: Path, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def score_batch_gradient(
    model,
    env,
    optimizer,
    parameters: Sequence[torch.nn.Parameter],
    problems: torch.Tensor,
    rollout_seed: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Leave this candidate's gradients attached to the model parameters."""
    set_rollout_seed(rollout_seed, device)
    optimizer.zero_grad(set_to_none=True)
    loss, score = pomo_forward_loss(model, env, problems)
    loss.backward()
    norm = gradient_l2_norm(parameters)
    return float(norm.item()), float(loss.item()), float(score.item())


def train_one_update_rankgrad(
    model,
    env,
    optimizer,
    parameters: Sequence[torch.nn.Parameter],
    candidate_pool_cpu: torch.Tensor,
    rollout_seed: int,
    device: torch.device,
    selection_rank_from_largest: int,
) -> Dict[str, object]:
    """Score every candidate and train on the requested descending norm rank.

    Only the current top-k candidate gradients are retained, where
    k=selection_rank_from_largest. Thus selecting the third-largest gradient
    requires at most three saved gradient copies and no ninth backward pass.
    """
    candidate_count = int(candidate_pool_cpu.size(0))
    if not 1 <= selection_rank_from_largest <= candidate_count:
        raise ValueError(
            "selection_rank_from_largest must be in "
            f"[1, {candidate_count}], got {selection_rank_from_largest}."
        )

    norms: List[float] = []
    # Tuples: (norm, candidate_index, loss, score, cloned_gradients).
    # The list is kept in descending norm order; ties prefer the lower index.
    top_candidates: List[
        Tuple[
            float,
            int,
            float,
            float,
            List[Optional[torch.Tensor]],
        ]
    ] = []

    start = time.perf_counter()
    for candidate_index in range(candidate_count):
        problems = candidate_pool_cpu[candidate_index].to(device, non_blocking=True)
        norm, loss, score = score_batch_gradient(
            model, env, optimizer, parameters, problems, rollout_seed, device
        )
        norms.append(norm)

        enters_top_k = (
            len(top_candidates) < selection_rank_from_largest
            or norm > top_candidates[-1][0]
            or (
                norm == top_candidates[-1][0]
                and candidate_index < top_candidates[-1][1]
            )
        )
        if enters_top_k:
            record = (
                norm,
                candidate_index,
                loss,
                score,
                clone_gradients(parameters),
            )
            top_candidates.append(record)
            top_candidates.sort(key=lambda item: (-item[0], item[1]))
            if len(top_candidates) > selection_rank_from_largest:
                top_candidates.pop()

    if len(top_candidates) != selection_rank_from_largest:
        raise RuntimeError(
            "Failed to retain enough ranked candidate gradients: "
            f"kept={len(top_candidates)}, requested={selection_rank_from_largest}."
        )

    selected_norm, selected_index, selected_loss, selected_score, selected_gradients = (
        top_candidates[selection_rank_from_largest - 1]
    )

    # Full descending order is saved for auditability. This does not affect RNG.
    descending_indices = sorted(
        range(candidate_count),
        key=lambda index: (-norms[index], index),
    )
    expected_selected_index = descending_indices[selection_rank_from_largest - 1]
    if selected_index != expected_selected_index:
        raise RuntimeError(
            "Internal rank-selection mismatch: "
            f"retained={selected_index}, sorted={expected_selected_index}."
        )

    optimizer.zero_grad(set_to_none=True)
    restore_gradients(parameters, selected_gradients)
    optimizer.step()

    stats = norm_statistics(norms)
    selected_rank_from_smallest = candidate_count - selection_rank_from_largest + 1
    selected_z = (
        (selected_norm - stats["norm_mean"]) / max(stats["norm_std"], 1e-12)
        if stats["norm_std"] > 0
        else 0.0
    )
    return {
        **stats,
        "selected_index": selected_index,
        "selected_rank_from_largest": selection_rank_from_largest,
        "selected_rank_from_smallest": selected_rank_from_smallest,
        "selected_norm": selected_norm,
        "selected_norm_z": selected_z,
        "selected_loss": selected_loss,
        "selected_score": selected_score,
        "candidate_norms_json": json.dumps(norms, separators=(",", ":")),
        "candidate_order_desc_json": json.dumps(
            descending_indices, separators=(",", ":")
        ),
        "scoring_and_step_seconds": time.perf_counter() - start,
    }

def train_one_update_uniform(
    model,
    env,
    optimizer,
    parameters: Sequence[torch.nn.Parameter],
    candidate_pool_cpu: torch.Tensor,
    rollout_seed: int,
    device: torch.device,
) -> Dict[str, object]:
    start = time.perf_counter()
    problems = candidate_pool_cpu[0].to(device, non_blocking=True)
    norm, loss, score = score_batch_gradient(
        model, env, optimizer, parameters, problems, rollout_seed, device
    )
    # The current gradients are already the exact gradients to train on.
    optimizer.step()
    return {
        "norm_min": norm,
        "norm_q25": norm,
        "norm_median": norm,
        "norm_q75": norm,
        "norm_max": norm,
        "norm_mean": norm,
        "norm_std": 0.0,
        "norm_max_over_mean": 1.0,
        "norm_max_z": 0.0,
        "selected_index": 0,
        "selected_rank_from_largest": 1,
        "selected_rank_from_smallest": 1,
        "selected_norm": norm,
        "selected_norm_z": 0.0,
        "selected_loss": loss,
        "selected_score": score,
        "candidate_norms_json": json.dumps([norm], separators=(",", ":")),
        "candidate_order_desc_json": json.dumps([0], separators=(",", ":")),
        "scoring_and_step_seconds": time.perf_counter() - start,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pomo-tsp-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--additional-epochs", type=int, default=50)
    parser.add_argument("--train-episodes", type=int, default=100_000)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--candidate-batches", type=int, default=8)
    parser.add_argument(
        "--cycle-pattern",
        default="maxgrad,maxgrad,clustered,uniform,uniform",
        help=(
            "Comma-separated epoch types. Supported values: maxgrad, "
            "clustered, uniform."
        ),
    )
    parser.add_argument("--cluster-k-min", type=int, default=2)
    parser.add_argument("--cluster-k-max", type=int, default=10)
    parser.add_argument("--cluster-alpha", type=float, default=0.8)
    parser.add_argument("--rbf-rho", type=float, default=0.35)
    parser.add_argument("--cluster-seed-offset", type=int, default=2_000_003)
    parser.add_argument("--save-interval-epochs", type=int, default=10)
    parser.add_argument("--log-interval-updates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--expected-source-epoch", type=int, default=2000)
    parser.add_argument("--expected-initial-lr", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_batches < 1:
        raise ValueError("--candidate-batches must be positive.")
    if args.additional_epochs < 1:
        raise ValueError("--additional-epochs must be positive.")
    if not 1 <= args.cluster_k_min <= args.cluster_k_max <= ENV_PARAMS["problem_size"]:
        raise ValueError(
            "Require 1 <= cluster-k-min <= cluster-k-max <= problem_size."
        )
    if not 0.0 <= args.cluster_alpha <= 1.0:
        raise ValueError("--cluster-alpha must lie in [0,1].")
    if args.rbf_rho <= 0.0:
        raise ValueError("--rbf-rho must be positive.")

    cycle_pattern = tuple(
        token.strip().lower()
        for token in args.cycle_pattern.split(",")
        if token.strip()
    )
    supported_modes = {"maxgrad", "clustered", "uniform"}
    if not cycle_pattern:
        raise ValueError("--cycle-pattern must contain at least one epoch type.")
    unknown_modes = set(cycle_pattern).difference(supported_modes)
    if unknown_modes:
        raise ValueError(
            f"Unknown cycle-pattern entries: {sorted(unknown_modes)}."
        )

    configure_pomo_imports(args.pomo_tsp_root)

    from TSPEnv import TSPEnv as Env
    from TSPModel import TSPModel as Model
    from torch.optim import Adam as Optimizer
    from torch.optim.lr_scheduler import MultiStepLR as Scheduler

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_dir = args.output_dir.resolve()
    logger = setup_logging(output_dir)
    seed_everything(args.seed)

    # Original POMO code creates many tensors without explicit device arguments.
    if device.type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    model = Model(**MODEL_PARAMS).to(device)
    env = Env(**ENV_PARAMS)
    optimizer = Optimizer(model.parameters(), **OPTIMIZER_CONSTRUCTOR_PARAMS)
    scheduler = Scheduler(optimizer, **SCHEDULER_CONSTRUCTOR_PARAMS)
    parameters = list(model.parameters())

    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(args.seed + 1_000_003)

    # Independent from uniform data generation and CUDA rollout randomness.
    cluster_generator = torch.Generator(device="cpu")
    cluster_generator.manual_seed(args.seed + args.cluster_seed_offset)

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
    }

    # The mixed schedule must always be anchored to the original source epoch,
    # including when resuming from an intermediate mixed checkpoint.
    if args.resume is None:
        checkpoint_path = args.source_checkpoint
        checkpoint = torch_load_full(checkpoint_path, map_location=device)
        source_epoch = int(checkpoint["epoch"])
    else:
        source_metadata = torch_load_full(
            args.source_checkpoint, map_location=torch.device("cpu")
        )
        if "epoch" not in source_metadata:
            raise KeyError("Source checkpoint is missing key: epoch")
        source_epoch = int(source_metadata["epoch"])
        checkpoint_path = args.resume
        checkpoint = torch_load_full(checkpoint_path, map_location=device)

    missing = required_keys.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing)}")
    if source_epoch != args.expected_source_epoch:
        raise RuntimeError(
            f"Source checkpoint epoch is {source_epoch}, expected "
            f"{args.expected_source_epoch}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    loaded_epoch = int(checkpoint["epoch"])
    global_update = int(checkpoint.get("global_update", 0))
    if loaded_epoch < source_epoch:
        raise RuntimeError(
            f"Loaded epoch {loaded_epoch} precedes source epoch {source_epoch}."
        )

    if args.resume is not None:
        restore_rng_state(
            checkpoint,
            data_generator,
            cluster_generator,
            require_cluster_state=True,
        )
        logger.info("Resuming MaxGrad/RBF-soft mixed checkpoint: %s", checkpoint_path)
    else:
        logger.info("Starting from source POMO checkpoint: %s", checkpoint_path)

    config = RunConfig(
        mode="mixed_maxgrad_rbf_soft_uniform",
        source_checkpoint=str(args.source_checkpoint.resolve()),
        output_dir=str(output_dir),
        pomo_tsp_root=str(args.pomo_tsp_root.resolve()),
        schedule_origin_epoch=source_epoch,
        cycle_pattern=",".join(cycle_pattern),
        additional_epochs=args.additional_epochs,
        train_episodes=args.train_episodes,
        train_batch_size=args.train_batch_size,
        candidate_batches=args.candidate_batches,
        cluster_k_min=args.cluster_k_min,
        cluster_k_max=args.cluster_k_max,
        cluster_alpha=args.cluster_alpha,
        rbf_rho=args.rbf_rho,
        cluster_seed_offset=args.cluster_seed_offset,
        save_interval_epochs=args.save_interval_epochs,
        log_interval_updates=args.log_interval_updates,
        seed=args.seed,
        expected_source_epoch=args.expected_source_epoch,
        expected_initial_lr=args.expected_initial_lr,
        device=str(device),
    )
    with (output_dir / "config.json").open("w") as handle:
        json.dump(asdict(config), handle, indent=2)

    start_epoch = loaded_epoch + 1
    end_epoch = loaded_epoch + args.additional_epochs
    loaded_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    logger.info("Loaded epoch=%d; continuation epochs=%d..%d", loaded_epoch, start_epoch, end_epoch)
    logger.info(
        "Restored optimizer LR(s)=%s; scheduler.last_epoch=%s",
        loaded_lrs,
        scheduler.last_epoch,
    )

    if args.expected_initial_lr > 0:
        for lr in loaded_lrs:
            if not math.isclose(lr, args.expected_initial_lr, rel_tol=1e-6, abs_tol=1e-12):
                raise RuntimeError(
                    f"Checkpoint LR is {lr}, expected {args.expected_initial_lr}. "
                    "The script will not override it. Verify the checkpoint first."
                )

    logger.info(
        "mode=mixed_maxgrad_rbf_soft_uniform | candidate_batches=%d | "
        "cycle_pattern=%s | train_batch_size=%d | train_episodes=%d",
        args.candidate_batches,
        ",".join(cycle_pattern),
        args.train_batch_size,
        args.train_episodes,
    )
    logger.info(
        "RBF-soft generator: K~Uniform{%d,...,%d}, one shared K per full "
        "batch; alpha=%.6f; rho=%.6f; centers sampled independently "
        "from each instance.",
        args.cluster_k_min,
        args.cluster_k_max,
        args.cluster_alpha,
        args.rbf_rho,
    )
    logger.info(
        "Data alignment: every full update generates all %d uniform candidates; "
        "only MaxGrad epochs score all candidates.",
        args.candidate_batches,
    )

    history_path = output_dir / "training_history.csv"
    history_fields = [
        "epoch",
        "continuation_epoch_index",
        "cycle_index",
        "cycle_position",
        "epoch_mode",
        "update_in_epoch",
        "global_update",
        "selected_instances_in_epoch",
        "candidate_batch_size",
        "configured_candidate_batches",
        "effective_candidate_batches",
        "scored_candidate_batches",
        "cluster_k",
        "cluster_alpha",
        "rbf_rho",
        "rbf_sigma_min",
        "rbf_sigma_max",
        "rbf_sigma_mean",
        "rbf_sigma_std",
        "center_nn_median_mean",
        "hard_cluster_size_min",
        "hard_cluster_size_max",
        "hard_cluster_size_mean",
        "hard_cluster_size_std",
        "hard_cluster_empty_count",
        "soft_assignment_entropy_mean",
        "soft_assignment_max_weight_mean",
        "displacement_mean",
        "displacement_max",
        "rbf_sigma_floor_hits",
        "selection_rule",
        "mode",
        "lr",
        "scheduler_last_epoch",
        "rollout_seed",
        "selected_index",
        "selected_rank_from_largest",
        "selected_rank_from_smallest",
        "selected_norm",
        "selected_norm_z",
        "selected_loss",
        "selected_score",
        "norm_min",
        "norm_q25",
        "norm_median",
        "norm_q75",
        "norm_max",
        "norm_mean",
        "norm_std",
        "norm_max_over_mean",
        "norm_max_z",
        "candidate_norms_json",
        "candidate_order_desc_json",
        "scoring_and_step_seconds",
    ]

    train_start = time.perf_counter()
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.perf_counter()

        continuation_epoch_index = epoch - source_epoch
        cycle_length = len(cycle_pattern)
        cycle_index = (continuation_epoch_index - 1) // cycle_length + 1
        cycle_position = (continuation_epoch_index - 1) % cycle_length + 1
        epoch_mode = cycle_pattern[cycle_position - 1]

        # Preserve original TSPTrainer.run() ordering.
        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        selected_instances = 0
        update_in_epoch = 0
        epoch_scores: List[float] = []
        epoch_losses: List[float] = []
        epoch_norms: List[float] = []

        logger.info(
            "===== epoch=%d/%d | continuation=%d | cycle=%d position=%d/%d "
            "| epoch_mode=%s | lr=%.12g | scheduler.last_epoch=%s =====",
            epoch,
            end_epoch,
            continuation_epoch_index,
            cycle_index,
            cycle_position,
            cycle_length,
            epoch_mode,
            current_lr,
            scheduler.last_epoch,
        )

        while selected_instances < args.train_episodes:
            remaining = args.train_episodes - selected_instances
            batch_size = min(args.train_batch_size, remaining)
            update_in_epoch += 1
            global_update += 1

            # The original POMO epoch contains 1,562 full 64-instance updates
            # followed by one final 32-instance update.
            #
            # IMPORTANT FOR ALIGNMENT:
            # Every full update generates all M candidates, including in a
            # uniform epoch. Uniform epochs only evaluate candidate 0. Thus the
            # independent CPU generator advances exactly as it did in the older
            # aligned Uniform / MaxGrad / Third-largest experiments.
            #
            # The final 32-instance remainder generates one batch and is trained
            # normally, with no MaxGrad selection.
            is_remainder_update = batch_size < args.train_batch_size
            effective_candidate_batches = (
                1 if is_remainder_update else args.candidate_batches
            )

            candidate_pool_cpu = generate_uniform_candidate_pool(
                effective_candidate_batches,
                batch_size,
                ENV_PARAMS["problem_size"],
                data_generator,
            )

            # Common random numbers: all candidates scored in a full MaxGrad
            # update share the same stochastic-rollout seed. The same formula was
            # used in the preceding aligned experiments.
            rollout_seed = args.seed + 10_000_019 * global_update

            cluster_stats = {
                "cluster_k": "",
                "cluster_alpha": "",
                "rbf_rho": "",
                "rbf_sigma_min": "",
                "rbf_sigma_max": "",
                "rbf_sigma_mean": "",
                "rbf_sigma_std": "",
                "center_nn_median_mean": "",
                "hard_cluster_size_min": "",
                "hard_cluster_size_max": "",
                "hard_cluster_size_mean": "",
                "hard_cluster_size_std": "",
                "hard_cluster_empty_count": "",
                "soft_assignment_entropy_mean": "",
                "soft_assignment_max_weight_mean": "",
                "displacement_mean": "",
                "displacement_max": "",
                "rbf_sigma_floor_hits": "",
            }

            if is_remainder_update:
                # Always ordinary uniform continuation, including at the end of
                # clustered and MaxGrad epochs.
                selection_rule = "random_remainder"
                scored_candidate_batches = 1
                stats = train_one_update_uniform(
                    model,
                    env,
                    optimizer,
                    parameters,
                    candidate_pool_cpu,
                    rollout_seed,
                    device,
                )
            elif epoch_mode == "maxgrad":
                # Compare only the M aligned uniform candidate batches.
                selection_rule = "maxgrad_uniform_candidates"
                scored_candidate_batches = args.candidate_batches
                stats = train_one_update_rankgrad(
                    model,
                    env,
                    optimizer,
                    parameters,
                    candidate_pool_cpu,
                    rollout_seed,
                    device,
                    selection_rank_from_largest=1,
                )
            elif epoch_mode == "clustered":
                # Exactly one K is sampled for this complete batch and shared
                # by all 64 instances. Candidate 0 is the aligned uniform base.
                # Each instance independently chooses K of its own vertices as
                # centers. No cross-family gradient-norm comparison is used.
                clustered_cpu, cluster_metadata = (
                    generate_rbf_soft_contracted_batch(
                        candidate_pool_cpu[0],
                        generator=cluster_generator,
                        k_min=args.cluster_k_min,
                        k_max=args.cluster_k_max,
                        alpha=args.cluster_alpha,
                        rho=args.rbf_rho,
                    )
                )
                selection_rule = "clustered_rbf_soft_candidate0"
                scored_candidate_batches = 1
                stats = train_one_update_uniform(
                    model,
                    env,
                    optimizer,
                    parameters,
                    clustered_cpu.unsqueeze(0),
                    rollout_seed,
                    device,
                )
                cluster_stats = cluster_metadata.summary()
            elif epoch_mode == "uniform":
                selection_rule = "uniform_candidate0"
                scored_candidate_batches = 1
                stats = train_one_update_uniform(
                    model,
                    env,
                    optimizer,
                    parameters,
                    candidate_pool_cpu,
                    rollout_seed,
                    device,
                )
            else:
                raise RuntimeError(f"Unsupported epoch mode: {epoch_mode}")

            selected_instances += batch_size
            epoch_scores.append(float(stats["selected_score"]))
            epoch_losses.append(float(stats["selected_loss"]))
            epoch_norms.append(float(stats["selected_norm"]))

            row = {
                "epoch": epoch,
                "continuation_epoch_index": continuation_epoch_index,
                "cycle_index": cycle_index,
                "cycle_position": cycle_position,
                "epoch_mode": epoch_mode,
                "update_in_epoch": update_in_epoch,
                "global_update": global_update,
                "selected_instances_in_epoch": selected_instances,
                "candidate_batch_size": batch_size,
                "configured_candidate_batches": args.candidate_batches,
                "effective_candidate_batches": effective_candidate_batches,
                "scored_candidate_batches": scored_candidate_batches,
                **cluster_stats,
                "selection_rule": selection_rule,
                "mode": "mixed_maxgrad_rbf_soft_uniform",
                "lr": current_lr,
                "scheduler_last_epoch": scheduler.last_epoch,
                "rollout_seed": rollout_seed,
                **stats,
            }
            append_history_row(history_path, row, history_fields)

            if (
                update_in_epoch <= 10
                or update_in_epoch % args.log_interval_updates == 0
                or selected_instances == args.train_episodes
            ):
                logger.info(
                    "epoch=%d cycle=%d position=%d epoch_mode=%s "
                    "update=%d selected=%d/%d batch=%d "
                    "generated=%d scored=%d cluster_k=%s alpha=%s rho=%s "
                    "sigma_mean=%s entropy=%s "
                    "norm(mean/median/max)=%.6f/%.6f/%.6f max_z=%.3f "
                    "rule=%s candidate=%d selected_norm=%.6f selected_z=%.3f "
                    "score=%.6f loss=%.6f time=%.3fs",
                    epoch,
                    cycle_index,
                    cycle_position,
                    epoch_mode,
                    update_in_epoch,
                    selected_instances,
                    args.train_episodes,
                    batch_size,
                    effective_candidate_batches,
                    scored_candidate_batches,
                    cluster_stats["cluster_k"],
                    cluster_stats["cluster_alpha"],
                    cluster_stats["rbf_rho"],
                    cluster_stats["rbf_sigma_mean"],
                    cluster_stats["soft_assignment_entropy_mean"],
                    stats["norm_mean"],
                    stats["norm_median"],
                    stats["norm_max"],
                    stats["norm_max_z"],
                    selection_rule,
                    stats["selected_index"],
                    stats["selected_norm"],
                    stats["selected_norm_z"],
                    stats["selected_score"],
                    stats["selected_loss"],
                    stats["scoring_and_step_seconds"],
                )

        logger.info(
            "Epoch %d done | cycle=%d position=%d epoch_mode=%s | "
            "updates=%d | instances=%d | score_mean=%.6f | "
            "loss_mean=%.6f | selected_norm_mean=%.6f | elapsed=%.2fs",
            epoch,
            cycle_index,
            cycle_position,
            epoch_mode,
            update_in_epoch,
            selected_instances,
            float(np.mean(epoch_scores)),
            float(np.mean(epoch_losses)),
            float(np.mean(epoch_norms)),
            time.perf_counter() - epoch_start,
        )

        save_checkpoint(
            output_dir / "checkpoint-latest.pt",
            epoch,
            global_update,
            model,
            optimizer,
            scheduler,
            data_generator,
            cluster_generator,
            config,
        )

        if epoch % args.save_interval_epochs == 0 or epoch == end_epoch:
            save_checkpoint(
                output_dir / f"checkpoint-{epoch}.pt",
                epoch,
                global_update,
                model,
                optimizer,
                scheduler,
                data_generator,
                cluster_generator,
                config,
            )
            logger.info("Saved numbered checkpoint for epoch %d", epoch)

    save_checkpoint(
        output_dir / "checkpoint-final.pt",
        end_epoch,
        global_update,
        model,
        optimizer,
        scheduler,
        data_generator,
        cluster_generator,
        config,
    )
    logger.info("Training complete in %.2fs", time.perf_counter() - train_start)
    logger.info("Final checkpoint: %s", output_dir / "checkpoint-final.pt")


if __name__ == "__main__":
    main()
