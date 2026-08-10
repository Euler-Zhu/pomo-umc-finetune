#!/usr/bin/env python3
"""Gaussian-RBF soft-contraction generator for clustered Euclidean TSP.

For one complete batch:

1. Receive a batch of aligned uniform base instances U in [0,1]^(B x N x 2).
2. Sample exactly one shared center count K ~ Uniform{k_min, ..., k_max}.
   The same K is used by all B instances in this batch.
3. For each instance independently, choose K distinct existing vertices as
   RBF centers.
4. Set an instance-adaptive bandwidth

       sigma = rho * median_j min_{l != j} ||mu_j - mu_l||_2.

5. Compute Gaussian soft assignments

       w_ij = softmax_j(-||u_i-mu_j||^2 / (2 sigma^2)).

6. Form a soft center and contract:

       mu_bar_i = sum_j w_ij mu_j,
       x_i = (1-alpha) u_i + alpha mu_bar_i.

The construction is a smooth counterpart of the Voronoi contraction
generator. It uses the same shared-K rule, the same center-selection rule, and
the same contraction strength. The intended experimental change is hard
nearest-center assignment versus Gaussian soft assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import torch


_SIGMA_FLOOR = 1e-6


@dataclass(frozen=True)
class RBFSoftBatchMetadata:
    k: int
    alpha: float
    rho: float
    center_indices: torch.Tensor
    centers: torch.Tensor
    sigma: torch.Tensor
    soft_weights: torch.Tensor
    hard_assignments: torch.Tensor
    hard_cluster_sizes: torch.Tensor
    median_center_nn_distance: torch.Tensor
    displacement_norms: torch.Tensor
    sigma_floor_hits: int

    def summary(self) -> Dict[str, float]:
        sizes = self.hard_cluster_sizes.to(dtype=torch.float64)
        sigma = self.sigma.to(dtype=torch.float64)
        center_spacing = self.median_center_nn_distance.to(
            dtype=torch.float64
        )
        weights = self.soft_weights.to(dtype=torch.float64)
        entropy = -(
            weights.clamp_min(torch.finfo(weights.dtype).tiny)
            * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
        ).sum(dim=2)
        normalized_entropy = entropy / math.log(float(self.k))
        max_weight = weights.max(dim=2).values
        displacement = self.displacement_norms.to(dtype=torch.float64)

        return {
            "cluster_k": int(self.k),
            "cluster_alpha": float(self.alpha),
            "rbf_rho": float(self.rho),
            "rbf_sigma_min": float(sigma.min().item()),
            "rbf_sigma_max": float(sigma.max().item()),
            "rbf_sigma_mean": float(sigma.mean().item()),
            "rbf_sigma_std": float(
                sigma.std(unbiased=False).item()
            ),
            "center_nn_median_mean": float(
                center_spacing.mean().item()
            ),
            "hard_cluster_size_min": int(sizes.min().item()),
            "hard_cluster_size_max": int(sizes.max().item()),
            "hard_cluster_size_mean": float(sizes.mean().item()),
            "hard_cluster_size_std": float(
                sizes.std(unbiased=False).item()
            ),
            "hard_cluster_empty_count": int(
                (sizes == 0).sum().item()
            ),
            "soft_assignment_entropy_mean": float(
                normalized_entropy.mean().item()
            ),
            "soft_assignment_max_weight_mean": float(
                max_weight.mean().item()
            ),
            "displacement_mean": float(displacement.mean().item()),
            "displacement_max": float(displacement.max().item()),
            "rbf_sigma_floor_hits": int(self.sigma_floor_hits),
        }


def _validate_base_points(base_points: torch.Tensor) -> None:
    if base_points.ndim != 3 or base_points.size(-1) != 2:
        raise ValueError(
            "base_points must have shape (batch_size, problem_size, 2); "
            f"got {tuple(base_points.shape)}."
        )
    if not base_points.is_floating_point():
        raise TypeError("base_points must be a floating-point tensor.")
    if base_points.device.type != "cpu":
        raise ValueError(
            "The generator intentionally operates on CPU tensors so its "
            "random stream is independent of CUDA execution."
        )
    if not torch.isfinite(base_points).all():
        raise ValueError("base_points contains NaN or infinity.")
    if bool((base_points < 0).any()) or bool((base_points > 1).any()):
        raise ValueError("base_points must lie inside [0,1]^2.")


def sample_shared_center_count(
    *,
    generator: torch.Generator,
    k_min: int,
    k_max: int,
    problem_size: int,
) -> int:
    """Sample one integer K uniformly from the inclusive range."""
    if k_min < 2:
        raise ValueError("k_min must be at least 2 for adaptive bandwidth.")
    if k_max < k_min:
        raise ValueError("k_max must be at least k_min.")
    if k_max > problem_size:
        raise ValueError(
            f"k_max={k_max} exceeds problem_size={problem_size}."
        )

    return int(
        torch.randint(
            low=k_min,
            high=k_max + 1,
            size=(1,),
            generator=generator,
            device="cpu",
        ).item()
    )


def generate_rbf_soft_contracted_batch(
    base_points: torch.Tensor,
    *,
    generator: torch.Generator,
    k_min: int = 2,
    k_max: int = 10,
    alpha: float = 0.8,
    rho: float = 0.35,
) -> Tuple[torch.Tensor, RBFSoftBatchMetadata]:
    """Generate one RBF-soft clustered batch from an aligned uniform batch.

    A single K is sampled for the entire batch. Each instance independently
    selects K distinct centers from its own point set. Bandwidth is adaptive
    per instance but rho is fixed across the experiment.
    """
    _validate_base_points(base_points)

    batch_size, problem_size, coordinate_dim = base_points.shape
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1].")
    if rho <= 0.0:
        raise ValueError("rho must be positive.")

    k = sample_shared_center_count(
        generator=generator,
        k_min=k_min,
        k_max=k_max,
        problem_size=problem_size,
    )

    # Independent random permutations per instance. The first K indices are
    # sampled without replacement. K itself is shared by the whole batch.
    random_keys = torch.rand(
        (batch_size, problem_size),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    center_indices = random_keys.argsort(dim=1)[:, :k]
    gather_index = center_indices[:, :, None].expand(
        batch_size, k, coordinate_dim
    )
    centers = torch.gather(base_points, dim=1, index=gather_index)

    # Pairwise center distances. The diagonal is excluded when finding each
    # center's nearest neighboring center.
    center_distances = torch.cdist(centers, centers, p=2)
    diagonal_mask = torch.eye(
        k, dtype=torch.bool, device="cpu"
    )[None, :, :]
    center_distances_excluding_self = center_distances.masked_fill(
        diagonal_mask, float("inf")
    )
    nearest_center_distance = (
        center_distances_excluding_self.min(dim=2).values
    )
    median_center_nn_distance = nearest_center_distance.median(
        dim=1
    ).values

    raw_sigma = rho * median_center_nn_distance
    sigma_floor_hits = int((raw_sigma < _SIGMA_FLOOR).sum().item())
    sigma = raw_sigma.clamp_min(_SIGMA_FLOOR)

    squared_distances = (
        base_points[:, :, None, :] - centers[:, None, :, :]
    ).square().sum(dim=-1)
    logits = -squared_distances / (
        2.0 * sigma[:, None, None].square()
    )
    soft_weights = torch.softmax(logits, dim=2)

    soft_centers = torch.einsum(
        "bnk,bkd->bnd", soft_weights, centers
    )
    clustered = (
        (1.0 - alpha) * base_points + alpha * soft_centers
    ).contiguous()

    # Argmax of Gaussian weights is nearest-center assignment because one
    # scalar sigma is shared by all centers within each instance.
    hard_assignments = soft_weights.argmax(dim=2)
    hard_cluster_sizes = torch.zeros(
        (batch_size, k), dtype=torch.int64, device="cpu"
    )
    hard_cluster_sizes.scatter_add_(
        dim=1,
        index=hard_assignments,
        src=torch.ones_like(hard_assignments, dtype=torch.int64),
    )

    # Every selected center has zero distance to itself, so under nonduplicate
    # centers every hard Voronoi cell contains at least its selected vertex.
    # Continuous uniform instances make duplicates probability zero.
    empty_count = int((hard_cluster_sizes == 0).sum().item())
    if empty_count != 0:
        raise RuntimeError(
            f"RBF soft construction unexpectedly produced {empty_count} "
            "empty hard-assignment cells."
        )

    if not torch.isfinite(clustered).all():
        raise RuntimeError("RBF-soft generator produced NaN or infinity.")
    tolerance = 1e-6
    if bool((clustered < -tolerance).any()) or bool(
        (clustered > 1.0 + tolerance).any()
    ):
        raise RuntimeError(
            "Convex-combination construction left the unit square."
        )

    displacement_norms = (clustered - base_points).norm(dim=2)

    metadata = RBFSoftBatchMetadata(
        k=k,
        alpha=float(alpha),
        rho=float(rho),
        center_indices=center_indices,
        centers=centers,
        sigma=sigma,
        soft_weights=soft_weights,
        hard_assignments=hard_assignments,
        hard_cluster_sizes=hard_cluster_sizes,
        median_center_nn_distance=median_center_nn_distance,
        displacement_norms=displacement_norms,
        sigma_floor_hits=sigma_floor_hits,
    )
    return clustered, metadata
