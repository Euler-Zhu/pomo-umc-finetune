# POMO U/M/C Fine-Tuning

This repository contains the training and evaluation framework used for
TSP100 continuation experiments based on POMO.

Training starts from the pretrained POMO checkpoint at epoch 2000 and
supports arbitrary schedules composed of three training primitives:

- **U**: Uniform training
- **M**: MaxGrad training
- **C**: RBFSoft clustered training

The main trainer is generic: a new Python training script is **not**
required for each U/M/C schedule. The schedule is specified through the
`--cycle-pattern` argument.

---

## 1. Repository Structure

```text
pomo_umc_finetune/
├── README.md
├── requirements.txt
├── checkpoints/
│   └── checkpoint-2000.pt
├── config/
│   └── paths.env.example
├── data/
│   ├── data_farm/
│   │   ├── tsp100_uniform.txt
│   │   ├── tsp100_clustered1.txt
│   │   ├── tsp100_clustered2.txt
│   │   ├── tsp100_explosion.txt
│   │   └── tsp100_implosion.txt
│   └── solution_farm/
│       ├── gurobi_tsp100_uniform.txt
│       ├── gurobi_tsp100_clustered1.txt
│       ├── gurobi_tsp100_clustered2.txt
│       ├── gurobi_tsp100_explosion.txt
│       └── gurobi_tsp100_implosion.txt
├── pomo_patch/
│   └── tspmodel_cuda_device.patch
├── scripts/
│   ├── train_pomo100_umc.py
│   ├── rbf_soft_clustered_generator.py
│   ├── eval_std_pgd_generalization.py
│   └── jobs/
│       ├── train_MMMMM.sbatch
│       ├── train_MMCUU.sbatch
│       ├── train_UUCUU.sbatch
│       └── evaluate.sbatch
└── logs/
```

Training and evaluation outputs are written under `outputs/`.

Slurm logs are written under `logs/`.

Generated outputs and machine-specific path configurations are excluded
from Git.

---

## 2. Training Primitives

### U: Uniform

`U` denotes ordinary POMO training on uniformly generated TSP100
instances.

For an ordinary training update, the selected TSP100 batch is passed
through the standard POMO policy-gradient training procedure.

In the aligned U/M/C implementation, the candidate stream is kept
consistent across schedules so that differences between schedules are
not caused simply by different random instance streams.

---

### M: MaxGrad

`M` denotes MaxGrad training.

For each complete training update:

1. Generate 8 aligned Uniform candidate batches.
2. Compute the standard POMO policy-gradient loss for each candidate.
3. Compute the full-model gradient norm for each candidate batch.
4. Select the candidate batch with the largest gradient norm.
5. Perform the standard POMO training update using the selected batch.

Thus,

```text
M = MaxGrad selection among 8 Uniform candidate batches.
```

The gradient norm is used only for selecting the training batch.
The actual optimization step remains the standard POMO policy-gradient
update.

---

### C: RBFSoft

`C` denotes training using the RBFSoft clustered-instance generator.

For a complete C update:

1. Generate the aligned Uniform candidate batches.
2. Take candidate batch 0.
3. Transform candidate batch 0 using the RBFSoft generator.
4. Train POMO on the transformed batch using the standard POMO update.

No MaxGrad comparison is performed during a C update.

The aligned candidate generation is retained so that U/M/C schedules
can be compared using consistent random candidate streams.

---

## 3. Generic U/M/C Trainer

The main training script is:

```text
scripts/train_pomo100_umc.py
```

The trainer supports exactly the following epoch modes:

```text
uniform
maxgrad
clustered
```

An arbitrary schedule is supplied through:

```text
--cycle-pattern
```

For example,

```bash
--cycle-pattern "maxgrad,maxgrad,clustered,uniform,uniform"
```

corresponds to

```text
M M C U U
```

and

```bash
--cycle-pattern "uniform,clustered,maxgrad,clustered,uniform"
```

corresponds to

```text
U C M C U
```

The specified cycle is repeated until the requested number of
continuation epochs has been completed.

---

## 4. Included Example Schedules

Three example Slurm jobs are provided.

### MMMMM

```text
M M M M M ...
```

All continuation epochs use MaxGrad.

Run from the repository root:

```bash
sbatch scripts/jobs/train_MMMMM.sbatch
```

The job uses:

```text
--cycle-pattern "maxgrad"
```

---

### MMCUU

```text
M M C U U
```

The five-epoch cycle is repeated throughout the continuation.

For a 50-epoch continuation, this gives:

```text
40% M
20% C
40% U
```

Run:

```bash
sbatch scripts/jobs/train_MMCUU.sbatch
```

The job uses:

```text
--cycle-pattern "maxgrad,maxgrad,clustered,uniform,uniform"
```

---

### UUCUU

```text
U U C U U
```

The five-epoch cycle is repeated throughout the continuation.

For a 50-epoch continuation, this gives:

```text
0% M
20% C
80% U
```

Run:

```bash
sbatch scripts/jobs/train_UUCUU.sbatch
```

The job uses:

```text
--cycle-pattern "uniform,uniform,clustered,uniform,uniform"
```

These jobs are examples only. New schedules can be created by changing
`--cycle-pattern`; no modification of the Python trainer is required.

---

## 5. Default Continuation Configuration

The included experiments use the following default configuration:

```text
Problem size:              TSP100
Source checkpoint epoch:   2000
Continuation epochs:       50
Final epoch:               2050

Training instances/epoch:  100000
Training batch size:       64
MaxGrad candidates:        8

Seed:                      20260717
Expected initial LR:       1e-6

Optimizer:                 restored Adam state
Scheduler:                 restored checkpoint state
```

The continuation restores the model state, optimizer state, and
scheduler state from the epoch-2000 checkpoint.

Because

```text
100000 = 1562 * 64 + 32,
```

each epoch contains 1562 complete 64-instance updates followed by one
final incomplete 32-instance remainder.

The final 32-instance remainder is trained as ordinary Uniform rather
than applying MaxGrad or RBFSoft processing.

---

## 6. RBFSoft Clustered Generator

The RBFSoft implementation is located at:

```text
scripts/rbf_soft_clustered_generator.py
```

The default C configuration is:

```text
K_min = 2
K_max = 10
alpha = 0.8
rho = 0.35
```

For each complete transformed batch, one integer

```text
K ~ Uniform{2, ..., 10}
```

is sampled and shared across the batch.

For each TSP instance independently, `K` centers are selected from the
nodes of that instance.

Let the selected centers be

```text
mu_1, ..., mu_K.
```

For every center, compute the distance to its nearest other center.
The RBF bandwidth is

```text
sigma
    = rho
      * median(nearest-neighbor distance among the K centers).
```

For node `x_i`, the unnormalized RBF weight associated with center
`mu_j` is

```text
exp(
    - ||x_i - mu_j||^2
    / (2 * sigma^2)
)
```

and the weights are normalized across the `K` centers.

The resulting soft center is

```text
soft_center_i
    = sum_j w_ij * mu_j.
```

The transformed node is

```text
x_i'
    = (1 - alpha) * x_i
      + alpha * soft_center_i.
```

With the default setting,

```text
alpha = 0.8
rho   = 0.35.
```

No min-max normalization is applied after the transformation.

No clipping-based geometric renormalization is required.

---

## 7. POMO Dependency

This repository does not duplicate the complete POMO source tree.

The experiments use the official POMO repository:

```text
https://github.com/yd-kwon/POMO.git
```

with commit:

```text
d7c3d6ea580499a53e874fe9e065f69e799a8551
```

Clone the official repository:

```bash
git clone https://github.com/yd-kwon/POMO.git
cd POMO
```

Checkout the required commit:

```bash
git checkout d7c3d6ea580499a53e874fe9e065f69e799a8551
```

The expected POMO TSP directory is then:

```text
/path/to/POMO/NEW_py_ver/TSP
```

---

## 8. POMO CUDA Device Patch

The experiments use one small modification to the official POMO
`TSPModel.py`.

The patch is included at:

```text
pomo_patch/tspmodel_cuda_device.patch
```

It changes the initialization of the first selected-node tensor and
probability tensor so that they are created on the same device as the
POMO state.

Conceptually, the modification is:

```python
selected = torch.arange(
    pomo_size,
    device=state.BATCH_IDX.device,
)[None, :].expand(batch_size, pomo_size)

prob = torch.ones(
    size=(batch_size, pomo_size),
    device=state.BATCH_IDX.device,
)
```

From the root directory of the cloned official POMO repository, apply
the patch with:

```bash
git apply /path/to/pomo_umc_finetune/pomo_patch/tspmodel_cuda_device.patch
```

The other local modifications that may have existed in the original
development copy of POMO are not required by this U/M/C training and
evaluation repository.

---

## 9. Source Checkpoint

The starting POMO TSP100 checkpoint is included:

```text
checkpoints/checkpoint-2000.pt
```

Its SHA-256 checksum is:

```text
1a55eea5cce3e4f9fd5a456d7a86089330aae211c2745c80656932d246dc71b5
```

The checkpoint size is approximately 15 MB.

The continuation trainer expects:

```text
source epoch = 2000
initial learning rate = 1e-6
```

and restores the model, Adam optimizer, and scheduler states from this
checkpoint.

The checksum can be verified with:

```bash
sha256sum checkpoints/checkpoint-2000.pt
```

---

## 10. Python Environment

The original experiments were run using:

```text
Python 3.10.20
torch 2.7.1+cu118
numpy 1.26.4
pandas 2.3.3
scipy 1.15.3
matplotlib 3.10.9
tqdm 4.67.3
```

The package versions are recorded in:

```text
requirements.txt
```

The exact CUDA/PyTorch installation procedure may depend on the GPU
cluster and CUDA environment.

---

## 11. Local Path Configuration

Machine-specific paths are stored in:

```text
config/paths.env
```

This file is intentionally excluded from Git.

Create it from the example:

```bash
cp config/paths.env.example config/paths.env
```

Then edit it for the local machine:

```text
PROJECT_ROOT=/path/to/pomo_umc_finetune
POMO_TSP_ROOT=/path/to/POMO/NEW_py_ver/TSP
POMO_CHECKPOINT=/path/to/pomo_umc_finetune/checkpoints/checkpoint-2000.pt
VENV=/path/to/pomo/venv
```

The variables are:

```text
PROJECT_ROOT
    Root directory of this repository.

POMO_TSP_ROOT
    Path to NEW_py_ver/TSP inside the cloned official POMO repository.

POMO_CHECKPOINT
    Path to the supplied epoch-2000 TSP100 checkpoint.

VENV
    Python virtual environment used to run POMO.
```

All supplied Slurm commands should be submitted from the root directory
of this repository because the job scripts source:

```text
config/paths.env
```

---

## 12. Training

After configuring POMO and `config/paths.env`, run one of the supplied
training jobs from the repository root.

MMMMM:

```bash
sbatch scripts/jobs/train_MMMMM.sbatch
```

MMCUU:

```bash
sbatch scripts/jobs/train_MMCUU.sbatch
```

UUCUU:

```bash
sbatch scripts/jobs/train_UUCUU.sbatch
```

The default continuation length can also be overridden through the
environment variable used by the job scripts.

For example:

```bash
ADDITIONAL_EPOCHS=10 sbatch scripts/jobs/train_MMCUU.sbatch
```

The default random seed can similarly be overridden:

```bash
SEED=12345 sbatch scripts/jobs/train_MMCUU.sbatch
```

Training outputs are written under:

```text
outputs/
```

---

## 13. Evaluation Datasets

Five fixed TSP100 evaluation families are included.

### Uniform

```text
data/data_farm/tsp100_uniform.txt
data/solution_farm/gurobi_tsp100_uniform.txt
```

### Clustered 1

```text
data/data_farm/tsp100_clustered1.txt
data/solution_farm/gurobi_tsp100_clustered1.txt
```

### Clustered 2

```text
data/data_farm/tsp100_clustered2.txt
data/solution_farm/gurobi_tsp100_clustered2.txt
```

### Explosion

```text
data/data_farm/tsp100_explosion.txt
data/solution_farm/gurobi_tsp100_explosion.txt
```

### Implosion

```text
data/data_farm/tsp100_implosion.txt
data/solution_farm/gurobi_tsp100_implosion.txt
```

Each test family contains:

```text
2000 TSP100 instances.
```

Each exact-solution file is aligned instance-by-instance with its
corresponding test-instance file.

---

## 14. Evaluation Protocol

The common evaluation script is:

```text
scripts/eval_std_pgd_generalization.py
```

The supplied evaluation covers all five test families:

```text
Uniform
Clustered 1
Clustered 2
Explosion
Implosion
```

Evaluation reports POMO performance using:

```text
No augmentation
Aug8
```

and computes optimality gaps against the supplied exact solutions.

The epoch-2000 source model can be evaluated together with the
fine-tuned model as a common reference.

---

## 15. Generic Evaluation Slurm Job

The generic evaluation job is:

```text
scripts/jobs/evaluate.sbatch
```

It is not tied to MMMMM, MMCUU, or UUCUU.

Any trained POMO checkpoint can be evaluated by supplying:

```text
MODEL_NAME
MODEL_CHECKPOINT
```

For example, to evaluate an MMCUU epoch-2050 checkpoint:

```bash
sbatch \
  --export=ALL,MODEL_NAME=MMCUU2050,MODEL_CHECKPOINT=/absolute/path/to/checkpoint-2050.pt \
  scripts/jobs/evaluate.sbatch
```

To evaluate UUCUU:

```bash
sbatch \
  --export=ALL,MODEL_NAME=UUCUU2050,MODEL_CHECKPOINT=/absolute/path/to/checkpoint-2050.pt \
  scripts/jobs/evaluate.sbatch
```

To evaluate MMMMM:

```bash
sbatch \
  --export=ALL,MODEL_NAME=MMMMM2050,MODEL_CHECKPOINT=/absolute/path/to/checkpoint-2050.pt \
  scripts/jobs/evaluate.sbatch
```

A newly designed schedule, for example UCMCM, uses exactly the same
evaluation pipeline:

```bash
sbatch \
  --export=ALL,MODEL_NAME=UCMCM2050,MODEL_CHECKPOINT=/absolute/path/to/checkpoint-2050.pt \
  scripts/jobs/evaluate.sbatch
```

No new evaluation Python script is required.

---

## 16. Reproducibility Notes

The main continuation experiments share the following configuration:

```text
TSP size:                  100
Source checkpoint:         POMO epoch 2000
Continuation:              50 epochs
Training instances/epoch:  100000
Batch size:                64
Candidate batches:         8
Seed:                      20260717
Expected LR:               1e-6
```

For RBFSoft C epochs:

```text
K in {2, ..., 10}
alpha = 0.8
rho = 0.35
```

The U/M/C framework was designed so that different schedules can use
aligned candidate-generation streams, making schedule comparisons less
sensitive to differences in random training-instance generation.

---

## 17. Creating a New Schedule

To create a new schedule, either edit one of the example Slurm jobs or
create a copy.

For example:

```bash
cp scripts/jobs/train_MMCUU.sbatch scripts/jobs/train_UCMCM.sbatch
```

Change the Slurm job name and output directory, then replace:

```text
--cycle-pattern "maxgrad,maxgrad,clustered,uniform,uniform"
```

with:

```text
--cycle-pattern "uniform,clustered,maxgrad,clustered,maxgrad"
```

The generic Python trainer requires no other modification.

---

## 18. Important Files

The core method implementation is:

```text
scripts/train_pomo100_umc.py
```

The RBFSoft generator is:

```text
scripts/rbf_soft_clustered_generator.py
```

The common evaluator is:

```text
scripts/eval_std_pgd_generalization.py
```

The official-POMO compatibility patch is:

```text
pomo_patch/tspmodel_cuda_device.patch
```

The continuation source checkpoint is:

```text
checkpoints/checkpoint-2000.pt
```

The example Slurm entry points are:

```text
scripts/jobs/train_MMMMM.sbatch
scripts/jobs/train_MMCUU.sbatch
scripts/jobs/train_UUCUU.sbatch
scripts/jobs/evaluate.sbatch
```

---

## 19. Output and Version-Control Policy

Machine-specific configuration is excluded:

```text
config/paths.env
```

Training and evaluation outputs are excluded:

```text
outputs/
```

Generated Slurm output/error files are excluded.

The repository keeps:

```text
logs/.gitkeep
```

so that the `logs/` directory exists immediately after cloning.

Python cache files are excluded:

```text
__pycache__/
*.pyc
*.pyo
```

---

## 20. Summary

The repository provides one unified pipeline:

```text
POMO TSP100 epoch-2000 checkpoint
            |
            v
   arbitrary U/M/C schedule
            |
            v
      continued training
            |
            v
     trained checkpoint
            |
            v
 five fixed TSP100 test families
            |
            v
   exact optimality-gap evaluation
```

The main design goal is to make U/M/C schedules interchangeable while
keeping the underlying POMO training and evaluation pipeline fixed.
