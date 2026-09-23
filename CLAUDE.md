# CLAUDE.md — TransGap v2
## Project
Neural network pruning research codebase. Structured pruning via Transformation Gap (TG).
Targets CIFAR-10 (ResNet-56/110) and ImageNet (ResNet-50).

## Cluster: Truba HPC
- SSH: `mhassan@172.16.6.11`
- Code dir: `/arf/home/mhassan/transgap_v2/`
- Scratch (data/checkpoints/logs): `/arf/scratch/mhassan/transgap_v2/`
- GPU partition: `akya-cuda`, max wall time per job: 3 days
- **Always add `#SBATCH --chdir=/arf/scratch/mhassan/transgap_v2`** to every new SLURM script — Truba rejects jobs not running under `/arf/scratch/`
- Submit pattern: `cd /arf/scratch/mhassan/transgap_v2 && sbatch /arf/home/mhassan/transgap_v2/scripts/<job>.slurm`
- Module load: `module purge && module load comp/python/miniconda3 && source activate`

## Sync
```bash
rsync -av --exclude='.git' --exclude='*.pth' --exclude='data/' --exclude='__pycache__' \
    . mhassan@172.16.6.11:/arf/home/mhassan/transgap_v2/
```

## Key conventions

### Data loaders
Use `get_cifar_loaders()` from `training.baseline` — not `data.cifar` (that module does not exist).

### Block removal
Use `model.replace_block_with_identity(block_name)` (defined on `ResNet` in `models/resnet_cifar.py`).
Never zero-out weights; always use `IdentityBlock`.

### Checkpoints
Format: `{'model_state_dict', 'optimizer_state_dict', 'epoch', 'best_acc', 'config'}`.
Load with `load_checkpoint(path, model, device=device)` from `utils.checkpoint`.
Naming convention: `<model>_<dataset>_<tag>.pth`, e.g. `resnet110_cifar10_baseline.pth`.

### Fine-tune recipe (150-epoch standard — matches manuscript §4.4/§5.7, verbatim-confirmed
### against "TransGap_manuscript formatted.docx" in old-work/paper1-transgap/archive/00_2026_root_archive.zip)
```python
ft_config = {
    'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
    'epochs': 150, 'warmup_epochs': 5,
    'label_smoothing': 0.1,
    'use_mixup': True, 'mixup_alpha': 0.2,
    'use_kd': True, 'kd_alpha': 0.5, 'kd_temperature': 4.0,
    'grad_clip': 5.0, 'save_every': 50,
}
```

### TG threshold
`TG < 0.025` is the canonical pruning cutoff for both R56 and R110 on CIFAR-10.

## Experiment index

| ID | Script | What it does |
|----|--------|-------------|
| Baseline | `run_baseline.py` | Train R56/R110 from scratch |
| Main pruning | `run_pruning.py` | Full block+channel pruning pipeline |
| A1–A7 | `run_ablations.py` | Ablation suite |
| A8 | `run_ablation_A8_full.py` | Single-block removal vs TG (R56/R110/R50, §4.4 recipe: 150 epochs CIFAR / 90 epochs ImageNet) |
| E1 | `run_e1_r110_correlation.py` | Cross-backbone TG correlation (R110, 150 epochs) |
| Orthogonality | `run_orthogonality.py` | TG distribution analysis |
| Multi-seed | `run_multiseed.py` | Robustness across seeds |
| ImageNet | `run_imagenet_full.py` | R50/ImageNet full pipeline |
| Idea12-P1 | `run_phase1_validation.py` | Idea #12 Phase 1: PropTG vs. Table 7 ground truth (10 known R56 blocks, forward-passes only, no fine-tuning). See `../IDEA12_MULTIHOP_PROPAGATED_IMPACT_PLAN.md` for the full spec and Gate 1 decision rule. Unit tests: `tests/test_propagated_tg.py` (6/6 passing locally, CPU-only, no checkpoint needed). |

## E1 workflow (current active experiment)
```bash
# 1. scan  → e1_r110_scan.json
sbatch scripts/truba_e1_scan.slurm

# 2. per-block fine-tune (parallel)
bash scripts/launch_e1_r110.sh

# 3. aggregate → Pearson r
python -m experiments.run_e1_r110_correlation --aggregate \
    --save_dir /arf/scratch/mhassan/transgap_v2/checkpoints
```
Results saved as `e1_r110_<block>_summary.json` and `e1_r110_aggregate.json`.

## ResNet-110 structure
- 54 total blocks: layer1[0..17], layer2[0..17], layer3[0..17]
- Non-prunable (downsampling): layer2[0], layer3[0]
- Baseline accuracy (CIFAR-10): **95.89%**
