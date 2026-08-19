<h1 align="center">RL-100</h1>

<h3 align="center">Performant Robotic Manipulation with Real-World Reinforcement Learning</h3>

<p align="center">
  <b>A unified library for diffusion / flow policy RL post-training, iterative offline data flywheels, and real-world robot RL.</b>
</p>

Flexiv 双臂、双 Inspire 手、三视角 RGB 的 30 Hz 数据接入、BC 训练与安全部署说明：
[FLEXIV_DP30.zh-CN.md](FLEXIV_DP30.zh-CN.md)。

<p align="center">
  <a href="https://arxiv.org/abs/2510.14830"><img src="https://img.shields.io/badge/Paper-arXiv%202510.14830-b31b1b?style=for-the-badge" alt="Paper"></a>
  <a href="https://lei-kun.github.io/RL-100/"><img src="https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge" alt="Project Page"></a>
  <a href="https://youtu.be/OrnHUgMGmlU"><img src="https://img.shields.io/badge/Video-YouTube-dc2626?style=for-the-badge" alt="YouTube"></a>
  <a href="https://x.com/kunlei15/status/1978840280297255124"><img src="https://img.shields.io/badge/Social-Twitter%20%2F%20X-111827?style=for-the-badge" alt="Twitter / X"></a>
</p>

<p align="center">
  <a href="https://lei-kun.github.io/blogs/RL100.html"><img src="https://img.shields.io/badge/Blog-RL--100-7c3aed?style=flat-square" alt="Blog RL-100"></a>
  <a href="https://lei-kun.github.io/blogs/rl.html"><img src="https://img.shields.io/badge/Blog-RL-7c3aed?style=flat-square" alt="Blog RL"></a>
  <a href="https://zhuanlan.zhihu.com/p/1961019618517836122"><img src="https://img.shields.io/badge/Article-Zhihu-2563eb?style=flat-square" alt="Zhihu"></a>
  <a href="https://lei-kun.github.io/my_files/RL-100.pdf"><img src="https://img.shields.io/badge/Talk-Slides-f59e0b?style=flat-square" alt="Talk Slides"></a>
  <a href="https://www.bilibili.com/video/BV1UurLBoEJK/"><img src="https://img.shields.io/badge/Talk-Bilibili-ec4899?style=flat-square" alt="Talk Bilibili"></a>
  <a href="https://twitter.com/RoboPapers/status/2011453253314134277"><img src="https://img.shields.io/badge/Talk-RoboPapers-111827?style=flat-square" alt="Talk RoboPapers"></a>
</p>

---

<h4 align="center">Authors</h4>

<p align="center">
  <a href="https://lei-kun.github.io/">Kun Lei</a><sup>*†</sup> ·
  <a href="https://li-huanyu.github.io/">Huanyu Li</a><sup>*</sup> ·
  <a href="https://manutdmoon.github.io/">Dongjie Yu</a><sup>*</sup> ·
  <a href="https://zhenyuwei2003.github.io/">Zhenyu Wei</a><sup>*</sup>
  <br>
  <a href="https://lingxiao-guo.github.io/">Lingxiao Guo</a> ·
  <a href="https://jzndd.github.io/">Zhennan Jiang</a> ·
  <a href="https://wadiuvatzy.github.io">Ziyu Wang</a> ·
  <a href="https://www.liang-shiyu.com/">Shiyu Liang</a> ·
  <a href="http://hxu.rocks/">Huazhe Xu</a>
</p>

<p align="center"><b>
  <sup>*</sup> Equal contribution &nbsp;·&nbsp; <sup>†</sup> Project Lead
</b>
</p>

![RL-100 repository overview](media/overview.jpg)

---

## Overview

**RL-100** is a real-world reinforcement learning framework for robotic manipulation built on top of diffusion and flow visuomotor policies. It targets deployment-level **reliability**, **efficiency**, and **robustness** through a unified imitation-to-reinforcement learning pipeline.

RL-100 aims to provide one of the most complete codebases for diffusion-policy RL and real-world robot RL post-training. The framework covers the following combinations in a unified repo.

> **The supported options below are designed to be freely combined across axes for different robot post-training setups.**
>
> <span style="color:#dc2626"><strong>All combinations are implemented through a compact offline RL / online RL training interface.</strong></span>

<table width="100%">
<thead>
<tr>
  <th align="left">Axis</th>
  <th align="left">Supported options</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>Policy backbone</b></td>
  <td>
    <span style="color:#16a34a">✓</span> Diffusion policy<br>
    <span style="color:#16a34a">✓</span> Flow policy
  </td>
</tr>
<tr>
  <td><b>One-step deployment</b></td>
  <td>
    <span style="color:#2563eb">✓</span> Diffusion-to-CM one-step distillation<br>
    <span style="color:#2563eb">✓</span> Flow-to-flow one-step on-policy distillation
  </td>
</tr>
<tr>
  <td><b>Observation modality</b></td>
  <td>
    <span style="color:#f59e0b">✓</span> 3D point cloud<br>
    <span style="color:#f59e0b">✓</span> 2D RGB image
  </td>
</tr>
<tr>
  <td><b>Control mode</b></td>
  <td>
    <span style="color:#dc2626">✓</span> Action chunking<br>
    <span style="color:#dc2626">✓</span> Single-action high-frequency control
  </td>
</tr>
<tr>
  <td><b>Policy extraction strategy</b></td>
  <td>
    <span style="color:#7c3aed">✓</span> Policy gradient<br>
    <span style="color:#7c3aed">✓</span> IDQL-style extraction with reject sampling<br>
    <span style="color:#7c3aed">✓</span> Hybrid variants
  </td>
</tr>
<tr>
  <td><b>RL stage</b></td>
  <td>
    <span style="color:#0891b2">✓</span> Offline policy gradient<br>
    <span style="color:#0891b2">✓</span> Online policy gradient
  </td>
</tr>
<tr>
  <td><b>Training data regime</b></td>
  <td>
    <span style="color:#db2777">✓</span> Teleoperation datasets<br>
    <span style="color:#db2777">✓</span> Iterative offline rollout datasets<br>
    <span style="color:#db2777">✓</span> Online rollouts<br>
    <span style="color:#db2777">✓</span> Offline-to-online mixed training
  </td>
</tr>
</tbody>
</table>

> Starting from human teleoperation demonstrations, RL-100 iteratively refines policies with offline RL, deploys them to collect new real-robot rollouts, merges those rollouts back into the offline dataset, and optionally performs lightweight online RL fine-tuning.

---

## Table of Contents

- [0. Quick Install Pointer](#0-quick-install-pointer)
  - [Download the Smoke-Test Dataset](#download-the-smoke-test-dataset)
- [1. Project Overview](#1-project-overview)
- [2. RL Post-Training Framework](#2-rl-post-training-framework)
- [3. Real-Robot Training and Data Flywheel](#3-real-robot-training-and-data-flywheel)
- [Installation](#installation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## 0. Quick Install Pointer

For environment setup and dependency installation, see the [Installation](#installation) section and [`INSTALL.md`](INSTALL.md). We recommend using `adroit_door_medium` as the first smoke-test task.

### Download the Smoke-Test Dataset

The recommended `adroit_door_medium` zarr dataset is hosted on Hugging Face Datasets: [`leokk/RL-100-adroit-door-medium`](https://huggingface.co/datasets/leokk/RL-100-adroit-door-medium). From the repository root, download it with the Hugging Face CLI:

```bash
python -m pip install -U huggingface_hub
mkdir -p RL-100/data
hf download leokk/RL-100-adroit-door-medium \
  adroit_door_medium.zarr.tar.gz \
  --repo-type dataset \
  --local-dir /tmp
tar -xzf /tmp/adroit_door_medium.zarr.tar.gz -C RL-100/data
```

If direct URL downloads are more convenient in your environment, use:

```bash
mkdir -p RL-100/data
wget -O /tmp/adroit_door_medium.zarr.tar.gz \
  https://huggingface.co/datasets/leokk/RL-100-adroit-door-medium/resolve/main/adroit_door_medium.zarr.tar.gz
tar -xzf /tmp/adroit_door_medium.zarr.tar.gz -C RL-100/data
```

Optional checksum verification:

```bash
echo "73a7cd510a0715492e8f8041843ac51edf1a0feb6d30706a95a5e5857addef37  /tmp/adroit_door_medium.zarr.tar.gz" | sha256sum -c -
```

After extraction, the dataset should be available at:

```text
Repo/RL-100/data/adroit_door_medium.zarr
```

The dataset follows the RL-100 repository/data release terms. The formal dataset license will be finalized together with the public repository license.

## 1. Project Overview

### Why RL-100

RL-100 is designed as a **full-stack real-world robot learning system**, not only a collection of training scripts.

It combines

- **Behavior cloning**
- **Offline policy-gradient RL**
- **Iterative offline rollout data expansion**
- **Online policy-gradient fine-tuning**
- **One-step policy distillation**

in **one codebase**. The goal is to make real-world robot policy post-training a **repeatable pipeline** rather than a set of disconnected scripts.

### Repository Layout

```text
RL-100/
├── rl_100/
│   ├── policy/              # 2D and 3D visuomotor policy implementations
│   ├── unidpg/              # offline / online RL training logic
│   └── config/              # policy, task, and training configs
├── scripts/
│   ├── Diffusion/           # diffusion policy training and online scripts
│   └── Flow/                # flow policy training and online scripts
├── tools/
│   └── teleop_off2off_data/ # teleoperation and iterative offline data flywheel tools
├── third_party/             # external environments and dependencies
└── visualizer/              # lightweight point-cloud visualizer
```

---

## 2. RL Post-Training Framework

RL-100 turns the capability matrix above into a practical post-training pipeline for adapting large visuomotor policies from logged data and online interaction. This section focuses on the algorithmic stages and launchers; the real-robot iterative offline data flywheel is described in [Section 3](#3-real-robot-training-and-data-flywheel).

### Command Selector

Use the interactive selector to generate the launcher command from a few choices:

```bash
python scripts/select_recipe.py
```

It asks for:

<table width="100%">
<thead>
<tr>
  <th align="center" width="80">#</th>
  <th align="left">Choice</th>
</tr>
</thead>
<tbody>
<tr><td align="center">1</td><td>DDIM &nbsp;/&nbsp; Flow</td></tr>
<tr><td align="center">2</td><td>Offline &nbsp;/&nbsp; Online</td></tr>
<tr><td align="center">3</td><td>3D &nbsp;/&nbsp; 2D</td></tr>
<tr><td align="center">4</td><td>Chunk action &nbsp;/&nbsp; Single action</td></tr>
<tr><td align="center">5</td><td>PG &nbsp;/&nbsp; PG + IDQL</td></tr>
<tr><td align="center">6</td><td>One-step distill: Yes &nbsp;/&nbsp; No</td></tr>
</tbody>
</table>

The first four choices select the launcher path. The PG/IDQL and distillation choices print launcher-level suggestions, because those settings are usually controlled inside the bash sweep. For example, the selector will remind you to use launcher flags for **PG** or **PG + IDQL-style extraction**; for offline distillation, first finish the offline sweep and parameter selection, then set `distill_phase='after_offline'`; for online distillation, set `distill_phase='online'`.

<details>
<summary><b>Example output</b></summary>

```text
Command:
bash scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh rl100 adroit_door_medium 0112 100 8
```

</details>

### Training Stages

The full algorithmic pipeline is organized around the following stages.

#### Stage 1 · Behavior Cloning Initialization

Train an initial visuomotor policy from demonstrations. This stage provides a stable policy prior before RL post-training.

#### Stage 2 · Offline RL Post-Training

Improve the initialized policy with offline policy-gradient updates on logged trajectories. This stage supports diffusion and flow policies, 2D and 3D observation backbones, chunk-action and single-action control, and optional IDQL-style extraction configured through the launcher sweep.

**Example offline RL launchers:**

```bash
# Single-GPU 3D diffusion offline RL.
bash scripts/Diffusion/Offline/3D/train_policy.sh rl100 adroit_door_medium 0112 100

# DDP two-stage 3D diffusion offline RL.
# The last argument is the number of GPUs.
bash scripts/Diffusion/Offline/3D/train_policy_two_stage.sh rl100 adroit_door_medium 0112 100 4
```

The 2D and chunk-action offline recipes also provide DDP-style launchers, including:

- `scripts/Diffusion/Offline/2D/train_policy_image_unet_two_stage.sh`
- `scripts/Diffusion/Offline/2D/train_policy_image_unet_chunk_two_stage.sh`
- `scripts/Diffusion/Offline/3D/train_policy_chunk_two_stage.sh`

#### Stage 3 · Online RL Fine-Tuning

Continue improving the policy using real or simulated online rollouts. The current rollout policy collects transitions, and PPO-style updates reuse the existing rollout and log-probability flow.

This stage supports:

- Online policy gradient.
- Diffusion and flow online fine-tuning.
- 2D and 3D online runners.
- Chunked and high-frequency single-action control.
- Lightweight online adaptation after offline RL.

#### Stage 4 · One-Step Policy Distillation

For deployment efficiency, RL-100 includes one-step policy extraction and distillation paths:

- **Diffusion-to-CM** — distill a multi-step diffusion policy into a consistency-model style one-step policy.
- **Flow-to-flow** — distill or extract a one-step flow policy through on-policy training.

These paths are intended for fast real-robot inference while retaining the behavior improved by RL post-training.

#### Stage 5 · Policy Extraction

RL-100 supports policy extraction strategies used in robot policy post-training:

- Policy-gradient extraction from logged or online transitions.
- IDQL-style extraction with reject sampling from higher-quality sampled trajectories.
- Hybrid strategies that combine policy-gradient updates with filtered or selected data.

### Configuration Entry Points

The main implementation and configuration entry points are:

```text
RL-100/rl_100/policy/rl100_3d.py
RL-100/rl_100/policy/rl100_2d.py
RL-100/rl_100/unidpg/
RL-100/rl_100/config/
scripts/Diffusion/
scripts/Flow/
```

Detailed command lines and recommended configs are organized around these entry points and will be expanded as the release scripts are finalized.

---

## 3. Real-Robot Training and Data Flywheel

RL-100 includes a real-robot data flywheel for iterative offline reinforcement learning and final online improvement.

<table width="100%">
<thead>
<tr>
  <th align="left">Stage</th>
  <th align="left">Input</th>
  <th align="left">Output</th>
</tr>
</thead>
<tbody>
<tr><td><b>Teleoperation</b></td><td>Human demonstrations</td><td>Raw robot episodes</td></tr>
<tr><td><b>Data preparation</b></td><td>Raw teleop data and rollout data</td><td>Training zarr dataset</td></tr>
<tr><td><b>Offline RL</b></td><td>Merged offline dataset</td><td>Improved policy checkpoint</td></tr>
<tr><td><b>Real-robot rollout</b></td><td>Offline RL checkpoint</td><td>New policy rollout dataset</td></tr>
<tr><td><b>Iterative offline dataset merge</b></td><td>Base zarr and new rollout h5 files</td><td>Next-round zarr dataset</td></tr>
<tr><td><b>Online RL</b></td><td>Offline RL checkpoint and live rollouts</td><td>Final deployment checkpoint</td></tr>
</tbody>
</table>

```text
human teleoperation
   └─▶ raw teleop data
        └─▶ processed offline dataset
             └─▶ BC / offline RL training
                  └─▶ real-robot policy rollout
                       └─▶ rollout dataset
                            └─▶ merge with previous offline data
                                 └─▶ next offline RL round
                                      └─▶ online RL fine-tuning
```

### Real-Robot Setup

This section tracks the hardware and runtime assumptions needed for real-robot deployment:

- Robot platform and controller requirements.
- Camera and point-cloud setup.
- End-effector / hand / gripper setup.
- Workspace limits and reset protocol.
- Safety checks before running learned policies.

### Teleoperation Data Collection

Teleoperation and data processing utilities live in:

```text
tools/teleop_off2off_data/
```

The planned workflow is:

```bash
cd tools/teleop_off2off_data

# Collect teleoperation data
python teleop.py

# Or use the swapped teleoperation entry if required by the robot setup
python teleop_swapped.py
```

> Hardware-specific command-line arguments should be documented next to the corresponding robot setup.

### Data Preparation

The data preparation entry point is:

```bash
cd tools/teleop_off2off_data
python data_prepare.py --config configs/data_prepare.yaml
```

For the full teleoperation and iterative offline dataset workflow, see [`tools/teleop_off2off_data/DATA_PREPARE.md`](tools/teleop_off2off_data/DATA_PREPARE.md).

The tool supports three modes:

<table width="100%">
<thead>
<tr>
  <th align="left">Mode</th>
  <th align="left">Purpose</th>
</tr>
</thead>
<tbody>
<tr><td><code>raw_to_npy</code></td><td>Convert raw teleoperation episodes into processed <code>.npy</code> data.</td></tr>
<tr><td><code>build_zarr</code></td><td>Build a new training zarr from processed teleop data and selected rollout sources.</td></tr>
<tr><td><code>extend_zarr</code></td><td>Append new rollout sources to a base zarr and write a new zarr without modifying the base dataset in place.</td></tr>
</tbody>
</table>

### Iterative Offline RL Data Flywheel

Iterative offline training is the main data flywheel used by the real-robot pipeline. A policy trained by offline RL is rolled out on the robot, the resulting trajectories are stored as rollout datasets, and the next offline RL round trains on the merged dataset.

Policy rollouts collected on the real robot are treated as named data sources in YAML:

```yaml
rollout_sources:
  - name: off2off_004
    round: "004"
    enabled: true
    type: h5_rollout_dir
    path: /path/to/policy_rollouts/004/online_ft
```

**Typical usage:**

```bash
# Build a dataset from teleop data plus all enabled rollout sources.
python data_prepare.py --config configs/data_prepare.yaml --mode build_zarr --include-rollouts

# Extend an existing zarr with one new rollout source.
python data_prepare.py \
  --config configs/data_prepare.yaml \
  --mode extend_zarr \
  --rollout-source off2off_004 \
  --base-zarr-path /path/to/base.zarr \
  --zarr-output-path /path/to/new.zarr
```

The merge output preserves the training schema expected by RL-100:

```text
data/
├── point_cloud, next_point_cloud
├── state, next_state
├── action, next_action
└── reward, return, done, timeout
meta/
└── episode_ends
```

### Train on the Real-Robot Dataset

After generating a zarr dataset, point the RL-100 task config to the dataset path and run the selected offline RL script.

This section is reserved for the concrete real-robot training recipes:

- Real-robot task config examples.
- 3D point-cloud offline training commands.
- 2D image offline training commands.
- Recommended checkpoint selection and evaluation workflow.

### Real-Robot Rollout Collection

Iterative offline rollout collection reuses the online training launcher as a real-robot data collection wrapper. In practice, load the offline-RL-finetuned policy checkpoint, enter the online script, and use the evaluation call before or during online fine-tuning to roll out the policy on the robot and save `.h5` trajectories.

The real-robot offline and online training bash scripts are the same launchers introduced in [Section 2](#2-rl-post-training-framework). The only extra setting for iterative offline rollout collection is enabling `data_collect=True` in the collection run.

The offline and online launchers must describe the same policy network so that checkpoint weights load cleanly. Keep architecture-related options consistent between the offline training bash and the rollout-collection bash, including the policy family, observation modality, chunk/single-action setting, `policy.model`, `policy.encoder_type`, `policy.use_vib`, `policy.use_recon`, `horizon`, `n_action_steps`, `n_obs_steps`, encoder output size, and diffusion/flow scheduler settings. If the online bash changes these fields, the offline checkpoint may partially fail to load or silently run with a mismatched policy.

For RL data collection, the rollout policy should be stochastic rather than a deterministic evaluation policy. Set `data_collect=True` in the bash/config path used for collection. This enables two noise sources:

- 🔴 **Denoising/SDE action noise:** the real-robot runner switches `deterministic=False` when `data_collect=True`, so `policy.predict_action(...)` samples stochastic denoising transitions.
- 🔴 **VIB latent noise:** with `policy.use_vib=True` and `ppo.force_stochastic_online=True`, `train_real.py` keeps the observation encoder stochastic during `eval(..., data_collect=True)`.

Typical real-robot collection settings:

```bash
data_collect=True
policy.use_vib=True
ppo.force_stochastic_online=True
```

Use `offline_cp_timestamp` / `offline_cp_timestep` or the corresponding checkpoint-loading fields in the online launcher to point to the selected offline RL checkpoint. The generated rollout `.h5` files should then be added as a new `rollout_sources` entry in `tools/teleop_off2off_data/configs/data_prepare.yaml`.

### Online Real-Robot Fine-Tuning

This section covers the final online RL stage:

- Set `data_collect=False` before starting actual online RL fine-tuning. `data_collect=True` is only for iterative offline rollout dataset collection; leaving it enabled during online fine-tuning will make periodic eval calls run in collection mode and write rollout `.h5` files.
- Start from a BC or offline RL checkpoint.
- Collect short online rollouts on the real robot.
- Run online policy-gradient updates.
- Evaluate and save deployment checkpoints.

### Safety Notes

> ⚠ Before running policies on hardware:

- Verify workspace bounds and action scaling.
- Test controller latency and emergency stop.
- Start with low speed and conservative action limits.
- Run short supervised rollouts before autonomous long runs.
- **Do not** deploy a checkpoint on hardware before validating observation normalization, action dimensions, and reset behavior.

---

## Installation

See [`INSTALL.md`](INSTALL.md) for the full environment setup. We recommend using `adroit_door_medium` as the first smoke-test task.

**The verified server setup uses:**

<table width="100%">
<thead>
<tr>
  <th align="left">Component</th>
  <th align="left">Version</th>
</tr>
</thead>
<tbody>
<tr><td>NVIDIA driver</td><td><code>550.54.15</code></td></tr>
<tr><td>System CUDA</td><td><code>12.4</code></td></tr>
<tr><td>Python</td><td><code>3.8.20</code></td></tr>
<tr><td>PyTorch</td><td><code>2.4.0+cu121</code></td></tr>
</tbody>
</table>

**Minimal workflow:**

```bash
conda create -n rl100 --clone dp3 -y
conda activate rl100

export REPO_ROOT=$(pwd)
export PYTHONPATH=${REPO_ROOT}/RL-100:${PYTHONPATH}
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -m pip install -e third_party/dexart-release
python -m pip install -e third_party/gym-0.21.0
python -m pip install -e third_party/Metaworld
python -m pip install -e third_party/rrl-dependencies/mj_envs/.
python -m pip install -e third_party/rrl-dependencies/mjrl/.
python -m pip install -e third_party/mujoco-py-2.1.2.14
python -m pip install -e third_party/pytorch3d_simplified
python -m pip install -e visualizer
```

**Sanity check:**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import rl_100, zarr, hydra, einops, metaworld, mujoco_py, open3d as o3d; print('imports ok', o3d.__version__)"
```

**Verified flow online distillation entry:**

```bash
./scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh rl100 adroit_door_medium 0112 100 8
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{rl100,
  title  = {RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning},
  author = {Lei, Kun and Li, Huanyu and Yu, Dongjie and Wei, Zhenyu and Guo, Lingxiao and Jiang, Zhennan and Wang, Ziyu and Liang, Shiyu and Xu, Huazhe},
  journal = {arXiv preprint arXiv:2510.14830},
  year   = {2025}
}
```

<details>
<summary><b>Also consider citing the related RL post-training methods used by this project</b></summary>

```bibtex
@inproceedings{lei2024unio,
  title     = {Uni-O4: Unifying Online and Offline Deep Reinforcement Learning with Multi-Step On-Policy Optimization},
  author    = {Kun LEI and Zhengmao He and Chenhao Lu and Kaizhe Hu and Yang Gao and Huazhe Xu},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=tbFBh3LMKi}
}

@inproceedings{zhuang2023behavior,
  title     = {Behavior Proximal Policy Optimization},
  author    = {Zifeng Zhuang and Kun LEI and Jinxin Liu and Donglin Wang and Yilang Guo},
  booktitle = {The Eleventh International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=3c13LptpIph}
}
```

</details>

---

## Acknowledgements

RL-100 builds on prior work in **Uni-O4**, **BPPO**, **DP3**, **Diffusion Policy**, robotic manipulation environments, and reinforcement learning infrastructure. Detailed third-party acknowledgements and license notes will be maintained with the corresponding code and dependencies.

---

## License

This project is released under the [Apache License 2.0](LICENSE).

<div align="center">
<sub>Made with care by the RL-100 team.</sub>
</div>
