# Flexiv 双臂三视角 DP30 接入

本文说明 RL-100 fork 中新增的 Flexiv 双臂、两只 Inspire 六维手、三视角 RGB
接入。它与 `isaac_teleop_flexiv_inspire` 的职责严格分开：采集仓是唯一硬件 owner，
RL-100 只读取派生 Zarr，或通过受保护的 Policy RPC 做推理。

当前目标是先跑通 2D Diffusion Policy 行为克隆（BC）。离线 RL 的数据入口和奖励
门控已经加入，但任务奖励和自动 reset 不可能从遥操作数据中凭空得到；在线 RL
需要独立完成环境实现与真机验收。

## 1. 新增内容

- `rl_100/dataset/flexiv_dual_rgb_dataset.py`：严格校验并懒加载 Flexiv Zarr；
- `rl_100/config/task/flexiv_dual_rgb.yaml`：三路 RGB、26D 状态、24D 动作；
- `rl_100/config/rl100_flexiv_dp30_bc.yaml`：单卡 2D DP 行为克隆配置；
- `rl_100/real_world/flexiv_dp30.py`：Zarr 校验、checkpoint 离线回放、RPC 影子/执行；
- `train.py`：修复 `use_wandb=false`，并让 `only_bc=true` 在 BC 后真正停止，不再
  误进入 critic、dynamics 和 RL 阶段；
- 单元测试覆盖懒加载、维度、奖励门控和离线 RL transition 生成。

## 2. 数据契约

输入必须是采集仓导出的 Zarr v2，根属性为：

```text
schema_id = flexiv_rl100_dp_rgb_v1
profile   = joint_proprio_cartesian_v1
fps       = 30.0
```

BC 必需数组：

| 路径 | shape | 语义 |
|---|---|---|
| `data/state` | `[N,26]` | 双臂 q14 + 双手测量状态 12 |
| `data/action` | `[N,24]` | 双臂末端 delta rotvec 12 + 双手绝对目标 12 |
| `data/rgb_head` | `[N,3,H,W]` | 头部 RGB |
| `data/rgb_left_wrist` | `[N,3,H,W]` | 左腕 RGB |
| `data/rgb_right_wrist` | `[N,3,H,W]` | 右腕 RGB |
| `meta/episode_ends` | `[E]` | episode 累积结束下标 |

图像以 `uint8` 保存在 Zarr，dataset 按需读取并转成 `float32`；RL-100 的
`MultiImageObsEncoder` 自动把 `[0,255]` 转成 `[0,1]`，再做 ImageNet normalize。
三视角使用共享的 ResNet18 权重，以减少显存和参数量；当前配置从随机权重训练，
不会在启动时自动下载模型。

模型使用最近两个观测，输出 8 个 30 Hz 动作位置，约覆盖 0.267 s。默认每 4 个
控制 tick 重新规划一次；历史序列和 episode padding 由现有 `SequenceSampler`
处理，不允许跨 episode 取样。

## 3. 环境

5070 机器使用已有 Conda 环境：

```bash
conda activate RL100
export RL100_REPO=/home/hb/chp_ws/rl100_dp30/RL-100
cd "$RL100_REPO/RL-100"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

仅做 Zarr 训练不需要 ROS。影子或真实 RPC 推理还需要安装采集仓中的两个便携包：

```bash
python -m pip install -e \
  /home/hb/chp_ws/rl100_dp30/isaac_teleop_flexiv_inspire/libs/policy_contracts
python -m pip install -e \
  /home/hb/chp_ws/rl100_dp30/isaac_teleop_flexiv_inspire/libs/policy_client
```

这两个包只提供 schema、24D/30D 映射和 RPC client，不会直接访问机器人硬件。

## 4. 训练前校验

先验证 Zarr，不加载网络：

```bash
python -m rl_100.real_world.flexiv_dp30 validate \
  --zarr /absolute/path/to/joint_proprio_cartesian_v1.zarr
```

输出应明确显示 30 Hz、26D、24D、三个图像 shape、frame/episode 数。若要作为
离线 RL 数据检查，还要加：

```bash
python -m rl_100.real_world.flexiv_dp30 validate \
  --zarr /absolute/path/to/joint_proprio_cartesian_v1.zarr \
  --require-offline-rl
```

后者要求存在 `reward`、`done`、`return` 且 `offline_rl_ready=true`；普通示教 Zarr
会按设计失败。

建议在正式训练前做一次加载冒烟测试：

```bash
python - <<'PY'
from rl_100.dataset.flexiv_dual_rgb_dataset import FlexivDualRGBDataset

dataset = FlexivDualRGBDataset(
    "/absolute/path/to/joint_proprio_cartesian_v1.zarr",
    horizon=9,
    pad_before=1,
    pad_after=7,
)
sample = dataset[0]
print(len(dataset), sample["obs"]["agent_pos"].shape, sample["action"].shape)
PY
```

预期单样本状态为 `[9,26]`，动作为 `[9,24]`。训练时 policy 会根据
`n_obs_steps=2` 和 `n_action_steps=8` 使用所需部分。

## 5. 2D DP 行为克隆

在仓库内层目录运行：

```bash
cd /home/hb/chp_ws/rl100_dp30/RL-100/RL-100
conda activate RL100
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python train.py \
  --config-name=rl100_flexiv_dp30_bc.yaml \
  task.dataset.zarr_path=/absolute/path/to/joint_proprio_cartesian_v1.zarr
```

默认配置：

- `RL1002D` + DDIM diffusion；
- 三路 RGB，resize 到 224 x 224；
- 26D agent position 和 24D action；
- batch size 16、100 epochs、EMA；
- 不用点云、不用触觉、不连接环境、不启用 WandB；
- `only_bc=true`，BC 完成后保存并退出。

输出目录在：

```text
data/outputs/flexiv_dual_rgb_dp30/YYYY.MM.DD/HH.MM.SS/
```

其中 `checkpoints/latest.ckpt` 包含配置、模型、EMA 和 normalizer，可供本文件后续
命令直接加载；`bc_final/` 是最终策略自身的保存目录。首次正式训练前，可用命令行
覆盖 `training.num_epochs=1 training.max_train_steps=2 dataloader.num_workers=0`
做小规模冒烟测试。

## 6. checkpoint 离线回放

该模式只读 Zarr，不连接硬件。它逐帧构造历史观测，执行策略推理，并报告预测块
第一步与示教动作的 MSE：

```bash
python -m rl_100.real_world.flexiv_dp30 replay \
  --zarr /absolute/path/to/joint_proprio_cartesian_v1.zarr \
  --checkpoint /absolute/path/to/checkpoints/latest.ckpt \
  --device cuda:0 \
  --max-steps 32
```

默认加载 EMA；需要检查原模型时加 `--no-ema`。这里的 MSE 只是数据/模型链路冒烟
指标，不等价于任务成功率。命令还会拒绝 NaN/Inf、非 `[N,24]` 输出和越界的
`[0,1]` 手目标。

## 7. RPC 影子模式

硬件端在采集仓启动只读 server。为避免自动 Home，第一次只做影子验证时使用：

```bash
cd /home/hb/chp_ws/rl100_dp30/isaac_teleop_flexiv_inspire
source scripts/env/activate_ros.sh
robot --config config/site_rl100_dp30.yaml policy-serve --no-reset
```

RL-100 端默认 `live` 不发送任何动作：

```bash
python -m rl_100.real_world.flexiv_dp30 live \
  --checkpoint /absolute/path/to/checkpoints/latest.ckpt \
  --target 127.0.0.1:50051 \
  --insecure-loopback \
  --max-ticks 300
```

影子模式仍严格检查 RPC schema、session、状态/手/图像新鲜度和三路图像 shape，
但不会申请控制 lease，也不会发 action。非 loopback 必须使用 server CA；如 server
要求双向 TLS，还要提供 client certificate 和 key。

## 8. 真实执行安全门

真实执行代码已经接通，但不会因运行普通 `live` 命令而启用。必须同时给出：

```text
--execute --confirm FLEXIV-RL100-EXECUTE
```

这只是客户端的第一层显式确认。硬件 server 仍要求：

- 现场 Reset、F/T 清零和 Home 已完成；
- 当前硬件 session 和本地授权有效；
- 物理中踏板持续按下；
- policy 独占 lease、deadman、动作 TTL 和 acknowledgment 正常；
- 双臂/双手状态健康，动作有限，Rotation-6D 可构造，手目标在范围内；
- 断连、超时、踏板释放或客户端退出立即 hold。

在完成方向、坐标系、单位和限幅验收前不要使用上述执行参数。验收顺序应为：影子
稳定运行 -> 单臂微小平移 -> 单臂微小旋转 -> 双臂 -> 双手 -> 动作块 -> 故障注入。

## 9. 离线 RL 与在线 RL 的真实边界

采集仓可给每个导出 episode 添加完整成功/失败标签，生成 terminal sparse reward、
done 和 return。本 dataset 设置 `return_transitions=true` 后会额外返回 next_obs、
next_action、reward、not_done、return，并在标签不完整时 fail closed。

但当前 `rl100_flexiv_dp30_bc.yaml` 有意只运行 BC。要宣称离线 RL 完整可用，还需：

1. 确认 RL-100 所选 critic/dynamics 算法与 24D chunk 语义；
2. 为任务定义可信 reward，而不只是把所有示教标为成功；
3. 建立独立 offline 配置并进行数值、回报和 checkpoint 验证；
4. 用离线评估和小范围真机 rollout 检查策略退化。

在线 RL 还必须实现任务环境：自动或人工 reset、在线成功检测、episode timeout、
失败恢复、在线数据回写、策略版本管理和现场看护。现有 Policy RPC 解决的是观测与
安全动作传输，不会自动补齐这些任务语义。

因此当前可准确表述为：数采、三视角 DP Zarr、BC 训练、离线推理和影子闭环已经
接通；离线 RL 具备显式标注的数据入口；真机策略运动和在线 RL 仍需现场分阶段验收。
