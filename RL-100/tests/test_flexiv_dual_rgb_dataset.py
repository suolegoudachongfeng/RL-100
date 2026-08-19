from __future__ import annotations

import numpy as np
import pytest
import zarr

from rl_100.dataset.flexiv_dual_rgb_dataset import FlexivDualRGBDataset
from rl_100.real_world.flexiv_dp30 import validate_zarr


def _dataset(path, *, with_rewards=False):
    root = zarr.group(store=zarr.DirectoryStore(str(path)), overwrite=True)
    root.attrs.update(
        {
            "schema_id": "flexiv_rl100_dp_rgb_v1",
            "profile": "joint_proprio_cartesian_v1",
            "fps": 30.0,
            "image_layout": "CHW",
            "offline_rl_ready": bool(with_rewards),
        }
    )
    data = root.create_group("data")
    meta = root.create_group("meta")
    count = 24
    data.array("state", np.random.randn(count, 26).astype(np.float32))
    actions = np.random.randn(count, 24).astype(np.float32)
    actions[:, 12:24] = np.random.rand(count, 12)
    data.array("action", actions)
    for key in ("rgb_head", "rgb_left_wrist", "rgb_right_wrist"):
        data.array(key, np.zeros((count, 3, 12, 16), dtype=np.uint8))
    meta.array("episode_ends", np.asarray([12, 24], dtype=np.int64))
    if with_rewards:
        reward = np.zeros((count, 1), dtype=np.float32)
        reward[[11, 23]] = 1.0
        done = np.zeros((count, 1), dtype=np.float32)
        done[[11, 23]] = 1.0
        data.array("reward", reward)
        data.array("done", done)
        data.array("return", reward.copy())
    return root


def test_bc_dataset_reads_core_arrays_lazily(tmp_path):
    path = tmp_path / "demo.zarr"
    _dataset(path)

    dataset = FlexivDualRGBDataset(
        path,
        horizon=9,
        pad_before=1,
        pad_after=7,
        val_ratio=0.5,
        load_to_memory=False,
    )
    sample = dataset[0]

    assert sample["obs"]["agent_pos"].shape == (9, 26)
    assert sample["obs"]["rgb_head"].shape == (9, 3, 12, 16)
    assert sample["action"].shape == (9, 24)
    assert "next_obs" not in sample
    assert dataset.replay_buffer.backend == "zarr"


def test_offline_rl_mode_rejects_unlabelled_demonstrations(tmp_path):
    path = tmp_path / "demo.zarr"
    _dataset(path)

    with pytest.raises(ValueError, match="explicit reward labels"):
        FlexivDualRGBDataset(path, return_transitions=True)


def test_offline_rl_mode_builds_next_transitions_after_labelling(tmp_path):
    path = tmp_path / "demo.zarr"
    _dataset(path, with_rewards=True)

    dataset = FlexivDualRGBDataset(
        path,
        horizon=9,
        return_transitions=True,
        val_ratio=0.0,
    )
    sample = dataset[0]

    assert sample["next_obs"]["agent_pos"].shape == (9, 26)
    assert sample["next_action"].shape == (9, 24)
    assert sample["reward"].shape == (9, 1)
    assert sample["not_done"].shape == (9, 1)


def test_schema_validator_reports_bc_and_offline_readiness(tmp_path):
    path = tmp_path / "demo.zarr"
    _dataset(path, with_rewards=True)

    report = validate_zarr(path, require_offline_rl=True)

    assert report["frames"] == 24
    assert report["episodes"] == 2
    assert report["offline_rl_ready"] is True
