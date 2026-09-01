"""RL-100 dataset for the Flexiv dual-arm, dual-hand, three-RGB profile."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import zarr

from rl_100.common.pytorch_util import dict_apply
from rl_100.common.replay_buffer import ReplayBuffer
from rl_100.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.model.common.normalizer import LinearNormalizer


SCHEMA_ID = "flexiv_rl100_dp_rgb_v1"
IMAGE_KEYS = ("rgb_head", "rgb_left_wrist", "rgb_right_wrist")
PROFILE_CONTRACTS = {
    "joint_proprio_cartesian_v1": (SCHEMA_ID, 26, 24, IMAGE_KEYS),
    "right_joint_proprio_cartesian_v1": (
        "flexiv_rl100_right_dp_rgb_v1",
        13,
        12,
        ("rgb_head", "rgb_right_wrist"),
    ),
}


class FlexivDualRGBDataset(BaseDataset):
    """Lazy Zarr dataset for the registered dual- or right-arm RGB contract."""

    def __init__(
        self,
        zarr_path,
        horizon=9,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=None,
        sequence_stride=1,
        load_to_memory=False,
        return_transitions=False,
        profile="joint_proprio_cartesian_v1",
    ):
        super().__init__()
        try:
            self.schema_id, self.state_dim, self.action_dim, self.image_keys = (
                PROFILE_CONTRACTS[str(profile)]
            )
        except KeyError as exc:
            raise ValueError(f"unsupported Flexiv dataset profile: {profile}") from exc
        self.profile = str(profile)
        self.core_keys = ("state", "action", *self.image_keys)
        self.zarr_path = str(Path(zarr_path).expanduser())
        source = zarr.open(self.zarr_path, mode="r")
        self._validate_source(source, return_transitions=return_transitions)

        keys = list(self.core_keys)
        if return_transitions:
            keys.extend(("reward", "done", "return"))
        self.replay_buffer = (
            ReplayBuffer.copy_from_path(self.zarr_path, keys=keys)
            if load_to_memory
            else ReplayBuffer.create_from_path(self.zarr_path, mode="r")
        )
        self.return_transitions = bool(return_transitions)
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.sequence_stride = int(sequence_stride)
        self._sequence_length = self.horizon + (1 if self.return_transitions else 0)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        self.sampler = self._make_sampler(train_mask)
        self.train_mask = train_mask
        if len(self.sampler) == 0:
            raise ValueError("Flexiv RL-100 dataset contains no trainable sequences")

    def _validate_source(self, root, *, return_transitions: bool) -> None:
        if str(root.attrs.get("schema_id", "")) != self.schema_id:
            raise ValueError(f"expected Zarr schema_id={self.schema_id}")
        if str(root.attrs.get("profile", "")) != self.profile:
            raise ValueError(f"expected {self.profile} profile")
        if float(root.attrs.get("fps", 0.0)) != 30.0:
            raise ValueError("Flexiv DP dataset must use a 30 Hz timeline")
        if str(root.attrs.get("image_layout", "")) != "CHW":
            raise ValueError("Flexiv DP images must use CHW layout")
        if "data" not in root or "meta/episode_ends" not in root:
            raise ValueError("Zarr must contain data and meta/episode_ends")
        missing = [key for key in self.core_keys if f"data/{key}" not in root]
        if missing:
            raise ValueError(f"Zarr is missing required arrays: {missing}")
        if root["data/state"].shape[1:] != (self.state_dim,):
            raise ValueError(f"state must have shape [N,{self.state_dim}]")
        if root["data/action"].shape[1:] != (self.action_dim,):
            raise ValueError(f"action must have shape [N,{self.action_dim}]")
        for key in self.image_keys:
            shape = root[f"data/{key}"].shape
            if len(shape) != 4 or shape[1] != 3:
                raise ValueError(f"{key} must have shape [N,3,H,W]")
        if return_transitions:
            missing_rl = [
                key for key in ("reward", "done", "return") if f"data/{key}" not in root
            ]
            if missing_rl:
                raise ValueError(
                    "offline RL requires explicit reward labels; missing arrays: "
                    f"{missing_rl}"
                )
            if not bool(root.attrs.get("offline_rl_ready", False)):
                raise ValueError("Zarr reward labels have not been validated for offline RL")

    def _make_sampler(self, episode_mask) -> SequenceSampler:
        keys = list(self.core_keys)
        if self.return_transitions:
            keys.extend(("reward", "done", "return"))
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self._sequence_length,
            pad_before=self.pad_before,
            pad_after=self.pad_after + (1 if self.return_transitions else 0),
            episode_mask=episode_mask,
            sequence_stride=self.sequence_stride,
            keys=keys,
        )

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = result._make_sampler(~self.train_mask)
        result.train_mask = ~self.train_mask
        return result

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        if self.return_transitions:
            data["next_action"] = self.replay_buffer["action"]
            data["next_agent_pos"] = self.replay_buffer["state"]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def get_length(self) -> int:
        return len(self.sampler)

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.replay_buffer["action"][:]))

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> dict:
        stop = self.horizon
        obs = {
            "agent_pos": sample["state"][:stop].astype(np.float32),
            **{
                key: sample[key][:stop].astype(np.float32)
                for key in self.image_keys
            },
        }
        result = {
            "obs": obs,
            "action": sample["action"][:stop].astype(np.float32),
        }
        if self.return_transitions:
            result.update(
                {
                    "next_obs": {
                        "agent_pos": sample["state"][1 : stop + 1].astype(np.float32),
                        **{
                            key: sample[key][1 : stop + 1].astype(np.float32)
                            for key in self.image_keys
                        },
                    },
                    "next_action": sample["action"][1 : stop + 1].astype(np.float32),
                    "reward": sample["reward"][:stop].astype(np.float32),
                    "not_done": 1.0 - sample["done"][:stop].astype(np.float32),
                    "return": sample["return"][:stop].astype(np.float32),
                }
            )
        return result

    def get_shape_info(self, n_action_steps, n_obs_steps):
        sample = self.sampler.sample_sequence(0)
        return {
            "obs": {
                "agent_pos": (n_obs_steps, self.state_dim),
                **{
                    key: (n_obs_steps, *sample[key].shape[1:])
                    for key in self.image_keys
                },
            },
            "action": (n_action_steps, self.action_dim),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._sample_to_data(self.sampler.sample_sequence(idx))
        return dict_apply(data, torch.from_numpy)
