"""Offline replay and fail-closed live runner for the Flexiv DP30 profile."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import dill
import hydra
import numpy as np
import torch
import zarr


SCHEMA_ID = "flexiv_rl100_dp_rgb_v1"
PROFILE_ID = "joint_proprio_cartesian_v1"
IMAGE_KEYS = ("rgb_head", "rgb_left_wrist", "rgb_right_wrist")
LIVE_IMAGE_KEYS = {
    "rgb_head": "observation.images.head_image",
    "rgb_left_wrist": "observation.images.left_wrist_image",
    "rgb_right_wrist": "observation.images.right_wrist_image",
}
EXECUTE_CONFIRMATION = "FLEXIV-RL100-EXECUTE"


def validate_zarr(path: str | Path, *, require_offline_rl: bool = False) -> dict:
    source = Path(path).expanduser().resolve(strict=True)
    root = zarr.open(str(source), mode="r")
    errors: list[str] = []
    if str(root.attrs.get("schema_id", "")) != SCHEMA_ID:
        errors.append(f"schema_id must be {SCHEMA_ID}")
    if str(root.attrs.get("profile", "")) != PROFILE_ID:
        errors.append(f"profile must be {PROFILE_ID}")
    if float(root.attrs.get("fps", 0.0)) != 30.0:
        errors.append("fps must be 30")
    expected_shapes = {"state": (26,), "action": (24,)}
    for key, tail in expected_shapes.items():
        if f"data/{key}" not in root or root[f"data/{key}"].shape[1:] != tail:
            errors.append(f"data/{key} must have shape [N,{tail[0]}]")
    for key in IMAGE_KEYS:
        if f"data/{key}" not in root:
            errors.append(f"data/{key} is missing")
            continue
        shape = root[f"data/{key}"].shape
        if len(shape) != 4 or shape[1] != 3:
            errors.append(f"data/{key} must have shape [N,3,H,W]")
    if "meta/episode_ends" not in root:
        errors.append("meta/episode_ends is missing")
        episode_ends = np.zeros(0, dtype=np.int64)
    else:
        episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        count = int(root["data/state"].shape[0]) if "data/state" in root else 0
        if not len(episode_ends) or int(episode_ends[-1]) != count:
            errors.append("episode_ends does not terminate at the final frame")
    offline_ready = bool(root.attrs.get("offline_rl_ready", False))
    if require_offline_rl:
        for key in ("reward", "done", "return"):
            if f"data/{key}" not in root:
                errors.append(f"offline RL requires data/{key}")
        if not offline_ready:
            errors.append("offline_rl_ready attribute is false")
    if errors:
        raise ValueError("invalid Flexiv RL-100 Zarr: " + "; ".join(errors))
    return {
        "path": str(source),
        "frames": int(root["data/state"].shape[0]),
        "episodes": int(len(episode_ends)),
        "fps": 30.0,
        "state_dimension": 26,
        "action_dimension": 24,
        "image_shapes": {key: list(root[f"data/{key}"].shape[1:]) for key in IMAGE_KEYS},
        "offline_rl_ready": offline_ready,
    }


def load_policy_checkpoint(
    checkpoint: str | Path,
    *,
    device: str,
    use_ema: bool = True,
):
    path = Path(checkpoint).expanduser().resolve(strict=True)
    payload = torch.load(
        path.open("rb"),
        map_location="cpu",
        pickle_module=dill,
        weights_only=False,
    )
    cfg = payload.get("cfg")
    states = payload.get("state_dicts", {})
    if cfg is None or "model" not in states:
        raise ValueError("checkpoint does not contain cfg and model state")
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if use_ema and states.get("ema_model") is not None else "model"
    policy.load_state_dict(states[state_key])
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg, state_key


def _zarr_frame(root, index: int) -> dict[str, np.ndarray]:
    return {
        "state": np.asarray(root["data/state"][index], dtype=np.float32),
        **{
            key: np.asarray(root[f"data/{key}"][index], dtype=np.float32)
            for key in IMAGE_KEYS
        },
    }


def _live_frame(frame: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {
        "state": np.asarray(frame["observation.state"], dtype=np.float32),
    }
    for target, source in LIVE_IMAGE_KEYS.items():
        image = np.asarray(frame[source])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"live {source} must be HWC RGB")
        result[target] = np.ascontiguousarray(image.transpose(2, 0, 1)).astype(
            np.float32
        )
    return result


def _model_observation(
    history: Sequence[Mapping[str, np.ndarray]], *, device: str
) -> dict[str, torch.Tensor]:
    return {
        "agent_pos": torch.from_numpy(
            np.stack([item["state"] for item in history])
        ).unsqueeze(0).to(device),
        **{
            key: torch.from_numpy(np.stack([item[key] for item in history]))
            .unsqueeze(0)
            .to(device)
            for key in IMAGE_KEYS
        },
    }


@torch.no_grad()
def infer_action_chunk(
    policy,
    history: Sequence[Mapping[str, np.ndarray]],
    *,
    device: str,
    deterministic: bool = True,
) -> np.ndarray:
    result = policy.predict_action(
        _model_observation(history, device=device), deterministic=deterministic
    )
    action = result["action"][0].detach().cpu().numpy().astype(np.float32)
    if action.ndim != 2 or action.shape[1] != 24 or not np.all(np.isfinite(action)):
        raise ValueError(f"policy returned invalid action chunk shape {action.shape}")
    hands = action[:, 12:24]
    if np.any(hands < 0.0) or np.any(hands > 1.0):
        raise ValueError("policy hand targets left the trained [0,1] domain")
    return action


def replay_checkpoint(
    zarr_path: str | Path,
    checkpoint: str | Path,
    *,
    device: str,
    max_steps: int = 32,
    use_ema: bool = True,
) -> dict:
    summary = validate_zarr(zarr_path)
    root = zarr.open(str(Path(zarr_path).expanduser()), mode="r")
    policy, cfg, state_key = load_policy_checkpoint(
        checkpoint, device=device, use_ema=use_ema
    )
    n_obs_steps = int(cfg.n_obs_steps)
    history: deque[dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)
    errors = []
    evaluated = 0
    count = min(summary["frames"], max(1, int(max_steps)))
    for index in range(count):
        frame = _zarr_frame(root, index)
        if not history:
            for _ in range(n_obs_steps - 1):
                history.append(frame)
        history.append(frame)
        prediction = infer_action_chunk(policy, history, device=device)
        target = np.asarray(root["data/action"][index], dtype=np.float32)
        errors.append(float(np.mean((prediction[0] - target) ** 2)))
        evaluated += 1
    return {
        **summary,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "checkpoint_state": state_key,
        "evaluated_steps": evaluated,
        "first_action_mse": float(np.mean(errors)),
    }


def run_live(
    *,
    checkpoint: str | Path,
    target: str,
    device: str,
    server_ca: str = "",
    client_cert: str = "",
    client_key: str = "",
    insecure_loopback: bool = False,
    execute: bool = False,
    confirmation: str = "",
    max_ticks: int = 300,
    replan_steps: int = 4,
    action_ttl_ms: float = 300.0,
) -> dict:
    from policy_runtime_client import SyncPolicyActionClient, SyncPolicyProfileClient

    if execute and confirmation != EXECUTE_CONFIRMATION:
        raise ValueError(
            f"live execution requires --confirm {EXECUTE_CONFIRMATION}"
        )
    policy, cfg, state_key = load_policy_checkpoint(checkpoint, device=device)
    n_obs_steps = int(cfg.n_obs_steps)
    client = SyncPolicyProfileClient(
        target=target,
        profile_id=PROFILE_ID,
        insecure_loopback=insecure_loopback,
        server_ca=server_ca or None,
        client_cert=client_cert or None,
        client_key=client_key or None,
        state_max_age_ms=50.0,
        hand_max_age_ms=100.0,
        image_max_age_ms=75.0,
        snapshot_ready_timeout_s=2.0,
    )
    client.connect()
    action_client = (
        SyncPolicyActionClient(
            client,
            client_id="rl100-flexiv-dp30",
            action_ttl_ms=action_ttl_ms,
            requested_lease_ms=2_000,
        )
        if execute
        else None
    )
    history: deque[dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)
    period_ns = int(round(1e9 / 30.0))
    deadline = time.monotonic_ns()
    inferred = 0
    sent_chunks = 0
    tick = 0
    try:
        while max_ticks <= 0 or tick < max_ticks:
            current = _live_frame(client.get_frame())
            if not history:
                for _ in range(n_obs_steps - 1):
                    history.append(current)
            history.append(current)
            if tick % max(1, int(replan_steps)) == 0:
                actions = infer_action_chunk(policy, history, device=device)
                inferred += 1
                if action_client is not None:
                    max_points = min(
                        len(actions),
                        max(1, int((action_client.action_ttl_ns - 1) // period_ns) + 1),
                    )
                    offsets = tuple(index * period_ns for index in range(max_points))
                    action_client.send_chunk(actions[:max_points], execute_after_ns=offsets)
                    sent_chunks += 1
            tick += 1
            deadline += period_ns
            delay_ns = deadline - time.monotonic_ns()
            if delay_ns > 0:
                time.sleep(delay_ns / 1e9)
    finally:
        if action_client is not None:
            action_client.close(reason="rl100-runner-exit")
        client.close()
    return {
        "mode": "execute" if execute else "shadow",
        "ticks": tick,
        "inference_calls": inferred,
        "sent_chunks": sent_chunks,
        "checkpoint_state": state_key,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Flexiv RL-100 DP30 tools")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--zarr", required=True)
    validate.add_argument("--require-offline-rl", action="store_true")
    replay = sub.add_parser("replay")
    replay.add_argument("--zarr", required=True)
    replay.add_argument("--checkpoint", required=True)
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument("--max-steps", type=int, default=32)
    replay.add_argument("--no-ema", action="store_true")
    live = sub.add_parser("live")
    live.add_argument("--checkpoint", required=True)
    live.add_argument("--target", default="127.0.0.1:50051")
    live.add_argument("--device", default="cuda:0")
    live.add_argument("--server-ca", default="")
    live.add_argument("--client-cert", default="")
    live.add_argument("--client-key", default="")
    live.add_argument("--insecure-loopback", action="store_true")
    live.add_argument("--execute", action="store_true")
    live.add_argument("--confirm", default="")
    live.add_argument("--max-ticks", type=int, default=300)
    live.add_argument("--replan-steps", type=int, default=4)
    live.add_argument("--action-ttl-ms", type=float, default=300.0)
    args = parser.parse_args(argv)

    if args.command == "validate":
        result = validate_zarr(args.zarr, require_offline_rl=args.require_offline_rl)
    elif args.command == "replay":
        result = replay_checkpoint(
            args.zarr,
            args.checkpoint,
            device=args.device,
            max_steps=args.max_steps,
            use_ema=not args.no_ema,
        )
    else:
        result = run_live(
            checkpoint=args.checkpoint,
            target=args.target,
            device=args.device,
            server_ca=args.server_ca,
            client_cert=args.client_cert,
            client_key=args.client_key,
            insecure_loopback=args.insecure_loopback,
            execute=args.execute,
            confirmation=args.confirm,
            max_ticks=args.max_ticks,
            replan_steps=args.replan_steps,
            action_ttl_ms=args.action_ttl_ms,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
