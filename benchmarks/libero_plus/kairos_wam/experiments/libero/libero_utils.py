"""Utils for evaluating policies in LIBERO simulation environments."""

import logging
import math
import os
import time
import pathlib
from contextlib import contextmanager

import imageio
from PIL import Image, ImageDraw
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from kairoswam.utils.video_io import save_mp4

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _egl_init_lock():
    if not _env_truthy("LIBERO_EGL_INIT_LOCK", default=True):
        yield
        return

    import fcntl

    lock_file = os.environ.get("LIBERO_EGL_INIT_LOCK_FILE", "/tmp/libero_mujoco_egl_init.lock")
    pathlib.Path(lock_file).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _create_offscreen_env_with_retry(env_args: dict):
    max_attempts = max(1, int(os.environ.get("LIBERO_EGL_INIT_RETRIES", "5")))
    retry_sleep = float(os.environ.get("LIBERO_EGL_INIT_RETRY_SLEEP_SEC", "5"))
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            with _egl_init_lock():
                return OffScreenRenderEnv(**env_args)
        except RuntimeError as exc:
            last_exc = exc
            msg = str(exc)
            is_egl_device_error = (
                "MUJOCO_EGL_DEVICE_ID environment variable" in msg
                or "Cannot initialize a EGL device display" in msg
            )
            if not is_egl_device_error or attempt == max_attempts:
                raise
            logging.warning(
                "LIBERO EGL init failed on attempt %s/%s; retrying in %.1fs: %s",
                attempt,
                max_attempts,
                retry_sleep,
                exc,
            )
            time.sleep(retry_sleep)

    raise RuntimeError("LIBERO EGL init failed") from last_exc


def get_libero_env(task, resolution, seed, env_num=1):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = str(
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if env_num > 1:
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    else:
        env = _create_offscreen_env_with_retry(env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def close_libero_env(env) -> None:
    """Release MuJoCo / EGL rendering resources explicitly (robosuite-recommended)."""
    if env is None:
        return
    try:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception as exc:
        logging.warning("Failed to close LIBERO env cleanly: %s", exc)

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    
    # [yc] wrist image
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    
    return {
        "image": img,
        "wrist_image": wrist_img
    }

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, log_file=None, fps=24):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if isinstance(img, dict):
            image = []
            for key, value in img.items():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                pil_img = Image.fromarray(value_array)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"{key}", fill=(255, 255, 255))
                image.append(np.array(pil_img))
            frame = np.concatenate(image, axis=1)
        elif isinstance(img, Image.Image):
            frame = np.array(img.convert("RGB"))
        else:
            frame = np.array(img)
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_prediction_video(
    rollout_dir,
    gt_frames,
    pred_frames,
    idx,
    replan_idx,
    success,
    task_description,
    log_file=None,
    fps=8,
):
    """Saves an MP4 comparison of ground-truth and predicted future frames for one replanning clip."""
    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames <= 0:
        raise ValueError("Cannot save prediction video with empty GT/pred frame lists.")

    stitched_frames = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        if isinstance(gt_frame, dict):
            gt_images = []
            for value in gt_frame.values():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                gt_images.append(value_array)
            gt_image = np.concatenate(gt_images, axis=1)
        elif isinstance(gt_frame, Image.Image):
            gt_image = np.array(gt_frame.convert("RGB"))
        else:
            gt_image = np.array(gt_frame)

        if isinstance(pred_frame, Image.Image):
            pred_image = np.array(pred_frame.convert("RGB"))
        else:
            pred_image = np.array(pred_frame)

        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_pil = Image.fromarray(gt_image)
        ImageDraw.Draw(gt_pil).text((10, 10), "gt", fill=(255, 255, 255))
        pred_pil = Image.fromarray(pred_image)
        ImageDraw.Draw(pred_pil).text((10, 10), "pred", fill=(255, 255, 255))
        stitched_frames.append(
            Image.fromarray(np.concatenate([np.array(pred_pil), np.array(gt_pil)], axis=0))
        )

    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    try:
        replan_tag = f"{int(replan_idx):04d}"
    except (TypeError, ValueError):
        replan_tag = str(replan_idx)
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}"
        f"--task={processed_task_description}--replan={replan_tag}--gt-pred.mp4"
    )
    save_mp4(stitched_frames, mp4_path, fps=fps)
    print(f"Saved predicted future comparison MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved predicted future comparison MP4 at path {mp4_path}\n")
    return mp4_path

def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = (v > 0.5)
    return np.asarray(bin_val, dtype=np.float32)


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
