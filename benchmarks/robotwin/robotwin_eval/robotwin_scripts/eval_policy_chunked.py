import sys
import os
import subprocess
import json
import contextlib
sys.path.append("./")
sys.path.append("./script")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
import gc
import numpy as np
from pathlib import Path
from collections import deque
import traceback
import time
import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _robotwin_root() -> Path:
    return Path(os.environ.get("ROBOTWIN_ROOT", os.getcwd())).expanduser().resolve()


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = _robotwin_root() / "task_config" / "_camera_config.yml"

    assert camera_config_path.is_file(), "task config file is missing"

    with camera_config_path.open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def _as_rgb_uint8(frame):
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected RGB frame with 3 dims, got shape={frame.shape}")
    if frame.shape[2] > 3:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _pad_width(frame, width):
    if frame.shape[1] == width:
        return frame
    if frame.shape[1] > width:
        return frame[:, :width, :]
    pad_w = width - frame.shape[1]
    return np.pad(frame, ((0, 0), (0, pad_w), (0, 0)), mode="constant")


def _make_eval_video_frame(env):
    obs = env.now_obs["observation"]
    head_rgb = _as_rgb_uint8(obs["head_camera"]["rgb"])

    if "left_camera" not in obs or "right_camera" not in obs:
        return head_rgb

    left_rgb = _as_rgb_uint8(obs["left_camera"]["rgb"])
    right_rgb = _as_rgb_uint8(obs["right_camera"]["rgb"])
    bottom_rgb = np.concatenate([left_rgb, right_rgb], axis=1)
    target_w = max(head_rgb.shape[1], bottom_rgb.shape[1])
    return np.concatenate(
        [_pad_width(head_rgb, target_w), _pad_width(bottom_rgb, target_w)],
        axis=0,
    )


class _EvalVideoStdinProxy:
    def __init__(self, ffmpeg, env):
        self._ffmpeg = ffmpeg
        self._env = env

    def write(self, _data):
        frame = _make_eval_video_frame(self._env)
        return self._ffmpeg.stdin.write(frame.tobytes())

    def close(self):
        return self._ffmpeg.stdin.close()


class _EvalVideoFFmpegProxy:
    def __init__(self, ffmpeg, env):
        self._ffmpeg = ffmpeg
        self.stdin = _EvalVideoStdinProxy(ffmpeg, env)

    def wait(self):
        return self._ffmpeg.wait()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


def _release_episode_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    chunk_idx = int(usr_args.get("chunk_idx", 0))
    chunk_size = int(usr_args.get("chunk_size", eval_num_episodes))
    chunk_seed_start = usr_args.get("chunk_seed_start", None)
    chunk_seed_candidate_count = int(usr_args.get("chunk_seed_candidate_count", max(chunk_size * 10, chunk_size)))
    global_episode_start = int(usr_args.get("global_episode_start", chunk_idx * chunk_size))
    total_task_episodes = int(usr_args.get("total_task_episodes", eval_num_episodes))

    if chunk_size <= 0:
        raise ValueError(f"`chunk_size` must be > 0, got: {chunk_size}")
    if chunk_seed_candidate_count < chunk_size:
        raise ValueError(
            f"`chunk_seed_candidate_count` must be >= chunk_size, "
            f"got candidate_count={chunk_seed_candidate_count}, chunk_size={chunk_size}"
        )
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    default_st_seed = 100000 * (1 + seed)
    st_seed = int(chunk_seed_start) if chunk_seed_start is not None else default_st_seed
    suc_nums = []
    test_num = chunk_size
    topk = 1

    model = get_model(usr_args)
    try:
        st_seed, chunk_stats = eval_policy(task_name,
                                           TASK_ENV,
                                           args,
                                           model,
                                           st_seed,
                                           test_num=test_num,
                                           video_size=video_size,
                                           instruction_type=instruction_type,
                                           skip_get_obs_within_replan=skip_get_obs_within_replan,
                                           chunk_idx=chunk_idx,
                                           global_episode_start=global_episode_start,
                                           chunk_seed_start=st_seed,
                                           chunk_seed_candidate_count=chunk_seed_candidate_count)
    finally:
        print("[ROBOTWIN_CHUNK_CLEANUP] start deep cleanup after chunk", flush=True)
        _cleanup_chunk_runtime(TASK_ENV, model)
        print("[ROBOTWIN_CHUNK_CLEANUP] finish deep cleanup after chunk", flush=True)

    suc_num = int(chunk_stats["success_count"])
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    result_suffix = _result_suffix_from_task_config(task_config)
    success_rate = float(suc_num / test_num) if test_num > 0 else 0.0

    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Timestamp: {eval_ts}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Task Name: {task_name}\n")
        file.write(f"Task Config: {task_config}\n")
        file.write(f"Chunk Index: {chunk_idx}\n")
        file.write(f"Chunk Size: {chunk_size}\n")
        file.write(f"Global Episode Start: {global_episode_start}\n")
        file.write(f"Total Task Episodes: {total_task_episodes}\n")
        file.write(f"Candidate Seed Start: {chunk_stats['candidate_seed_start']}\n")
        file.write(f"Candidate Seed Count: {chunk_stats['candidate_seed_count']}\n")
        file.write(f"Candidate Seed End: {chunk_stats['candidate_seed_end']}\n")
        file.write(f"Exceeded Candidate Range: {chunk_stats['exceeded_candidate_range']}\n")
        file.write(f"Episode Count: {test_num}\n")
        file.write(f"Success Count: {suc_num}\n")
        file.write(f"Success Rate: {success_rate}\n\n")
        file.write(str(success_rate))

    chunk_json_path = Path(save_dir) / "chunk_result.json"
    chunk_payload = {
        "timestamp": eval_ts,
        "task_name": task_name,
        "task_config": task_config,
        "phase": result_suffix,
        "instruction_type": instruction_type,
        "chunk_idx": chunk_idx,
        "chunk_size": chunk_size,
        "global_episode_start": global_episode_start,
        "total_task_episodes": total_task_episodes,
        "episode_count": test_num,
        "success_count": suc_num,
        "success_rate": success_rate,
        **chunk_stats,
    }
    chunk_json_path.write_text(
        json.dumps(chunk_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Data has been saved to {file_path}")
    print(f"Chunk json has been saved to {chunk_json_path}")


def _release_cuda_cache() -> None:
    gc.collect()

    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    with contextlib.suppress(Exception):
        from sapien.render import clear_cache as sapien_clear_cache
        sapien_clear_cache()

    gc.collect()


def _cleanup_chunk_runtime(env=None, model=None) -> None:
    if env is not None:
        with contextlib.suppress(Exception):
            if hasattr(env, "_del_eval_video_ffmpeg"):
                env._del_eval_video_ffmpeg()

        with contextlib.suppress(Exception):
            env.close_env(clear_cache=True)

        with contextlib.suppress(Exception):
            _clear_poisoned_runtime_state(env)

        with contextlib.suppress(Exception):
            if hasattr(env, "viewer") and env.viewer is not None:
                env.viewer.close()

        for attr_name in (
            "scene",
            "renderer",
            "viewer",
            "cameras",
            "robot",
            "laptop",
            "pot",
            "table",
            "wall",
            "now_obs",
        ):
            with contextlib.suppress(Exception):
                if hasattr(env, attr_name):
                    delattr(env, attr_name)

    if model is not None:
        with contextlib.suppress(Exception):
            if hasattr(model, "reset"):
                model.reset()
        with contextlib.suppress(Exception):
            if hasattr(model, "close"):
                model.close()
        with contextlib.suppress(Exception):
            if hasattr(model, "client"):
                delattr(model, "client")

    _release_cuda_cache()



def _cleanup_chunk_runtime(env=None, model=None) -> None:
    if env is not None:
        with contextlib.suppress(Exception):
            if hasattr(env, "_del_eval_video_ffmpeg"):
                env._del_eval_video_ffmpeg()

        with contextlib.suppress(Exception):
            env.close_env(clear_cache=True)

        with contextlib.suppress(Exception):
            _clear_poisoned_runtime_state(env)

        with contextlib.suppress(Exception):
            if hasattr(env, "viewer") and env.viewer is not None:
                env.viewer.close()

        for attr_name in (
            "scene",
            "renderer",
            "viewer",
            "cameras",
            "robot",
            "laptop",
            "pot",
            "table",
            "wall",
            "now_obs",
        ):
            with contextlib.suppress(Exception):
                if hasattr(env, attr_name):
                    delattr(env, attr_name)

    if model is not None:
        with contextlib.suppress(Exception):
            if hasattr(model, "reset"):
                model.reset()
        with contextlib.suppress(Exception):
            if hasattr(model, "close"):
                model.close()
        with contextlib.suppress(Exception):
            if hasattr(model, "client"):
                delattr(model, "client")

    _release_cuda_cache()


def _clear_poisoned_runtime_state(env, exc: Exception | None = None) -> None:
    """
    setup_demo can fail after partially creating env.robot / planner / ffmpeg.
    Since this script reuses the same TASK_ENV, clear runtime objects before
    trying the next seed.
    """
    if hasattr(env, "robot"):
        try:
            robot = getattr(env, "robot")

            for conn_name in ("left_conn", "right_conn"):
                conn = getattr(robot, conn_name, None)
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

            for proc_name in ("left_proc", "right_proc"):
                proc = getattr(robot, proc_name, None)
                if proc is not None:
                    try:
                        if proc.is_alive():
                            proc.terminate()
                        proc.join(timeout=1)
                    except Exception:
                        pass

            delattr(env, "robot")
        except Exception:
            pass

    if hasattr(env, "ffmpeg"):
        try:
            env._del_eval_video_ffmpeg()
        except Exception:
            pass



def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False,
                chunk_idx=0,
                global_episode_start=0,
                chunk_seed_start=None,
                chunk_seed_candidate_count=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    candidate_seed_start = int(chunk_seed_start if chunk_seed_start is not None else st_seed)
    candidate_seed_count = int(chunk_seed_candidate_count if chunk_seed_candidate_count is not None else max(test_num * 10, test_num))
    candidate_seed_end = candidate_seed_start + candidate_seed_count - 1
    exceeded_candidate_range = False
    overflow_seed_count = 0
    episode_records = []

    args["eval_mode"] = True
    valid = False

    while succ_seed < test_num:
        valid = False
        if now_seed > candidate_seed_end:
            if not exceeded_candidate_range:
                print(
                    f"[ROBOTWIN_CHUNK_WARNING] candidate seed range exhausted: "
                    f"task={task_name} config={args['task_config']} chunk={chunk_idx} "
                    f"start={candidate_seed_start} end={candidate_seed_end} now_seed={now_seed}. "
                    f"Continue with overflow seeds, may overlap with later chunks.",
                    flush=True,
                )
            exceeded_candidate_range = True
            overflow_seed_count += 1

        render_freq = args["render_freq"]
        args["render_freq"] = 0
       
        if expert_check:
            try:
                global_episode_idx = global_episode_start + now_id
                TASK_ENV.setup_demo(now_ep_num=global_episode_idx, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                valid = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
                TASK_ENV.close_env()
            except UnStableError as e:
                _clear_poisoned_runtime_state(TASK_ENV, e)
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                _clear_poisoned_runtime_state(TASK_ENV, e)
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(" -------------")
                print("Error: ", e)
                print("Stack Trace: ", traceback.format_exc())
                print(" -------------")
                _clear_poisoned_runtime_state(TASK_ENV, e)
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                _clear_poisoned_runtime_state(TASK_ENV, e)
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue
        
        if (not expert_check) or valid:
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq
        t0=time.perf_counter()
        try:
            global_episode_idx = global_episode_start + now_id
            TASK_ENV.setup_demo(now_ep_num=global_episode_idx, seed=now_seed, is_test=True, **args)
        except UnStableError as e:
            # This seed passed expert_check but failed during rollout env init.
            # Roll back the accepted-seed counter and skip to next seed.
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            _clear_poisoned_runtime_state(TASK_ENV, e)
            try:
                TASK_ENV.close_env()
            except Exception:
                pass
            _clear_poisoned_runtime_state(TASK_ENV, e)
            now_seed += 1
            continue
        except Exception as e:
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            print(" -------------")
            print("Error: ", e)
            print("Stack Trace: ", traceback.format_exc())
            print(" -------------")
            _clear_poisoned_runtime_state(TASK_ENV, e)
            try:
                TASK_ENV.close_env()
            except Exception:
                pass
            _clear_poisoned_runtime_state(TASK_ENV, e)
            now_seed += 1
            print("error occurs !")
            continue
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        current_video_path = None
        phase_name = "random" if "randomized" in str(args["task_config"]).lower() else "clean"
        if TASK_ENV.eval_video_path is not None:
            episode_idx = global_episode_start + TASK_ENV.test_num
            current_video_path = Path(TASK_ENV.eval_video_path) / f"{phase_name}_episode{episode_idx}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(current_video_path),
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(_EvalVideoFFmpegProxy(ffmpeg, TASK_ENV))

        succ = False
        reset_func(model)
        t2=time.perf_counter()
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            need_obs = True
            if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                need_obs = bool(model.should_request_observation())

            observation = None
            if need_obs:
                observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
            if current_video_path is None or not current_video_path.exists():
                raise FileNotFoundError(f"Expected eval video file not found: {current_video_path}")
            is_randomized = "randomized" in str(args["task_config"]).lower()
            renamed_video_path = (
                Path(TASK_ENV.eval_video_path)
                / (
                    f"{phase_name}_episode{episode_idx}"
                    f"_randomized-{str(is_randomized).lower()}"
                    f"_success-{str(succ).lower()}.mp4"
                )
            )
            current_video_path.rename(renamed_video_path)

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        episode_records.append(
            {
                "local_episode_idx": int(TASK_ENV.test_num),
                "global_episode_idx": int(global_episode_start + TASK_ENV.test_num),
                "seed": int(now_seed),
                "success": bool(succ),
                "instruction": instruction,
            }
        )

        now_id += 1

        should_clear_cache = ((succ_seed + 1) % clear_cache_freq == 0)
        try:
            TASK_ENV.close_env(clear_cache=should_clear_cache)
        finally:
            _release_episode_memory()

        if should_clear_cache:
            _clear_poisoned_runtime_state(TASK_ENV)
            _release_episode_memory()

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        print(
            f"[ROBOTWIN_PROGRESS] done={TASK_ENV.test_num} total={test_num}",
            flush=True,
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    stats = {
        "candidate_seed_start": int(candidate_seed_start),
        "candidate_seed_count": int(candidate_seed_count),
        "candidate_seed_end": int(candidate_seed_end),
        "last_tried_seed": int(now_seed),
        "exceeded_candidate_range": bool(exceeded_candidate_range),
        "overflow_seed_count": int(overflow_seed_count),
        "used_seeds": [int(x) for x in suc_test_seed_list],
        "episode_records": episode_records,
        "success_count": int(TASK_ENV.suc),
        "episode_count": int(TASK_ENV.test_num),
    }
    return now_seed, stats


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
