from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist

from flower.evaluation.multistep_sequences import get_sequences
from flower.evaluation.utils import get_default_mode_and_env, get_env_state_for_initial_condition, join_vis_lang, gen_heatmaps
from flower.rollout.rollout_video import RolloutVideo

logger = logging.getLogger(__name__)

ROOT_OUTPUT_PATH = Path(__file__).parents[2] / "outputs"


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


def get_log_dir(log_dir):
    if log_dir is None:
        log_dir = Path(__file__).parents[2] / "outputs"

    output_dirs = [os.path.join(log_dir, day, time) for day in os.listdir(log_dir) for time in os.listdir(Path(log_dir) / day)]
    latest_output_dir = max(output_dirs)

    print(f"logging to {log_dir}")

    return Path(latest_output_dir)


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(cfg, total_results, plan_dicts, attns_sequences=None, log_dir=None):
    if attns_sequences is not None:
        attns_sequences = attns_sequences[Path(cfg.checkpoint)]

    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)

    sequences = get_sequences(cfg.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        epoch = checkpoint.stem.split("=")[1] if "=" in checkpoint.stem else "best"
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
        wandb.log({"avrg_performance/avg_seq_len": avg_seq_len, "avrg_performance/chain_sr": chain_sr, "detailed_metrics/task_info": task_info})
        current_data[epoch] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file, indent=2)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")

    if cfg.visualize_attention and attns_sequences is not None:
        print()

        output_dirs = [os.path.join(ROOT_OUTPUT_PATH, day, time) for day in os.listdir(ROOT_OUTPUT_PATH) for time in os.listdir(Path(ROOT_OUTPUT_PATH) / day)]
        latest_output_dir = max(output_dirs)
        attvis_output_dir = f"{latest_output_dir}/attvis"

        heatmaps = gen_heatmaps(attns_sequences, output_dir=attvis_output_dir, merge_attn_heads=cfg.merge_attn_heads, num_heatmaps=cfg.num_attn_heatmaps)
        for sequence_number, heatmaps_sequence in tqdm(enumerate(heatmaps), total=len(heatmaps), desc="Uploading heatmaps to wandb"):
            i = 0
            num_zeros_subtask = len(str(len(heatmaps_sequence)))
            for subtask, heatmaps_subtask in heatmaps_sequence.items():
                num_zeros_step = len(str(max(heatmaps_subtask.keys())))
                for step_number, heatmaps_step in tqdm(heatmaps_subtask.items(), leave=False):
                    wandb.log({f"attention_heatmaps/sequence_{sequence_number}/{i:0{num_zeros_subtask}}_{subtask}/step_{step_number:0{num_zeros_step}}": heatmaps_step}) # keys ordered in input order as of python 3.7
                i += 1


def evaluate_policy(model, env, lang_embeddings, cfg, num_videos=0, save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations

    # video stuff
    if num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=save_dir,
            resolution_scale=1,
        )
    else:
        rollout_video = None

    eval_sequences = get_sequences(cfg.num_sequences)

    results = []
    plans = defaultdict(list)
    attns_sequences = []

    if not cfg.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        record = i < num_videos
        result, attns_sequence = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i
        )
        results.append(result)
        if cfg.visualize_attention:
            attns_sequences.append(attns_sequence)
        
        if record:
            rollout_video.write_to_tmp()
        if not cfg.debug:
            success_rates = count_success(results)
            average_rate = sum(success_rates) / len(success_rates) * 5
            description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)

    if num_videos > 0:
        # log rollout videos
        rollout_video._log_videos_to_file(0, save_as_video=False)
    return results, plans, attns_sequences


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    
    attns_sequence = []

    for subtask in eval_sequence:
        if record:
            rollout_video.new_subtask()
        
        success, attns_task = rollout(env, model, task_checker, cfg, subtask, lang_embeddings, val_annotations, record, rollout_video)
        
        if cfg.visualize_attention:
            attns_sequence.append({"subtask": subtask, "attns": attns_task})
        
        if record:
            rollout_video.draw_outcome(success)
        
        if success:
            success_counter += 1
        else:
            return success_counter, attns_sequence
    return success_counter, attns_sequence


def rollout(env, model, task_oracle, cfg, subtask, lang_embeddings, val_annotations, record=False, rollout_video=None):
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    # goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal = {}
    goal['lang_text'] = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    attns_task = []

    for step in range(cfg.ep_len):
        action, attns_step = model.step(obs, goal)
        if cfg.visualize_attention:
            attns_task.append(attns_step)
        
        obs, _, _, current_info = env.step(action)
        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True, attns_task
    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record:
        rollout_video.add_language_instruction(lang_annotation)
    return False, attns_task


@hydra.main(config_path="../../conf", config_name="eval_calvin")
def main(cfg):
    log_wandb = cfg.log_wandb
    # torch.cuda.set_device(cfg.device)
    seed_everything(0, workers=True) 
    lang_embeddings = None
    env = None
    results = {}
    plans = {}
    attns_sequences = {}

    print(cfg.device)
    model, env, _, lang_embeddings = get_default_mode_and_env(
        cfg.train_folder,
        cfg.dataset_path,
        cfg.checkpoint,
        env=env,
        lang_embeddings=lang_embeddings,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=cfg.device,
    )

    model = model.to(cfg.device)

    if cfg.num_sampling_steps is not None:
        model.num_sampling_steps = cfg.num_sampling_steps
    if cfg.multistep is not None:
        model.multistep = cfg.multistep
    print(model.num_sampling_steps, model.multistep)

    model.eval()

    log_dir = get_log_dir(cfg.log_dir)
    if log_wandb:
        os.makedirs(log_dir / "wandb", exist_ok=False)
        run = wandb.init(
            project='attvis_flower_calvin_eval',
            entity=cfg.wandb_entity,
            # group=cfg.model_name + cfg.sampler_type + '_' + str(cfg.num_sampling_steps) + '_steps_' + str(cfg.num_sequences) + '_rollouts_',
            config=OmegaConf.to_object(cfg),
            # dir=log_dir / "wandb",
        )

    results[Path(cfg.checkpoint)], plans[Path(cfg.checkpoint)], attns_sequences[Path(cfg.checkpoint)] = evaluate_policy(model, env, lang_embeddings, cfg, num_videos=cfg.num_videos, save_dir=Path(log_dir))
    
    if cfg.visualize_attention:
        print_and_save(cfg, results, plans, attns_sequences, log_dir=log_dir)
    else:
        print_and_save(cfg, results, plans, log_dir=log_dir)
    
    if log_wandb:
        run.finish()


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Add calvin env to path
    sys.path.append(str(Path(__file__).absolute().parents[2] / "calvin_env"))

    main()