from collections import Counter
from itertools import chain
import logging
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any

import hydra
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from termcolor import colored
import torch
import torch.distributed as dist
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parents[5]))
from iTRAP.models.Qwen3_VL.utils import setup_vlm_client, query_vlm, extract_gripper_points_and_actions, draw_trajectory_onto_image, save_trajectory_image
from flower.evaluation.multistep_sequences import get_sequences
from flower.evaluation.utils import get_env_state_for_initial_condition, join_vis_lang, LangEmbeddings
from flower.rollout.rollout_video import RolloutVideo

log_print = logging.getLogger(__name__)


def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    log_print.info(*args, **kwargs)


def divide_across_ranks(elements, world_size, rank):
    """
    Divide a number across subprocesses in multiprocessing.
    Example: distribute 4 elements in a world of size 3
    rank 0->2, rank 1->1, rank 2->1
    """
    assert rank < world_size
    rest = lambda n, w, i: 1 if n % w > i else 0
    return elements // world_size + rest(elements, world_size, rank)



def sequences_for_rank(num_sequences):
    """
    When using ddp, determine how many sequences every process should evaluate.
    """
    rank = dist.get_rank()
    ws = dist.get_world_size()
    num_seq_per_gpu = divide_across_ranks(num_sequences, ws, rank)
    num_workers = multiprocessing.cpu_count() // ws
    print(num_workers)
    print(num_seq_per_gpu)
    print(ws)
    print(rank)
    sequences = get_sequences(num_sequences, num_workers=num_workers)
    # print("Sequences:", sequences)
    
    print(f"Type of sequences: {type(sequences)}")
    print(f"Length of sequences: {len(sequences)}")
    print(f"First few elements of sequences: {sequences[:5]}")

    def manual_split(seq, n):
        avg = len(seq) // n
        remain = len(seq) % n
        last = 0
        results = []
        for _ in range(n):
            step = avg + (1 if remain > 0 else 0)
            results.append(seq[last:last+step])
            last += step
            remain -= 1
        return results

    try:
        sequences_np = np.array(sequences)
    except Exception as e:
        print(f"Exception when converting to numpy array: {e}")

    return manual_split(get_sequences(num_sequences, num_workers=num_workers), ws)[rank][:num_seq_per_gpu]


def gather_results(local_results):
    """
    Collect eval results from all processes.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_results
    results = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(results, local_results)
    return list(chain(*results))


class RolloutLongHorizon(Callback):
    """
    A class for performing rollouts during validation step.
    """

    def __init__(
        self,
        env_cfg,
        skip_epochs,
        rollout_freq,
        num_videos,
        num_sequences,
        replan_freq,
        ep_len,
        tasks,
        log_video_to_file,
        save_dir,
        vis_lang_folder,
        empty_cache,
        val_annotations,
        debug,
        traj_stretch_factor=1.0
    ):
        super().__init__()
        self.env = None  # type: Any
        self.env_cfg = env_cfg
        self.task_checker = hydra.utils.instantiate(tasks)
        self.skip_epochs = skip_epochs
        self.rollout_freq = rollout_freq
        self.num_videos = num_videos
        self.num_sequences = num_sequences
        self.replan_freq = replan_freq
        self.ep_len = ep_len
        self.log_video_to_file = log_video_to_file
        self.save_dir = save_dir
        self.rollout_video = None  # type: Any
        self.empty_cache = empty_cache
        self.device = None  # type: Any
        self.lang_embeddings = None
        self.vis_lang_folder = vis_lang_folder
        self.eval_sequences = None
        self.val_annotations = val_annotations
        self.debug = debug
        self.traj_stretch_factor = traj_stretch_factor

        complete_calvin_cfg = hydra.compose(config_name="config_calvin")
        val_transforms_cfg_static = complete_calvin_cfg.datamodule.transforms.val.rgb_static
        self.val_transforms_static = []
        for val_transform_cfg_static in val_transforms_cfg_static:
            self.val_transforms_static.append(hydra.utils.instantiate(val_transform_cfg_static))
        val_transforms_cfg_gripper = complete_calvin_cfg.datamodule.transforms.val.rgb_gripper
        self.val_transforms_gripper = []
        for val_transform_cfg_gripper in val_transforms_cfg_gripper:
            self.val_transforms_gripper.append(hydra.utils.instantiate(val_transform_cfg_gripper))

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""
        if self.env is None:
            self.device = pl_module.device

            try:
                # Get validation dataloader safely
                val_dataloaders = trainer.val_dataloaders
                
                # Handle the dict structure
                if isinstance(val_dataloaders, dict):
                    # Try to get 'vis_lang' dataloader first, fallback to first available dataloader
                    dataloader = val_dataloaders.get('vis_lang', next(iter(val_dataloaders.values())))
                elif isinstance(val_dataloaders, list):
                    dataloader = val_dataloaders[0]
                else:
                    dataloader = val_dataloaders

                # Get the dataset directly - it's an ExtendedDiskDataset
                dataset = dataloader.dataset
                
                # Initialize environment
                from flower.rollout.rollout import Rollout
                for callback in trainer.callbacks:
                    if isinstance(callback, Rollout) and callback.env is not None:
                        self.env = callback.env
                        break
                else:
                    self.env = hydra.utils.instantiate(self.env_cfg, dataset, pl_module.device)

                # Setup video logging if needed
                if self.num_videos > 0:
                    if dist.is_available() and dist.is_initialized():
                        self.num_videos = divide_across_ranks(
                            self.num_videos, 
                            dist.get_world_size(), 
                            dist.get_rank()
                        )
                    self.rollout_video = RolloutVideo(
                        logger=pl_module.logger,
                        empty_cache=self.empty_cache,
                        log_to_file=self.log_video_to_file,
                        save_dir=self.save_dir,
                    )

                # Initialize language embeddings with the dataset
                self.lang_embeddings = LangEmbeddings(
                    dataset.abs_datasets_dir, 
                    dataset.vis_lang_folder, 
                    device=pl_module.device
                )

                if dist.is_available() and dist.is_initialized():
                    self.eval_sequences = sequences_for_rank(self.num_sequences)
                else:
                    self.eval_sequences = get_sequences(self.num_sequences)

            except Exception as e:
                raise RuntimeError(f"Failed to initialize validation environment: {str(e)}")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation epoch ends."""
        # Skip evaluation in early epochs if configured
        if pl_module.current_epoch == 0 and self.skip_epochs > 0:
            self._log_zero_metrics(pl_module)
            return

        # Check if we should run evaluation this epoch
        should_evaluate = (
            pl_module.current_epoch == self.skip_epochs or 
            ((pl_module.current_epoch - self.skip_epochs) >= 0 and 
             (pl_module.current_epoch - self.skip_epochs) % self.rollout_freq == 0)
        )

        if should_evaluate:
            results = self.evaluate_policy(pl_module)
            results = gather_results(results)
            count = Counter(results)  # type: ignore
            print()
            for i in range(1, 6):
                n_success = sum(count[j] for j in reversed(range(i, 6)))
                sr = n_success / len(results)
                pl_module.log(f"eval_lh/sr_chain_{i}", torch.tensor(sr), on_step=False, sync_dist=True)
                log_rank_0(f"{i} / 5 subtasks: {n_success} / {len(results)} sequences, SR: {sr * 100:.1f}%")
            avg_seq_len = np.mean(results)
            pl_module.log("eval_lh/avg_seq_len", torch.tensor(avg_seq_len), on_epoch=True, sync_dist=True)
            log_rank_0(f"Average successful sequence length: {avg_seq_len:.1f}")
            print()

    def _log_zero_metrics(self, pl_module: LightningModule) -> None:
        """Log zero metrics for skipped evaluations."""
        for i in range(1, 6):
            pl_module.log(
                f"eval_lh/sr_chain_{i}", 
                torch.tensor(0.0), 
                on_step=False, 
                sync_dist=True
            )
        pl_module.log(
            "eval_lh/avg_seq_len", 
            torch.tensor(0.0), 
            on_step=False, 
            sync_dist=True
        )

    def evaluate_policy(self, model):
        # skip rollout as VLM setup not available
        self._log_zero_metrics(model)
        return [0]

        vlm_client = setup_vlm_client()

        results = []
        total_evaluations = len(self.eval_sequences)
        local_rank = int(dist.get_rank()) if (dist.is_available() and dist.is_initialized()) else 0
        for seq_nr, (initial_state, eval_sequence) in enumerate(tqdm(self.eval_sequences, desc=f"Evaluating Policy (rank={local_rank})",
                     total=total_evaluations, position=local_rank)):
            record = seq_nr < self.num_videos
            result = self.evaluate_sequence(vlm_client, model, initial_state, eval_sequence, seq_nr, record)
            results.append(result)
            if record:
                global_step = 504 * model.current_epoch # 504 steps per epoch with current setup
                self.rollout_video.log(global_step)
        return results

    def evaluate_sequence(self, vlm_client, model, initial_state, eval_sequence, seq_nr, record):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        
        if record:
            caption = " | ".join(eval_sequence)
            self.rollout_video.new_video(tag=self.get_video_tag(seq_nr), caption=caption)
        
        if self.debug:
            print()
            print()
            print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
            print("Subtask: ", end="")
        
        success_counter = 0

        for subtask_nr, subtask in enumerate(eval_sequence):
            if record:
                self.rollout_video.new_subtask()
            
            success = self.rollout(vlm_client, model, subtask, seq_nr, subtask_nr, record)

            if record:
                self.rollout_video.draw_outcome(success)
            
            if success:
                success_counter += 1
            else:
                return success_counter

        return success_counter

    def rollout(self, vlm_client, model, subtask, seq_nr, subtask_nr, record):
        if self.debug:
            print(f"{subtask} ", end="")
        obs = self.env.get_obs()    

        # Get lang goal embedding & annotation text for subtask
        goal = self.lang_embeddings.get_lang_goal(subtask)
        goal["lang_text"] = self.val_annotations[subtask][0]

        # get trajectory points & actions from initial state of scene & robot (static camera image untransformed as render() used instead of get_obs())
        untransformed_static_img = self.env.cameras[0].render()[0].squeeze()
        untransformed_gripper_img = self.env.cameras[1].render()[0].squeeze()
        vlm_response = query_vlm(untransformed_static_img, untransformed_gripper_img, vlm_client, subtask)
        traj_gripper_points, traj_gripper_actions, _ = extract_gripper_points_and_actions(vlm_response, untransformed_static_img.shape[0],
                                                                                       untransformed_static_img.shape[1], logger=log_print,
                                                                                       stretch_factor=self.traj_stretch_factor)

        model.reset()
        start_info = self.env.get_info()

        if record:
            # update video with initial state
            static_img = self.env.cameras[0].render()[0].squeeze()
            static_traj_img = draw_trajectory_onto_image(static_img, traj_gripper_points, traj_gripper_actions)
            normalized_static_traj_img = static_traj_img / 127.5 - 1 # normalize to [-1, 1]
            self.rollout_video.update(torch.tensor(normalized_static_traj_img).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to(self.device))
        
        local_rank = int(dist.get_rank()) if (dist.is_available() and dist.is_initialized()) else 0

        success = False
        for step in tqdm(range(self.ep_len), total=self.ep_len, desc=f"Rolling out policy for {subtask} (rank={local_rank})", leave=False):
            if step == self.ep_len / 2:
                # query_vlm again to help robot out of possibly wrong state
                untransformed_static_img = self.env.cameras[0].render()[0].squeeze()
                untransformed_gripper_img = self.env.cameras[1].render()[0].squeeze()
                vlm_response = query_vlm(untransformed_static_img, untransformed_gripper_img, vlm_client, subtask)
                traj_gripper_points, traj_gripper_actions, _ = extract_gripper_points_and_actions(vlm_response, untransformed_static_img.shape[0],
                                                                                               untransformed_static_img.shape[1], logger=log_print,
                                                                                               stretch_factor=self.traj_stretch_factor)
            
            if step % model.multistep == 0:
                # model predicts multistep actions per step => only draw trajectory once per multistep
                untransformed_static_img = self.env.cameras[0].render()[0].squeeze()
                untransformed_static_traj_img = draw_trajectory_onto_image(untransformed_static_img, traj_gripper_points, traj_gripper_actions)
                #save_trajectory_image(untransformed_static_traj_img, subtask, local_rank, seq_nr, subtask_nr, step)
                untransformed_gripper_img = self.env.cameras[1].render()[0].squeeze()
                untransformed_gripper_traj_img = untransformed_gripper_img.copy() # TODO: implement transform from static to gripper cam
                #save_trajectory_image(untransformed_gripper_traj_img, subtask, local_rank, seq_nr, subtask_nr, step)

                # apply transforms to trajectory images
                transformed_static_traj_img = torch.tensor(untransformed_static_traj_img).permute(2, 0, 1).unsqueeze(0)
                for val_transform_static in self.val_transforms_static:
                    transformed_static_traj_img = val_transform_static(transformed_static_traj_img)
                obs["vis_image_static"] = transformed_static_traj_img.unsqueeze(0).to(self.device)
                transformed_gripper_traj_img = torch.tensor(untransformed_gripper_traj_img).permute(2, 0, 1).unsqueeze(0)
                for val_transform_gripper in self.val_transforms_gripper:
                    transformed_gripper_traj_img = val_transform_gripper(transformed_gripper_traj_img)
                obs["vis_image_gripper"] = transformed_gripper_traj_img.unsqueeze(0).to(self.device)
            
            action = model.step(obs, goal)
            # print(action.shape)
            obs, _, _, current_info = self.env.step(action)
            if self.debug and os.environ.get("DISPLAY") is not None:
                img = self.env.render(mode="rgb_array")
                join_vis_lang(img, goal["lang_text"])
            if record:
                # update video
                static_img = self.env.cameras[0].render()[0].squeeze()
                static_traj_img = draw_trajectory_onto_image(static_img, traj_gripper_points, traj_gripper_actions)
                normalized_static_traj_img = static_traj_img / 127.5 - 1 # normalize to [-1, 1]
                self.rollout_video.update(torch.tensor(normalized_static_traj_img).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to(self.device))
            # check if current step solves a task
            current_task_info = self.task_checker.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                success = True
                break
        if self.debug:
            if success:
                print(colored("success", "green"), end=" ")
            else:
                print(colored("fail", "red"), end=" ")
        if record:
            self.rollout_video.add_language_instruction(goal["lang_text"])
        return success
