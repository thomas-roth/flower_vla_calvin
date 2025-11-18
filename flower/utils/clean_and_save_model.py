import logging
import os
from pathlib import Path
import torch
from safetensors.torch import save_file
import gc



def _get_latest_run_path(logs_path: Path) -> str:
    # Get saved models directory of latest run
    all_days_path = [dir for dir in logs_path.iterdir() if dir.is_dir()]
    all_days_path.sort(key=lambda dir: dir.stat().st_ctime, reverse=True)
    if len(all_days_path) == 0:
        return None
    last_day_path = all_days_path[0]

    all_runs_last_day_path = [dir for dir in last_day_path.iterdir() if dir.is_dir()]
    all_runs_last_day_path.sort()
    if len(all_runs_last_day_path) == 0:
        return None
    last_run_last_day_path = all_runs_last_day_path[-1]
    
    seed = last_run_last_day_path.name.split("seed")[-1]
    saved_models_last_run_last_day_path = Path(last_run_last_day_path / f"seed_{seed}" / "saved_models")
    if not saved_models_last_run_last_day_path.exists():
        return None

    return saved_models_last_run_last_day_path


def get_model_checkpoint_paths(logger, path_to_ckpt_file_or_dir=None):
    if path_to_ckpt_file_or_dir is None:
        logger.info("No path to model checkpoint file or directory given. Cleaning & saving all checkpoints of latest run.")

        runs_path = Path(__file__).absolute().parents[5] / "logs" / "runs"
        model_checkpoints_dir_path = _get_latest_run_path(runs_path)
        if model_checkpoints_dir_path is None:
            logger.warning("Aborting: no latest run found")
            return None
        logger.info(f"Found latest run: {model_checkpoints_dir_path}")

        model_checkpoint_paths = list(model_checkpoints_dir_path.rglob("*.ckpt"))
        if not model_checkpoint_paths:
            logger.warning("Aborting: no model checkpoint found in latest run")
            return None
        model_checkpoint_paths.sort()
    elif Path(path_to_ckpt_file_or_dir).is_dir():
        # is_dir() also checks if path exists
        logger.info(f"Path to model checkpoint directory given. Cleaning & saving checkpoints in: {path_to_ckpt_file_or_dir}")
        model_checkpoint_paths = list(Path(path_to_ckpt_file_or_dir).rglob("*.ckpt"))
        if not model_checkpoint_paths:
            logger.warning(f"Aborting: no model checkpoints found in directory: {path_to_ckpt_file_or_dir}")
            return None
        model_checkpoint_paths.sort()
    elif Path(path_to_ckpt_file_or_dir).is_file() and path_to_ckpt_file_or_dir.endswith(".ckpt"):
        # is_file() also checks if path exists
        logger.info(f"Path to model checkpoint file given. Cleaning & saving checkpoint at: {path_to_ckpt_file_or_dir}")
        model_checkpoint_paths = [path_to_ckpt_file_or_dir]
    else:
        logger.error(f"Aborting: given path to model checkpoint file or directory invalid: {path_to_ckpt_file_or_dir}")
        return None

    return model_checkpoint_paths


def clean_and_save_model(logger, path_to_ckpt_file_or_dir=None, delete_ckpt=True):
    model_checkpoint_paths = get_model_checkpoint_paths(logger, path_to_ckpt_file_or_dir)
    if model_checkpoint_paths is None:
        return
    
    for i, model_checkpoint_path in enumerate(model_checkpoint_paths):
        logger.info(f"Processing model checkpoint {i+1} of {len(model_checkpoint_paths)}: {str(model_checkpoint_path).split('runs/')[-1]}")

        logger.info("Loading model checkpoint")
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
        
        logger.info("Cleaning model checkpoint")
        state_dict = checkpoint["state_dict"]
        cleaned_state_dict = {} #k.replace('model.', ''): v for k, v in state_dict.items()
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")
            cleaned_state_dict[new_key] = value.clone() if torch.is_tensor(value) else value

        logger.info("Saving model checkpoint in safetensors format")
        save_file(cleaned_state_dict, os.path.join(Path(model_checkpoint_path).parent, "model_cleaned.safetensors"))

        if delete_ckpt:
            logger.info("Removing original checkpoint file")
            os.remove(model_checkpoint_path)

        # Clear memory
        del checkpoint
        del state_dict
        del cleaned_state_dict
        gc.collect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='\033[34m[%(asctime)s][%(name)s][%(levelname)s]\033[0m - %(message)s',
                        handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    clean_and_save_model(logger, path_to_ckpt_file_or_dir="/home/troth/code/hiwi/iTRAP/iTRAP/models/flower_vla_calvin/pretrained/finetuned_calvin_abc_both_cams/16-33-50_seed42/seed_42/saved_models",
                         delete_ckpt=True)
