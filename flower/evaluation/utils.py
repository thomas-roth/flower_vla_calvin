from collections import defaultdict
import contextlib
import logging
import os
from pathlib import Path
from typing import Union
import importlib

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
import pyhash
import torch
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
import wandb

from flower.utils.utils import add_text, format_sftp_path

ENC_IMAGE_RESIZE_SHAPE = (224, 224)
enc_resize_shape = lambda num_context_tokens: (25 * num_context_tokens, 25 * num_context_tokens)
DEC_SELF_RESIZE_SHAPE = (250, 250)
dec_cross_resize_shape = lambda num_context_tokens : (25 * num_context_tokens, 250) # number of context tokens varies bc of diff lengths of prompt, tuple flipped bc of cv2


hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def load_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_evaluation_checkpoint(cfg):
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(cfg.module_path).expanduser())
    pl_module = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    ).cuda()
    return pl_module


def get_checkpoint_i_from_dir(dir, i: int = -1):
    ckpt_paths = list(dir.rglob("*.ckpt"))
    if i == -1:
        for ckpt_path in ckpt_paths:
            if ckpt_path.stem == "last":
                return ckpt_path

    # Search for ckpt of epoch i
    for ckpt_path in ckpt_paths:
        split_path = str(ckpt_path).split("_")
        for k, word in enumerate(split_path):
            if word == "epoch":
                if int(split_path[k + 1]) == i:
                    return ckpt_path

    sorted(ckpt_paths, key=lambda f: f.stat().st_mtime)
    return ckpt_paths[i]


def get_config_from_dir(dir):
    dir = Path(dir)
    config_yaml = list(dir.rglob("*hydra/config.yaml"))[0]
    return OmegaConf.load(config_yaml)


def load_pl_module_from_checkpoint(
    filepath: Union[Path, str],
    epoch: int = 1,
    overwrite_cfg: dict = {},
    use_ema_weights: bool = False
):
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.is_dir():
        filedir = filepath
        ckpt_path = get_checkpoint_i_from_dir(dir=filedir, i=epoch)
    elif filepath.is_file():
        assert filepath.suffix == ".ckpt", "File must have .ckpt extension"
        ckpt_path = filepath
        filedir = filepath.parents[0]
    else:
        raise ValueError(f"not valid file path: {str(filepath)}")
    config = get_config_from_dir(filedir)
    class_name = config.model.pop("_target_")
    if "_recursive_" in config.model:
        del config.model["_recursive_"]
    print(f"class_name {class_name}")
    module_class = load_class(class_name)
    print(f"Loading model from {ckpt_path}")
    load_cfg = {**config.model, **overwrite_cfg}
    model = module_class.load_from_checkpoint(ckpt_path, **load_cfg)
     # Load EMA weights if they exist and the flag is set
    if use_ema_weights:
        checkpoint_data = torch.load(ckpt_path)
        if "ema_weights" in checkpoint_data['callbacks']['EMA']:
            ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']

            # Convert list of tensors to a state_dict format
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(model.named_parameters())}

            model.load_state_dict(ema_weights_dict)
            print("Successfully loaded EMA weights from checkpoint!")
        else:
            print("Warning: No EMA weights found in checkpoint!")

    print(f"Finished loading model {ckpt_path}")
    return model



def get_default_model_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, device_id=0):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    cfg = OmegaConf.load(train_cfg_path)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    datasets_cfg = hydra.compose("vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder])
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device(f"cuda:{device_id}")

    if lang_embeddings is None:
        lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

    if env is None:
        rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")

    # new stuff
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(train_folder).expanduser())
    model = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    )
    # model = Hulc.load_from_checkpoint(checkpoint)
    model.freeze()
    if cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings


def get_default_mode_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, prep_dm_and_deps=True, device_id=0, eval_cfg_overwrite={}):
    # Fix for the path issue - ensure we're working with the directory containing the config
    train_folder_path = Path(train_folder)
    
    # If the train_folder is already pointing to a .yaml file, use its parent directory
    if train_folder_path.suffix == '.yaml':
        train_folder_path = train_folder_path.parent.parent  # Go up two levels from config.yaml
    
    # Now construct the correct path to the config.yaml file
    train_cfg_path = train_folder_path / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    
    print(f"Loading config from: {train_cfg_path}")
    
    def_cfg = OmegaConf.load(train_cfg_path)
    eval_override_cfg = OmegaConf.create(eval_cfg_overwrite)
    cfg = OmegaConf.merge(def_cfg, eval_override_cfg)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    
    if device_id != 'cpu':
        device = torch.device(f"cuda:{device_id}")
    else:
        device = 'cpu'
    
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    
    if prep_dm_and_deps:
        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.val_dataloader()
        dataset = dataloader["lang"].dataset

        if lang_embeddings is None:
            lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

        if env is None:
            rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout_lh/calvin.yaml")
            env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    # Fix for checkpoint path handling
    checkpoint_path = Path(checkpoint).expanduser()
    print(f"Loading model from {checkpoint_path}")
    
    model = load_mode_from_safetensor(
        checkpoint_path,
        overwrite_cfg=eval_cfg_overwrite.get("model", {}),
    )
    
    model.freeze()
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings

def load_mode_from_safetensor(
    filepath: Path,
    overwrite_cfg: dict = {},
):
    """Load model from a checkpoint file or directory.
    
    Args:
        filepath: Path to the checkpoint file or directory
        overwrite_cfg: Dict with configuration overrides
        
    Returns:
        Instantiated model
    """
    filepath = Path(filepath)
    
    # Determine if we're dealing with a file or directory
    if filepath.is_file():
        # If it's a checkpoint file, use its parent directory to find the config
        ckpt_path = filepath
        config_dir = filepath.parent
        
        # Try to find config in parent directories
        hydra_dir = None
        current_dir = config_dir
        for _ in range(4):  # Look up to 4 levels up
            if (current_dir / ".hydra").exists():
                hydra_dir = current_dir / ".hydra"
                break
            current_dir = current_dir.parent
        
        if hydra_dir is None:
            # If we can't find .hydra directory, try looking at a common pattern
            # From the filepath, try to find base directory (e.g., calvin_abcd)
            parts = filepath.parts
            try:
                # Look for "best_checkpoints" in the path
                idx = parts.index("best_checkpoints")
                if idx + 1 < len(parts):
                    base_dir = Path(*parts[:idx+2])
                    if (base_dir / ".hydra").exists():
                        hydra_dir = base_dir / ".hydra"
            except ValueError:
                pass
                
        if hydra_dir is None:
            raise ValueError(f"Could not find .hydra directory for checkpoint: {str(filepath)}")
            
        config_path = hydra_dir / "config.yaml"
    elif filepath.is_dir():
        # If it's a directory, look for .hydra/config.yaml
        ckpt_path = filepath
        if (filepath / ".hydra").exists():
            config_path = filepath / ".hydra/config.yaml"
        else:
            raise ValueError(f"Directory does not contain .hydra/config.yaml: {str(filepath)}")
    else:
        raise ValueError(f"Path does not exist: {str(filepath)}")
    
    print(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)
    
    print(f"Loading model from {ckpt_path}")
    load_cfg = OmegaConf.create({**OmegaConf.to_object(config.model), **{"optimizer": None}, **overwrite_cfg})
    
    # Remove 'ckpt_path' if it exists in load_cfg to avoid the error
    if 'ckpt_path' in load_cfg:
        del load_cfg['ckpt_path']
    
    # Set the pretrained model path
    load_cfg["pretrained_model_path"] = str(ckpt_path)
    
    # Instantiate the model
    model = hydra.utils.instantiate(load_cfg)

    print(f"Finished loading model {ckpt_path}")
    return model


def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


class LangEmbeddings:
    def __init__(self, val_dataset_path, lang_folder, device=torch.device("cuda:0")):
        embeddings = np.load(Path(val_dataset_path) / lang_folder / "embeddings.npy", allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
        self.device = device

    def get_lang_goal(self, task):
        return {"lang": torch.from_numpy(self.lang_embeddings[task]).to(self.device).squeeze(0).float()}


def imshow_tensor(window, img_tensor, wait=0, resize=True, keypoints=None, text=None):
    img_tensor = img_tensor.squeeze()
    img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
    img = np.clip(((img / 2) + 0.5) * 255, 0, 255).astype(np.uint8)

    if keypoints is not None:
        key_coords = np.clip(keypoints * 200 + 100, 0, 200)
        key_coords = key_coords.reshape(-1, 2)
        cv_kp1 = [cv2.KeyPoint(x=pt[1], y=pt[0], _size=1) for pt in key_coords]
        img = cv2.drawKeypoints(img, cv_kp1, None, color=(255, 0, 0))

    if text is not None:
        add_text(img, text)

    if resize:
        cv2.imshow(window, cv2.resize(img[:, :, ::-1], (500, 500)))
    else:
        cv2.imshow(window, img[:, :, ::-1])
    cv2.waitKey(wait)


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    s = sum(demo_task_counter.values())
    success_rate = (sum(live_task_counter.values()) / s if s > 0 else 0) * 100
    logger.info(f"Average Success Rate {mod} = {success_rate:.0f}%")
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


def gen_heatmaps(attns_sequences, output_dir, merge_attn_heads=True, num_heatmaps=-1): # -1 means all heatmaps
    input_tokens_enc = lambda num_context_tokens: ["image1_spatial"] + [f"image1_temporal{i:02d}" for i in range(49)] + ["image2_spatial"] + [f"image2_temporal{i:02d}" for i in range(49)] \
        + ["task"] + [f"prompt{i:02d}" for i in range(num_context_tokens - 101)]
    input_tokens_dec_self = [f"action{i}" for i in range(10)]
    input_tokens_dec_cross = lambda num_context_tokens : [f"static-cam{i:02d}" for i in range(50)] + [f"gripper-cam{i:02d}" for i in range(50)] \
        + ["task"] + [f"prompt{i:02d}" for i in range(num_context_tokens - 101)]

    heatmaps = [defaultdict(lambda: defaultdict(list)) for _ in range(len(attns_sequences))]

    for sequence_number, attns_sequence in tqdm(enumerate(attns_sequences), total=len(attns_sequences), desc="Generating attn heatmaps for sequences"):
        for task_number, attns_task in tqdm(enumerate(attns_sequence), leave=False, total=len(attns_sequence), desc=f"Generating attn heatmaps for tasks in sequence {sequence_number}"):
            subtask = attns_task["subtask"].replace(" ", "_")

            for step_number, attns_step in tqdm(enumerate(attns_task["attns"]), leave=False, total=len(attns_task["attns"]), desc=f"Generating attn heatmaps for steps in task {task_number}"):
                if attns_step is None:
                    continue # skip if step action already predicted in previous step (multistep prediction)

                attns_step_enc_image = attns_step["attns_enc_image"] # [{"img": (B, C, H, W), "attn": (B, nh, Ta, Ta)}] = [{"img": (1, 3, 224, 224), "attn": (1, 8, 32, 32)}]
                attns_step_enc_image2 = attns_step["attns_enc_image2"] # None | [{"img": (B, C, H, W), "attn": (B, nh, Ta, Ta)}] = [{"img": (1, 3, 224, 224), "attn": (1, 8, 32, 32)}]
                attns_step_enc = attns_step["attns_enc"] # [(B, nh, Tb, Tb)] = [(1, 16, 13_, 13_)]
                attns_step_dec = attns_step["attns_dit_steps"] # [{"self": (B, nh, Tc, Tc), "cross": (B, nh, Tc, Tb)}] = [{"self": (1, 16, 10, 10), "cross": (1, 16, 10, 13_)}]

                # gen heatmaps for image encoder of VLM
                for flow_step_inv, attns_flow_step_enc_image in enumerate(attns_step_enc_image):
                    flow_step = len(attns_step_enc_image) - flow_step_inv - 1

                    heatmaps, num_heatmaps = _gen_heatmaps_for_layers(attns_flow_step_enc_image, merge_attn_heads, num_heatmaps, output_dir, sequence_number, task_number,
                                                                      subtask, step_number, heatmaps, attn_name="enc_image", flow_step=flow_step)
                
                if attns_step_enc_image2 is not None:
                    # gen heatmaps for image2 encoder of VLM
                    for flow_step_inv, attns_flow_step_enc_image2 in enumerate(attns_step_enc_image2):
                        flow_step = len(attns_step_enc_image2) - flow_step_inv - 1

                        heatmaps, num_heatmaps = _gen_heatmaps_for_layers(attns_flow_step_enc_image2, merge_attn_heads, num_heatmaps, output_dir, sequence_number, task_number,
                                                                          subtask, step_number, heatmaps, attn_name="enc_image2", flow_step=flow_step)

                # gen heatmaps for encoder of VLM
                heatmaps, num_heatmaps = _gen_heatmaps_for_layers(attns_step_enc, merge_attn_heads, num_heatmaps, output_dir, sequence_number, task_number,
                                                                  subtask, step_number, heatmaps, attn_name="enc", resize_shape=enc_resize_shape,
                                                                  x_labels=input_tokens_enc, y_labels=input_tokens_enc)
                
                # gen heatmaps for decoder of FLOWER
                for flow_step_inv, attns_time_dec in enumerate(attns_step_dec):
                    flow_step = len(attns_step_dec) - flow_step_inv - 1

                    heatmaps, num_heatmaps = _gen_heatmaps_for_layers(attns_time_dec, merge_attn_heads, num_heatmaps, output_dir, sequence_number, task_number,
                                                                      subtask, step_number, heatmaps, attn_name="dec_self", resize_shape=DEC_SELF_RESIZE_SHAPE,
                                                                      x_labels=input_tokens_dec_self, y_labels=input_tokens_dec_self, flow_step=flow_step)
                    heatmaps, num_heatmaps = _gen_heatmaps_for_layers(attns_time_dec, merge_attn_heads, num_heatmaps, output_dir, sequence_number, task_number,
                                                                      subtask, step_number, heatmaps, attn_name="dec_cross", resize_shape=dec_cross_resize_shape,
                                                                      x_labels=input_tokens_dec_cross, y_labels=input_tokens_dec_self, flow_step=flow_step)

    # list heatmap counts per sequence and subtask
    for sequence_number, heatmaps_sequence in enumerate(heatmaps):
        num_zeros_heatmaps = 0 # short variant like for num_zeros_seqs doesnt work if num_attvis_heatmaps != -1 bc some subtasks might not have any heatmaps
        for subtask in heatmaps_sequence.keys():
            if len(heatmaps_sequence[subtask]) > 0:
                num_zeros_heatmaps = max(num_zeros_heatmaps, len(str(len(heatmaps_sequence[subtask]))))
        num_zeros_seqs = max(len(str(sequence_number)) for sequence_number in range(len(heatmaps)))

        for subtask in heatmaps_sequence.keys():
            print(f"{len(heatmaps[sequence_number][subtask]):{num_zeros_heatmaps}} heatmaps for sequence {sequence_number:0{num_zeros_seqs}} and subtask {subtask}")
    
    return heatmaps


def _normalize_tensor_to_255(attns):
    attns = (attns - attns.min()) / (attns.max() - attns.min())
    attns = 255 * attns
    return attns


def _plot_heatmap(attns, resize_shape, x_labels=None, y_labels=None, blur=False):
    if blur:
        heatmap = cv2.resize(attns, resize_shape)
    else:
        heatmap = cv2.resize(attns, resize_shape, interpolation=cv2.INTER_NEAREST)
    
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if x_labels is not None and y_labels is not None:
        heatmap = _draw_token_labels_onto_heatmap(heatmap, x_labels, y_labels)
    
    return heatmap


def _draw_token_labels_onto_heatmap(heatmap, x_labels, y_labels):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 0)
    thickness = 1
    x_cell_height = max(cv2.getTextSize(x_label, font_face, font_scale, thickness)[0][0] for x_label in x_labels) # dynamic depending on longest x label
    y_cell_width = max(cv2.getTextSize(y_label, font_face, font_scale, thickness)[0][0] for y_label in y_labels) # dynamic depending on longest y label
    margin_heatmap_labels = 5 # no. of pixels between heatmap and labels
    x_labels_height = max(cv2.getTextSize(x_label, font_face, font_scale, thickness)[0][1] for x_label in x_labels) # dynamic depending on height of x labels
    y_labels_height = max(cv2.getTextSize(y_label, font_face, font_scale, thickness)[0][1] for y_label in y_labels) # dynamic depending on height of y labels
    
    heatmap_height, heatmap_width = heatmap.shape[:2]

    heatmap_canvas = np.ones((heatmap_height + x_cell_height + margin_heatmap_labels, heatmap_width + y_cell_width + margin_heatmap_labels, 3), dtype=np.uint8) * 255
    heatmap_canvas[:heatmap_height, -heatmap_width:] = heatmap # paste heatmap onto top right of canvas

    heatmap_canvas_rotated = cv2.rotate(heatmap_canvas, cv2.ROTATE_90_CLOCKWISE) # x labels are written rotated s.t. they fit onto canvas

    x_cell_width = round(heatmap_width / len(x_labels))
    x_cell_middle = x_labels_height + (x_cell_width - x_labels_height) // 2 - 1
    for i, label in enumerate(x_labels):
        x = max(0, x_cell_height - cv2.getTextSize(label, font_face, font_scale, thickness)[0][0]) # right align text w/ overflow protection
        y = y_cell_width + margin_heatmap_labels + i * x_cell_width + x_cell_middle
        cv2.putText(heatmap_canvas_rotated, label, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)

    heatmap_canvas = cv2.rotate(heatmap_canvas_rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)

    y_cell_height = round(heatmap_height / len(y_labels))
    y_cell_middle = y_labels_height + (y_cell_height - y_labels_height) // 2 - 1
    for i, label in enumerate(y_labels):
        x = max(0, y_cell_width - cv2.getTextSize(label, font_face, font_scale, thickness)[0][0]) # right align text w/ overflow protection
        y = i * y_cell_height + y_cell_middle
        cv2.putText(heatmap_canvas, label, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)
    
    return heatmap_canvas


def _store_heatmap(heatmap, output_path, sequence_number, task_number, subtask, step_number, layer, heatmap_name):
    attn_name = "attn"
    if "dec_self" in heatmap_name:
        attn_name = "dec_self"
    elif "dec_cross" in heatmap_name:
        attn_name = "dec_cross"
    elif "enc_image" in heatmap_name:
        attn_name = "enc_image"
    elif "enc_image2" in heatmap_name:
        attn_name = "enc_image2"
    elif "enc" in heatmap_name:
        attn_name = "enc"
    
    heatmap_path = f"{output_path}/seq-{sequence_number}/{task_number}-{subtask}/step-{step_number}/layer-{layer}/{attn_name}"
    os.makedirs(heatmap_path, exist_ok=True)
    cv2.imwrite(f"{heatmap_path}/{heatmap_name}.png", heatmap)


def _prepare_heatmaps_for_wandb(heatmap, img_name):
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # wandb expects RGB images
    return wandb.Image(heatmap, caption=img_name)


def _gen_heatmaps_for_layers(attns_layers, merge_attn_heads, num_heatmaps, output_dir, sequence_number, task_number, subtask, step_number,
                             heatmaps, attn_name, resize_shape=None, x_labels=None, y_labels=None, flow_step=None):
    if attn_name == "enc_image" or attn_name == "enc_image2":
        assert resize_shape is None and x_labels is None and y_labels is None, "If enc_iamge or enc_image2 attention, resize_shape, x_labels and y_labels must be None."

        img_layers = attns_layers["img"]
        attns_layers = torch.unsqueeze(attns_layers["attn"], 0) # only one attn => add list dim

        resize_shape = tuple(img_layers.shape[-2:]) # (H, W)
    else:
        assert resize_shape is not None and x_labels is not None and y_labels is not None, "If not enc_image or enc_image2 attention, resize_shape, x_labels and y_labels must not be None."

    for layer, attns_layer in enumerate(attns_layers):
        if num_heatmaps == 0:
            # if negative, all heatmaps are to be generated
            break

        if attn_name == "dec_self":
            attns_layer = attns_layer["self"]
        elif attn_name == "dec_cross":
            attns_layer = attns_layer["cross"]
        
        # specify labels and resize_shape for current attention layer
        if attn_name == "dec_cross" and layer == 0:
            x_labels = x_labels(attns_layer.shape[-1])
            resize_shape = resize_shape(attns_layer.shape[-1])
        elif attn_name == "enc" and layer == 0:
            x_labels = x_labels(attns_layer.shape[-1])
            y_labels = y_labels(attns_layer.shape[-1])
            resize_shape = resize_shape(attns_layer.shape[-1])

        attns_layer = attns_layer.cpu().detach().numpy()
        
        # normalize each head individually to [0, 255] range
        for head_number in range(attns_layer.shape[1]):
            attns_head = attns_layer[0][head_number]
            attns_layer[0][head_number] = _normalize_tensor_to_255(attns_head) # (Ta, Ta) | (Tb, Tb) | (Tc, Tc) | (Tc, Tb) | (Ts, Ts)

        if merge_attn_heads:
            attns_layer = attns_layer[0].sum(axis=0) # (Ta, Ta) | (Tb, Tb) | (Tc, Tc) | (Tc, Tb)
            attns_layer = _normalize_tensor_to_255(attns_layer).astype(np.uint8) # normalize after sum to ensure [0, 255] range

            attns_layer_heatmap = _plot_heatmap(attns_layer, resize_shape, x_labels, y_labels, blur=(attn_name == "enc_image" or attn_name == "enc_image2")) # (H, W, C)
            
            if attn_name == "enc_image" or attn_name == "enc_image2":
                attns_layer_heatmap = _overlay_heatmap_onto_image(img_layers, attns_layer_heatmap)

            if flow_step is None:
                attns_layer_heatmap_name = f"{attn_name}_layer-{layer}_merged-heads"
            else:
                attns_layer_heatmap_name = f"{attn_name}_flow-step-{flow_step}_layer-{layer}_merged-heads"
            
            _store_heatmap(attns_layer_heatmap, output_dir, sequence_number, task_number, subtask, step_number, layer, attns_layer_heatmap_name)

            attns_layer_heatmap_wandb = _prepare_heatmaps_for_wandb(attns_layer_heatmap, attns_layer_heatmap_name)
            heatmaps[sequence_number][subtask][step_number].append(attns_layer_heatmap_wandb)

            num_heatmaps -= 1
        else:
            attns_layer = attns_layer[0].astype(np.uint8) # (nh, Ts, Ts)

            for head_number, attns_layer_head in enumerate(attns_layer):
                if num_heatmaps == 0:
                    # if negative, all heatmaps are to be generated
                    break

                attns_layer_heatmap = _plot_heatmap(attns_layer_head, resize_shape, x_labels, y_labels, blur=(attn_name == "enc_image" or attn_name == "enc_image2")) # (H, W, C)

                if attn_name == "enc_image" or attn_name == "enc_image2":
                    attns_layer_heatmap = _overlay_heatmap_onto_image(img_layers, attns_layer_heatmap)
                
                if flow_step is None:
                    attns_layer_heatmap_name = f"{attn_name}_layer-{layer}_head-{head_number}"
                else:
                    attns_layer_heatmap_name = f"{attn_name}_flow-step-{flow_step}_layer-{layer}_head-{head_number}"
                
                _store_heatmap(attns_layer_heatmap, output_dir, sequence_number, task_number, subtask, step_number, layer, attns_layer_heatmap_name)

                attns_layer_heatmap_wandb = _prepare_heatmaps_for_wandb(attns_layer_heatmap, attns_layer_heatmap_name)
                heatmaps[sequence_number][subtask][step_number].append(attns_layer_heatmap_wandb)

                num_heatmaps -= 1
    return heatmaps, num_heatmaps


def _overlay_heatmap_onto_image(image, heatmap, alpha=0.3):
    if image.dim() == 4:
        image = image.squeeze(0)
    if image.dim() == 3 and image.shape[0] in [1,3]: # (C, H, W) -> (H, W, C)
        image = image.permute(1, 2, 0) 
    image = image.cpu().numpy()

    if image.dtype != np.uint8:
        # revert image transforms
        image = _revert_image_normalization(image)
        image = _normalize_tensor_to_255(image).astype(np.uint8)
        # skip resizing s.t. heatmaps are overlayed onto original image size
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert from RGB to BGR for OpenCV

    overlayed_image = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlayed_image


def _revert_image_normalization(image, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    # values for mean & std are from config calvin_transforms.yaml

    # reshape mean & std to match image shape (mean & std are automagically broadcasted to H & W dimensions of image)
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)

    image_unnormalized = image * std + mean

    return image_unnormalized
