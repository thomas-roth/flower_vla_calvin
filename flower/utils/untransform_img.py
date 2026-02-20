from pathlib import Path
import sys
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image


sys.path.append(str(Path(__file__).absolute().parents[2]))


def read_transform_cfg():
    cfg_path = str(Path(__file__).absolute().parents[2]) + "/conf/datamodule/transforms/calvin_transforms.yaml"
    with open(cfg_path, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    return cfg.val


def transform(img, cam, cfg=None):
    if cfg is None:
        cfg = read_transform_cfg()
    
    if cam == "static":
        cfg = cfg.rgb_static
    elif cam == "gripper":
        cfg = cfg.rgb_gripper
    else:
        raise ValueError(f"Unknown camera type: {cam}")

    for transform_cfg in cfg:
        transform = hydra.utils.instantiate(transform_cfg)
        img = transform(img)

    return img


def untransform(img, cam, cfg=None):
    if cfg is None:
        cfg = read_transform_cfg()
    
    if cam == "static":
        cfg = cfg.rgb_static
        resize_dim = 200
    elif cam == "gripper":
        cfg = cfg.rgb_gripper
        resize_dim = 84
    else:
        raise ValueError(f"Unknown camera type: {cam}")

    for transform_cfg in reversed(cfg):
        target = transform_cfg._target_

        if "Normalize" in target:
            mean = torch.tensor(list(transform_cfg.mean)).view(-1, 1, 1).to(img.device)
            std = torch.tensor(list(transform_cfg.std)).view(-1, 1, 1).to(img.device)
            img = img * std + mean

        elif "ScaleImageTensor" in target:
            img = img * 255.0

        elif "Resize" in target:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            if img.dim() == 5 and img.shape[0] == 1:
                img = img.squeeze(0)

            img = torch.nn.functional.interpolate(img, size=resize_dim, mode="bilinear", align_corners=False)

            if img.dim() == 4 and img.shape[0] == 1:
                img = img.squeeze(0)

    return img.clamp(0, 255).byte()


def plot_tensor(img_tensor, save_path="untransformed_img.png"):
    if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.shape[0] == 3:
        img_tensor = img_tensor.permute(1, 2, 0)
    
    assert img_tensor.dim() == 3 and img_tensor.shape[2] == 3, f"Expected image tensor of shape (H, W, 3), but got {img_tensor.shape}"
    
    img = img_tensor.cpu().numpy()
    Image.fromarray(img).save(save_path)


def test_untransform():
    episode = np.load("/DATA/calvin/task_ABC_D/validation/episode_0406364.npz", allow_pickle=True)
    img_static = torch.from_numpy(episode["rgb_static"]).to(torch.uint8)

    plot_tensor(img_static, save_path="img_static_original.png")

    cfg = read_transform_cfg()

    img_static_transformed = transform(img_static.permute(2, 0, 1), cam="static", cfg=cfg)
    img_static_untransformed = untransform(img_static_transformed, cam="static", cfg=cfg)

    plot_tensor(img_static_untransformed, save_path="img_static_untransformed.png")


if __name__ == "__main__":
    test_untransform()
