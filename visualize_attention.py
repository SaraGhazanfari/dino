# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import argparse
from pathlib import Path

# import cv2
import random
import colorsys

from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn

from torchvision import transforms as pth_transforms
import numpy as np

import vision_transformer as vits
from attack.attack import generate_attack
# from dino_utils.visualization_utils import visualize_att_map


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def main(ref_img, dataloader, args, model, device):
    distance_list = list()
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    for idx, data in tqdm(enumerate(dataloader)):
        inputs = data[0].to(device)
        input_embed = model(inputs)
        distance_list.extend(cos_sim(ref_img, input_embed[0:]))
    torch.save(distance_list, 'inter_class_distance_list.pt')
    #     if idx * args.batch_size > 5:
    #         break
    #     inputs = data[0].to(device)
    #     input_embed = model(inputs)  # .detach()
    #     output_dir = f'{args.model_name}/clean/{idx}'
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
    #     visualize_att_map(inputs.squeeze(0), img_idx=idx, model=model, device=device, patch_size=args.patch_size,
    #                       output_dir=output_dir)
    #     adv_inputs = generate_attack(attack=args.attack, eps=args.eps, x=inputs, target=input_embed, model=model)
    #
    #     adv_input_embed = model(adv_inputs)  # .detach()
    #
    #     cos_dist = 1 - cos_sim(input_embed.unsqueeze(0), adv_input_embed.unsqueeze(0))
    #
    #     distance_list.append(cos_dist)
    #     output_dir = f'{args.model_name}/adv/{idx}'
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
    #     visualize_att_map(adv_inputs.squeeze(0), img_idx=idx, model=model, device=device,
    #                       patch_size=args.patch_size,
    #                       output_dir=output_dir)
    #
    # torch.save(model, f'{args.model_name}/distance_list_{args.attack}_{args.eps}')


def load_dino_model():
    global p
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    return model


def load_data(args):
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
    ])
    dataset_val = ImageFolder(os.path.join(args.data_path, "val"), transform=transform)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    img_ref = transform(Image.open('ref.png', mode='r'))
    return data_loader_val, img_ref


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--model_name', default='dino', type=str,
                        choices=['dino', 'clip', 'open_clip'], help='ViT based model to use')
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
                        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--data_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument("--attack", default='PGD-L2', type=str, help='Attack L2, Linf')
    parser.add_argument('--eps', default=0.5, type=float, help='Perturbation budget for attack')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataloader, img_ref = load_data(args)
    # build model
    if args.model_name == 'dino':
        model = load_dino_model()

    # main(dataloader, args, dino_model, 'dino_model', device)
    else:
        from transformers import CLIPModel

        model = vits.VisionTransformer(
            **CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model.state_dict())
    img_ref_embed = model(img_ref).unsqueeze(0)
    main(img_ref_embed, dataloader, args, model, device)
