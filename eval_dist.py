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
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from attack.attack import generate_attack


def dist_wrapper(model, ref_features):
    def get_dist(x):
        dist = 1 - nn.CosineSimilarity(dim=1, eps=1e-6)(model(x), ref_features)
        return torch.stack((1 - dist, dist), dim=1)

    return get_dist


def get_model(args):
    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.cuda()
    return model


def get_data(args):
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    return dataset_val


def dist_attack(args, model):
    imgs_per_chunk = 64  # num_test_images // num_chunks

    dataset_val = get_data(args)
    data_loader = torch.utils.data.DataLoader(
        dataset_val,
        sampler=None,
        batch_size=imgs_per_chunk,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    metric_logger = utils.MetricLogger(delimiter="  ")
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    distance_list = list()
    for idx, (x, _) in enumerate(metric_logger.log_every(data_loader, 100)):
        idx *= imgs_per_chunk
        x = x.cuda()
        original_features = model(x).detach()
        x_adv = generate_attack(attack=args.attack, eps=args.eps, model=model, x=x,
                                target=original_features)  # torch.zeros(x.shape[0]).cuda(), )
        features = model(x_adv).detach()
        distance_list.append(1 - cos_sim(features, original_features))

    root = args.load_features if args.load_features else args.dump_features
    torch.save(distance_list, os.path.join(root, 'distance_list.pt'))


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
                        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)

    parser.add_argument("--attack", default=None, type=str, help='Attack L2, Linf')
    parser.add_argument('--eps', default=1.0, type=float, help='Perturbation budget for attack')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing model ... ============
    model = get_model(args)
    dist_attack(args, model)
    dist.barrier()
