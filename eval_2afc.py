import argparse
import sys
import torch
from torch import nn
from torch.backends import cudnn
from tqdm import tqdm

import utils
from data.night_dataset import NightDataset
import vision_transformer as vits
from torchvision import models as torchvision_models


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


def dreamsim_eval(model, root_dir, batch_size):
    data_loader, dataset_size = NightDataset(root_dir=root_dir, batch_size=batch_size,
                                             split='test_imagenet').get_dataloader()
    no_imagenet_data_loader, no_imagenet_dataset_size = NightDataset(root_dir=root_dir,
                                                                     batch_size=batch_size,
                                                                     split='test_no_imagenet').get_dataloader()
    print(len(data_loader), len(no_imagenet_data_loader))
    imagenet_score = get_2afc_score_eval(model, data_loader)
    print(f"ImageNet 2AFC score: {str(imagenet_score)}")
    torch.cuda.empty_cache()
    no_imagenet_score = get_2afc_score_eval(model, no_imagenet_data_loader)
    print(f"No ImageNet 2AFC score: {str(no_imagenet_score)}")
    overall_score = (imagenet_score * dataset_size + no_imagenet_score * no_imagenet_dataset_size) / (
            dataset_size + no_imagenet_dataset_size)
    print(f"Overall 2AFC score: {str(overall_score)}")


def one_step_2afc_score_eval(model, img_ref, img_left, img_right, target):
    # if self.config.attack:
    #     img_ref = self.generate_attack(img_ref, img_left, img_right, target, target_model=self.model_wrapper())
    dist_0, dist_1, _ = get_cosine_score_between_images(model, img_ref, img_left, img_right)
    if len(dist_0.shape) < 1:
        dist_0 = dist_0.unsqueeze(0)
        dist_1 = dist_1.unsqueeze(0)
    dist_0 = dist_0.unsqueeze(1)
    dist_1 = dist_1.unsqueeze(1)
    target = target.unsqueeze(1)
    return dist_0, dist_1, target


def get_2afc_score(d0s, d1s, targets):
    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores)
    return twoafc_score


def get_2afc_score_eval(model, test_loader):
    print("Evaluating NIGHTS dataset.")
    d0s = []
    d1s = []
    targets = []
    # with torch.no_grad()
    for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img_ref, img_left, img_right, target = img_ref.cuda(), img_left.cuda(), \
            img_right.cuda(), target.cuda()
        dist_0, dist_1, target = one_step_2afc_score_eval(model, img_ref, img_left, img_right, target)
        d0s.append(dist_0)
        d1s.append(dist_1)
        targets.append(target)

    twoafc_score = get_2afc_score(d0s, d1s, targets)
    return twoafc_score


def get_cosine_score_between_images(model, img_ref, img_left, img_right, requires_grad=False,
                                    requires_normalization=False):
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    embed_ref = model(img_ref)
    if not requires_grad:
        embed_ref = embed_ref.detach()
    embed_x0 = model(img_left).detach()
    embed_x1 = model(img_right).detach()
    if requires_normalization:
        norm_ref = torch.norm(embed_ref, p=2, dim=(1)).unsqueeze(1)
        embed_ref = embed_ref / norm_ref
        norm_x_0 = torch.norm(embed_x0, p=2, dim=(1)).unsqueeze(1)
        embed_x0 = embed_x0 / norm_x_0
        norm_x_1 = torch.norm(embed_x1, p=2, dim=(1)).unsqueeze(1)
        embed_x1 = embed_x1 / norm_x_1

    bound = torch.norm(embed_x0 - embed_x1, p=2, dim=(1)).unsqueeze(1)
    dist_0 = 1 - cos_sim(embed_ref, embed_x0)
    dist_1 = 1 - cos_sim(embed_ref, embed_x1)
    return dist_0, dist_1, bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
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
    dreamsim_eval(model, root_dir=args.data_path, batch_size=args.batch_size_per_gpu)
