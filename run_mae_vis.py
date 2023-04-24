# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import glob
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model

from datasets import DataAugmentationForMAE
import modeling_pretrain

from utility import superpixel

def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--img_path', type=str, help='input image path')
    parser.add_argument('--save_path', type=str, help='save image path')
    parser.add_argument('--model_path', type=str, help='checkpoint path of model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--mask_mode', default='block', type=str, help='mask_mode: block or superpixel')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


def load_model(args):
    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

from utility import image_tool
import cv2
from utils import ToCVImage
from masking_generator import RandomMaskingGenerator


def main(args):
    print(args)
    model = load_model(args)
    device = torch.device(args.device)
    img_path = args.img_path
    patch_size = model.encoder.patch_embed.patch_size
    os.makedirs(args.save_path, exist_ok=True)

    img_list = [img_path]
    if os.path.isdir(img_path):
        img_list = glob.glob(os.path.join(args.img_path, '**/*.jpg'), recursive=True)

    ratios = np.arange(0.1, 1, 0.05)
    times = 3

    for img_path in img_list:
        with open(img_path, 'rb') as f:
            pl_img = Image.open(f)
            pl_img.convert('RGB')
        print("img path:", img_path)
        img_basename = os.path.basename(img_path).split('.')[0]

        titles = []
        vis = []
        transforms = DataAugmentationForMAE(args)
        cropped_cv_image, img, _, _, _ = transforms(pl_img)
        img = img.to(device, non_blocking=True)

        # save original img
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
        ori_img = img * std + mean  # in [0, 1]
        transed_image = img[None, :]

        for r in ratios:
            vis.append(ToCVImage()(ori_img[0, :]))
            titles.append(f'random_crop')
            mask_generator = RandomMaskingGenerator(args.window_size[0] * args.window_size[1], r)

            for t_i in range(times):
                # bool_masked_pos = mask_generator()

                label_map = transforms.get_super_pix_label_map(cropped_cv_image)
                masked_map, masked_map_for_vis_block, bool_masked_pos = transforms.get_superpixel_info(cropped_cv_image, label_map,  r, mode='all')
                m_v = rearrange(masked_map_for_vis_block, 'h w c -> 1 c h w')
                m_v = torch.from_numpy(m_v).to(device, non_blocking=True)

                image = transed_image
                if args.mask_mode == 'superpixel':
                    image = transed_image.clone()
                    image[m_v] = 0
                else:
                    bool_masked_pos = mask_generator()

                bool_masked_pos = torch.from_numpy(bool_masked_pos).to(device, non_blocking=True)


                with torch.no_grad():
                    bool_masked_pos = bool_masked_pos[None, :]
                    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                    outputs = model(image, bool_masked_pos)

                    img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0],
                                            p2=patch_size[0])
                    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
                                img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')

                    if args.mask_mode == 'superpixel':
                        img_p_ii = (bool_masked_pos + 1).to(torch.bool)
                        img_patch[img_p_ii] = outputs
                    else:
                        img_patch[bool_masked_pos] = outputs

                    # make mask
                    H, W = int(ori_img.shape[2] / patch_size[0]), int(ori_img.shape[3] / patch_size[1])

                    # save reconstruction img
                    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
                    # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                    rec_img = rec_img * (
                                img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(
                        dim=-2, keepdim=True)
                    rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0],
                                        p2=patch_size[1], h=H, w=W)
                    vis.append(ToCVImage()(rec_img[0, :].clip(0, 0.996)))
                    titles.append(f'{r:.2f}_{t_i}_rec')

                    # save random mask img
                    mask = torch.ones_like(img_patch)
                    mask[bool_masked_pos] = 0
                    mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
                    mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1],
                                     h=H, w=W)
                    img_mask = rec_img * mask

                    cv_img_mask = ToCVImage()(img_mask[0, :])

                    if args.mask_mode == 'superpixel':
                        m_v = rearrange(masked_map_for_vis_block, 'h w c -> 1 c h w')
                        m_v = torch.from_numpy(m_v).to(device, non_blocking=True)
                        img_mask = ori_img.clone()
                        img_mask[m_v] = 0
                        cv_img_mask = ToCVImage()(img_mask[0, :])
                        # superpixel.vis_superpixel(cv_img_mask, label_map, color=(0,255,255))

                    vis.append(cv_img_mask)
                    titles.append(f'{r:.2f}_{t_i}_mask')

        vis_img = image_tool.my_plt_render(vis, titles, cols=times * 2 + 1, cell_hw=(300, 300), fast_vis=True)
        cv2.imwrite(f"{args.save_path}/{img_basename}_vis_img.jpg", vis_img)


if __name__ == '__main__':
    opts = get_args()
    main(opts)
