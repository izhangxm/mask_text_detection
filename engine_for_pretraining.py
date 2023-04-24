# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.utils.data
from utils import ToCVImage
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utility import image_tool
import numpy as np

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
                    normlize_target: bool = True, log_writer:utils.TensorboardLogger=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, mask_mode='block'):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # _, images, bool_masked_pos, masked_map = batch
        ori_cv_img, transed_images, bool_masked_pos, masked_map, masked_map_for_vis_block = batch

        transed_images = transed_images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        bool_masked_map = masked_map.to(device, non_blocking=True).to(torch.bool)
        masked_map_for_vis_block = masked_map_for_vis_block.to(device, non_blocking=True).to(torch.bool)

        masked_map_for_vis_block = rearrange(masked_map_for_vis_block, 'b h w c -> b c h w')

        images = transed_images
        if mask_mode == 'superpixel':
            images = transed_images.clone()
            images[masked_map_for_vis_block] = 0
        # img0 = ori_cv_img[0].cpu().numpy()

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = transed_images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

        B, _, C = images_patch.shape
        with torch.cuda.amp.autocast():
            outputs = model(images, bool_masked_pos, 'unshuffled_all')
            bool_masked_patch = rearrange(bool_masked_map, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = images_patch[bool_masked_patch]
            masked_outputs = outputs[bool_masked_patch]
            loss = loss_func(input=masked_outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        with torch.no_grad():
            B, C, H, W = images.shape
            # outputs: 2 196 768
            rec_imgs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
            # rec_img: 2 196 256 3
            rec_imgs = rec_imgs * (images_squeeze.var(dim=-2, unbiased=True,  keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
            rec_imgs = rearrange(rec_imgs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size,  h=H // patch_size, w=W // patch_size)
            # rec_imgs: 2 3 224 224
            vis = ToCVImage()(rec_imgs[0, :].detach())

            rec_imgs_recover = rec_imgs.clone()
            bool_masked_map = rearrange(bool_masked_map, 'b h w c -> b c h w')
            unnorm_images = transed_images * std + mean  # in [0, 1]
            try:
                rec_imgs_recover[~bool_masked_map] = unnorm_images[~bool_masked_map]
            except Exception as e:
                pass
                print(e)

            vis2 = ToCVImage()(rec_imgs_recover[0, :].detach())

            masked_vis = rec_imgs_recover.clone()
            masked_vis[bool_masked_map] = 0
            vis3 = ToCVImage()(masked_vis[0, :].detach())
            vis4 = ori_cv_img[0].cpu().numpy()

            row1 = np.concatenate([vis, vis2], axis=1)
            row2 = np.concatenate([vis3, vis4], axis=1)
            full_vis = np.concatenate([row1, row2], axis=0)

            titles = ['outputs', 'recover_vis', 'ori_masked', 'ori']
            # image_tool.show_cvimg_in_sciview(full_vis)
            log_img = torch.tensor(full_vis).permute(2, 0, 1)


        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=float(f'{loss_value:.8f}'))
        metric_logger.update(loss_scale=float(f'{loss_scale_value:.8f}'))
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            if step % print_freq == 0:
                log_writer.update_image(decoder=log_img)

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
