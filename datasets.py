# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform

from masking_generator import RandomMaskingGenerator
from dataset_folder import ImageFolder
import cv2
import numpy as np
from utility import image_tool, superpixel
from einops import rearrange
from fast_slic.avx2 import SlicAvx2


class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        # self.crop_resize_transform = transforms.RandomResizedCrop(args.input_size)
        self.crop_resize_transform = transforms.Resize(args.input_size)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size[0] * args.window_size[1], args.mask_ratio
        )
        self.args = args

    def get_superpixel_info(self, resized_image, mask_ratio, mode='vis'):
        """
        about mode
        - vis return (1 - mask_ratio) vis block and some ones may be masked
        - all return 100% vis block, all_block may be effect

        :param resized_image:
        :type resized_image:
        :param mask_ratio:
        :type mask_ratio:
        :param mode:
        :type mode:
        :return:
        :rtype:
        """
        h, w, _ = resized_image.shape
        rz = self.args.patch_size[0]
        win_h, win_w = self.args.window_size
        # slic = cv2.ximgproc.createSuperpixelLSC(img, region_size=rz)
        # slic = cv2.ximgproc.createSuperpixelSLIC(resized_image, region_size=rz, algorithm=cv2.ximgproc.MSLIC)
        # slic.iterate(10)  # 迭代次数，越大效果越好
        # label_slic = slic.getLabels()  # 获取超像素标签

        slic = SlicAvx2(num_components=1024, compactness=10)
        label_slic = slic.iterate( cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB))

        if mode == 'vis':
            masked_position_generator = RandomMaskingGenerator(win_h * win_w, mask_ratio)
            masked_position = masked_position_generator()

            label_slic = label_slic + 1
            label_patch = rearrange(label_slic, '(h p1) (w p2) ->(h w) (p1 p2)', p1=rz, p2=rz)
            label_path2 = label_patch.copy()
            max_cls = np.max(label_slic)

            for i, m in enumerate(masked_position):
                cls_arr, num = np.unique(label_patch[i], return_counts=True)
                cls = cls_arr[np.argmax(num)]
                if m > 0:
                    label_patch[np.where(label_patch == cls)] = 0
                    label_path2[np.where(label_path2 == cls)] = 0
                    label_path2[i] = max_cls + 1
            label_slic = rearrange(label_patch, '(h w) (p1 p2) -> (h p1) (w p2)', h=self.args.window_size[0], p1=rz)
            label_slic2 = rearrange(label_path2, '(h w) (p1 p2) -> (h p1) (w p2)', h=self.args.window_size[0], p1=rz)
            masked_map = np.array(label_slic == 0)
            masked_map_for_vis_block = np.array(label_slic2 == 0)

            masked_map = superpixel.expand_map_to_img(masked_map, c=3)
            masked_map_for_vis_block = superpixel.expand_map_to_img(masked_map_for_vis_block, c=3)

            return masked_map, masked_map_for_vis_block, masked_position

        elif mode == 'all':
            label_slic = label_slic + 1
            max_cls = np.max(label_slic)
            cl_numbers = np.max(label_slic) - np.min(label_slic)
            masked_position = RandomMaskingGenerator(cl_numbers, mask_ratio=mask_ratio)()
            for i, m in enumerate(masked_position):
                _cls = i + 1
                if m < 1:
                    continue
                label_slic[np.where(label_slic == _cls)] = 0
            masked_map = np.array(label_slic == 0)
            masked_position = RandomMaskingGenerator(win_h * win_w, mask_ratio=0)()
            masked_map = superpixel.expand_map_to_img(masked_map, c=3)
            masked_map_for_vis_block = masked_map

            return masked_map, masked_map_for_vis_block, masked_position

    def __call__(self, image):
        # pillow image
        cropped_pl_image = self.crop_resize_transform(image)

        mask_mode = self.args.mask_mode
        rz = self.args.patch_size[0]
        mask_ratio = self.args.mask_ratio

        cropped_cv_image = np.array(cropped_pl_image)[:, :, ::-1].copy()

        if mask_mode == 'block':
            masked_position = self.masked_position_generator()
            mask_map = np.zeros(cropped_cv_image.shape[:2])
            mask_patches = rearrange(mask_map, '(h p1) (w p2) -> (h w) (p1 p2)', p1=rz, p2=rz)
            new_mask_map = np.zeros_like(mask_patches)
            new_mask_map[masked_position.astype(np.bool8)] = 1
            masked_map = rearrange(new_mask_map.astype(bool), '(h w) (p1 p2) -> (h p1) (w p2)', h=cropped_cv_image.shape[0]//rz, p1=rz)
            masked_map_for_vis_block = np.zeros_like(masked_map, dtype=np.bool8)

            masked_map = superpixel.expand_map_to_img(masked_map, c=3)
            masked_map_for_vis_block = superpixel.expand_map_to_img(masked_map_for_vis_block, c=3)

        elif mask_mode == 'superpixel':
            masked_map, masked_map_for_vis_block, masked_position = self.get_superpixel_info(cropped_cv_image, mask_ratio, mode='all')
        else:
            raise Exception(f'unknown mask_mode:{mask_mode}')

        self.masked_vis_image = cropped_cv_image.copy()
        self.masked_all_image = cropped_cv_image.copy()
        self.masked_vis_image[masked_map_for_vis_block[:, :, 0]] = (0, 0, 0)
        self.masked_all_image[masked_map[:, :, 0]] = (255, 0, 255)


        # image_tool.show_cvimg_in_sciview(self.masked_vis_image, self.masked_all_image)

        res_ori_img = self.transform(cropped_cv_image[:, :, ::-1].copy())

        rtn = [cropped_cv_image, res_ori_img, masked_position,  masked_map, masked_map_for_vis_block]
        return rtn

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == '__main__':
    from PIL import Image
    from argparse import Namespace
    args = Namespace(input_size = 224, window_size=14)
    img_path = 'datasets01/all_text_db_full_size/test/img_1.jpg'
    with open(img_path, 'rb') as f:
        pl_img = Image.open(f)
        pl_img.convert('RGB')
    transforms = DataAugmentationForMAE(args)

    cropped_cv_image, img, masked_position, masked_map, masked_map_for_vis_block = transforms(pl_img)
    image_tool.show_cvimg_in_sciview(cropped_cv_image)
    print('ok')
