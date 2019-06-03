
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import random
import numpy as np

import pycocotools.mask as mask_utils
from .annotator_sim import SimulatedAnnotator

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCOIOUDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, cat_name=None, transform=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = list(sorted(self.coco.anns.keys()))
        self.transform = transform

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.category_name_to_category_id = {
            cat["name"]: cat["id"] for cat in self.coco.dataset["categories"]
        }

        self.cat_name = cat_name
        self.area_threshold = 1000
        self.compute_ious()
        self.filter_ids()
        self.set_targets()

    def compute_ious(self):
        print("Computing ious...")
        annotator = SimulatedAnnotator()
        ids = []
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            ann["iou"] = annotator.compute_iou(self.coco, ann)
            if ann["iou"] != 0:
                ids.append(ann_id)
        self.ids = ids
        print("Done")

    def filter_ids(self):
        # filter bad annotations
        ids = []
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            if has_valid_annotation([ann]) and ann["area"] > self.area_threshold:
                ids.append(ann_id)
        self.ids = ids

        # filter category
        if self.cat_name != None:
            ids = []
            cat_id = self.category_name_to_category_id[self.cat_name]
            for ann_id in self.ids:
                ann = self.coco.anns[ann_id]
                if ann["category_id"] == cat_id:
                    ids.append(ann_id)
            self.ids = ids

        # filter zero ious
        ids = []
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            if ann["iou"] != 0:
                ids.append(ann_id)
        self.ids = ids

        # correct image path
        for img_id in self.coco.imgs:
            img = self.coco.imgs[img_id]
            if "ade_challenge/images/" in img["file_name"]:
                img["file_name"] = img["file_name"].replace("ade_challenge/images/", "")

    def set_targets(self):
        self.targets = []
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            target = self.prepare_target(ann)
            self.targets.append(target)

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        img = self.coco.imgs[ann["image_id"]]

        input = self.prepare_input(img, ann)
        target = self.prepare_target(ann)
        return input, target, idx

    def prepare_input(self, img, ann):
        image = cv2.imread(os.path.join(self.root, img["file_name"]))[:,:,::-1]
        mask = mask_utils.decode(ann["segmentation"])  # [h, w, n]
        bbox = ann["bbox"]

        image = crop_square(image, bbox)
        mask = crop_square(mask, bbox)
        mask *= 255

        if self.transform is not None:
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            # Hack to ensure transform is the same
            seed = random.randint(0,2**32)
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            mask = self.transform(mask)
            image = torch.cat([image, mask])
        return image

    def prepare_target(self, ann):
        if self.cat_name == None:
            cat_label = self.json_category_id_to_contiguous_id[ann["category_id"]]
            return cat_label

        iou_bin = 0
        if ann["iou"] != None:
            iou_bin = int(ann["iou"]*10) + 1
        return iou_bin

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        img = self.coco.imgs[ann["image_id"]]
        return img

    def get_cat_info(self, idx):
        cat_id = self.contiguous_category_id_to_json_id[idx]
        cat = self.coco.cats[cat_id]
        return cat

def crop_square(image, bbox, margin=2, crop_size=256):
    x, y, w, h = bbox
    x_c = x + w/2
    y_c = y + h/2
    m = max(h,w) * margin
    m = min(m, max(image.shape[0], image.shape[1]))
    m = int(m / 2) * 2 # Needs to be even
    x0 = int(min(max(0, x_c - m/2), image.shape[1]))
    x1 = int(min(max(0, x_c + m/2), image.shape[1]))
    y0 = int(min(max(0, y_c - m/2), image.shape[0]))
    y1 = int(min(max(0, y_c + m/2), image.shape[0]))
    crop = image[y0:y1, x0:x1]

    # Pad with zeros
    if image.ndim == 2:
        pad = np.zeros((m, m), dtype='uint8')
    else:
        pad = np.zeros((m, m, 3), dtype='uint8')
    pad_l = int(-min(x_c - m/2, 0))
    pad_t = int(-min(y_c - m/2, 0))
    pad[pad_t:pad_t + crop.shape[0], pad_l:pad_l + crop.shape[1]] = crop
    resized = cv2.resize(pad, dsize=(crop_size, crop_size))
    return resized

_GREEN = (18, 127, 15)
def vis_mask(img, mask, alpha=0.4, color=_GREEN):
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * np.array(color)
    img = img.astype(np.uint8)
    return img

if __name__ == '__main__':
    data_dir = "./data"
    root = os.path.join(data_dir, "ade20k/images")
    val_ann_file = os.path.join(data_dir, "ade20k/annotations/predictions_val.json")

    dataset = COCODataset(
        val_ann_file, root,
        cat_name="car",
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    print("Dataset size:", len(dataset))
    print(dataset.train_labels)

    for inp, target, idx in dataset:
        print("Input shape:", inp.shape)
        print("Target:", target)
        print("Index:", idx)
        print("Img info", dataset.get_img_info(idx))

        inp = inp.numpy()
        image = np.array(inp[:3,:,:] * 255, dtype='uint8')
        mask = np.array(inp[3,:,:] * 255, dtype='uint8')
        image = np.transpose(image, (1,2,0))

        image_vis = vis_mask(image, mask)
        image_vis = Image.fromarray(image_vis)
        if target <= 5:
            image_vis.show()
            input("Press Enter to continue...")


