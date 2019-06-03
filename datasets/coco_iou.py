
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np

from coco import COCODataset, vis_mask
from annotator_sim import SimulatedAnnotator


class COCOIOUDataset(COCODataset):
    def __init__(
        self, ann_file, root, cat_name=None, transform=None
    ):
        super(COCOIOUDataset, self).__init__(root, ann_file, cat_name=cat_name, transform=transform)
        self.compute_ious()

    def compute_ious(self):
        print("Computing ious...")
        annotator = SimulatedAnnotator()
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            ann["iou"] = annotator.compute_iou(self.coco, ann)

    def prepare_target(self, ann):
        iou_bin = 0
        if ann["iou"] != None:
            iou_bin = int(ann["iou"]*10) + 1
        return iou_bin

    def get_label(self, target):
        label = "[{}, {}]".format((target-1)/10, target/10)
        return label

if __name__ == '__main__':
    ade_root = "./data/ade20k/images"
    ade_train_ann_file = "./data/ade20k/annotations/predictions_train.json"
    ade_val_ann_file = "./data/ade20k/annotations/predictions_val.json"

    dataset = COCOIOUDataset(
        ade_val_ann_file, ade_root,
        cat_name="car",
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    print("Dataset size:", len(dataset))
    print(dataset.get_targets())

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

        image_vis.show()
        input("Press Enter to continue...")
