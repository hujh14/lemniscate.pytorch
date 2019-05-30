import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask


ADE_GT = [
    "data/ade20k/annotations/instances_train.json",
    "data/ade20k/annotations/instances_val.json",
]

class SimulatedAnnotator:

    def __init__(self, gt_paths=None):
        self.thresholdIOU = 0.8
        self.gt_paths = gt_paths
        if gt_paths is None:
            self.gt_paths = ADE_GT

        self.setup()

    def setup(self):
        self.filenameToCoco = {}
        self.filenameToImg = {}
        self.catnameToCat = {}

        for gt_path in self.gt_paths:
            coco = COCO(gt_path)
            for imgId in coco.imgs:
                img = coco.imgs[imgId]
                filename = img["file_name"]
                self.filenameToCoco[filename] = coco
                self.filenameToImg[filename] = img
                self.filenameToCoco["ade_challenge/images/" + filename] = coco
                self.filenameToImg["ade_challenge/images/" + filename] = img

            for catId in coco.cats:
                cat = coco.cats[catId]
                catname = cat["name"]
                if catname not in self.catnameToCat:
                    self.catnameToCat[catname] = cat
                else:
                    if cat["id"] != self.catnameToCat[catname]["id"]:
                        print("Simulated annotator does not support misaligned categories")

    def compute_iou(self, coco, ann):
        img = coco.imgs[ann["image_id"]]
        cat = coco.cats[ann["category_id"]]
        if img["file_name"] not in self.filenameToImg:
            return None

        # coco ids do not match
        cocoGt = self.filenameToCoco[img["file_name"]]
        imgGt = self.filenameToImg[img["file_name"]]
        catGt = self.catnameToCat[cat["name"]]
        annIdsGt = cocoGt.getAnnIds(imgIds=[imgGt["id"]], catIds=[catGt["id"]])
        annsGt = cocoGt.loadAnns(annIdsGt)

        gts = [ann["segmentation"] for ann in annsGt]
        dts = [ann["segmentation"]]
        iscrowds = [0 for _ in gts]
        if len(gts) == 0 or len(dts) == 0:
            return 0

        ious = COCOmask.iou(dts, gts, iscrowds)
        iou = float(np.max(ious))
        return iou

    def pass_fail(self, coco, ann):
        iou = self.compute_iou(coco, ann)
        if iou == None:
            return False
        return iou > self.thresholdIOU

