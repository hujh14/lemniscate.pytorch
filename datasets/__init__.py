from .folder import ImageFolderInstance
from .cifar import CIFAR10Instance, CIFAR100Instance
from .mnist import MNISTInstance
from .coco import COCODataset
from .coco_iou import COCOIOUDataset

__all__ = ('ImageFolderInstance', 'MNISTInstance', 'CIFAR10Instance', 'CIFAR100Instance', 'COCODataset', 'COCOIOUDataset')

