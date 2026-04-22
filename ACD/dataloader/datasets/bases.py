from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
from copy import deepcopy
import random
import torch


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
          
        return img, pid, camid, img_path.split('/')[-1]

# class SyncMultipleApply:
#     """同步随机增强：保证同一张图像在多个 transform 下随机操作一致"""
#     def __init__(self, transforms_list):
#         self.transforms_list = transforms_list

#     def __call__(self, image):
#         # 保存当前随机状态
#         py_random_state = random.getstate()
#         torch_random_state = torch.get_rng_state()

#         outputs = []
#         for t in self.transforms_list:
#             random.setstate(py_random_state)
#             torch.set_rng_state(torch_random_state)
#             outputs.append(t(deepcopy(image)))
#         return outputs


class Distillation_ImageDataset(Dataset): 
    def __init__(self, dataset, s_transform=None, t_transform=None):
        if s_transform is None or t_transform is None:
            raise ValueError("Both 's_transform' and 't_transform' must be provided and cannot be None.")
        
        self.dataset = dataset
        # self.sync_augment = sync_augment

       
        self.s_transform = s_transform
        self.t_transform = t_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        # if self.sync_augment:
        #     s_img, t_img = self.multi_transform(img)
        # else:
        s_img = self.s_transform(img)
        t_img = self.t_transform(img)

        return s_img, t_img, pid, camid, img_path.split('/')[-1]