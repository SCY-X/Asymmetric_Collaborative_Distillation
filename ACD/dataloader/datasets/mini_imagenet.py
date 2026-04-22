import os.path as osp
import os
import glob
from .bases import BaseImageDataset

class mini_imagenet(BaseImageDataset): 
    dataset_dir = 'mini_imagenet'

    def __init__(self, root='../', verbose=True, **kwargs):
        super(mini_imagenet, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'val')
        self.gallery_dir = osp.join(self.dataset_dir, 'val')

        self._check_before_run()

        # 创建 nXXXX -> label 映射（基于训练集）
        self.class_to_label_train = self._create_class_mapping(self.train_dir)
        self.class_to_label_val = self._create_class_mapping(self.query_dir)

        # 处理数据集
        train = self._process_dir_train(self.train_dir)
        query = self._process_dir_val(self.query_dir)
        gallery = self._process_dir_val(self.gallery_dir)

        if verbose:
            print("=> ImageNet loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))


    def _create_class_mapping(self, data_dir):
        """
        创建 nXXXX -> label 映射
        """
        class_folders = sorted(os.listdir(data_dir))
        class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_folders)}
        return class_to_label

    def _process_dir_train(self, dir_path):
        """
        dir_path: 数据集目录，返回 [(img_path, label, camid), ...]
        """
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        dataset = []
        for camid, img_path in enumerate(img_paths):
            cls_name = osp.basename(osp.dirname(img_path))  # nXXXX
            pid = self.class_to_label_train[cls_name]            # 转为整数 label
            dataset.append((img_path, pid, camid))

        return dataset
    
    def _process_dir_val(self, dir_path):
        """
        dir_path: 数据集目录，返回 [(img_path, label, camid), ...]
        """
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        dataset = []
        for camid, img_path in enumerate(img_paths):
            cls_name = osp.basename(osp.dirname(img_path))  # nXXXX
            pid = self.class_to_label_val[cls_name]            # 转为整数 label
            dataset.append((img_path, pid, camid))

        return dataset