import os.path as osp
import glob
import re
from .bases import BaseImageDataset


class DeepFashion(BaseImageDataset):


    dataset_dir = 'DeepFashion'

    def __init__(self, root='../', verbose=True, **kwargs):
        super(DeepFashion, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()


        train = self._process_dir(self.train_dir, relabel=True, mode='train')
        query = self._process_dir(self.query_dir, relabel=False, mode='query')
        gallery = self._process_dir(self.gallery_dir, relabel=False, mode='gallery')

        if verbose:
            print("=> DeepFashion2 loaded")
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

    def _process_dir(self, dir_path, relabel=False,  mode='train'):

        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            pid = int(osp.basename(osp.dirname(img_path)).split("_")[0]) - 1
            #if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid = int(osp.basename(osp.dirname(img_path)).split("_")[0]) - 1
            
            if mode == 'train' or mode == 'query':
                camid = 0
            else:
                camid = 1
    
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset