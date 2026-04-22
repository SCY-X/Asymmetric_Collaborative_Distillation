import os.path as osp
import glob
import re
from .bases import BaseImageDataset


class INat2021(BaseImageDataset):
    """
    iNaturalist 2021 Mini Dataset (示例).
    - 训练/验证目录结构:
        train/
            00000_Animalia_xxx/
                img1.jpg
                img2.jpg
            00001_Plantae_xxx/
                img3.jpg
                ...
        val/
            00000_Animalia_xxx/
                ...
    """

    dataset_dir = 'iNaturalist_2021'

    def __init__(self, root='../', verbose=True, **kwargs):
        super(INat2021, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=False)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> iNaturalist-2021 loaded")
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

    def _process_dir(self, dir_path, relabel=False):

        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            pid = int(osp.basename(osp.dirname(img_path)).split("_")[0])
            #if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for camid, img_path in enumerate(img_paths):
            pid = int(osp.basename(osp.dirname(img_path)).split("_")[0])
            assert 0 <= pid <= 9999  # pid == 0 means background
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
