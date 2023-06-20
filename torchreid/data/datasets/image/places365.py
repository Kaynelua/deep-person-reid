from __future__ import division, print_function, absolute_import
import os
import glob
import os.path as osp
import gdown

from ..dataset import ImageDataset


class Places365(ImageDataset):
    """University-1652.

    Reference:
        - Zheng et al. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization. ACM MM 2020.

    URL: `<https://github.com/layumi/University1652-Baseline>`_
    OneDrive:
    https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/Ecrz6xK-PcdCjFdpNb0T0s8B_9J5ynaUy3q63_XumjJyrA?e=z4hpcz
    [Backup] GoogleDrive:
    https://drive.google.com/file/d/1iVnP4gjw-iHXa0KerZQ1IfIO0i1jADsR/view?usp=sharing
    [Backup] Baidu Yun:
    https://pan.baidu.com/s/1H_wBnWwikKbaBY1pMPjoqQ password: hrqp
        
        Dataset statistics:
            - buildings: 1652 (train + query).
            - The dataset split is as follows: 
    | Split | #imgs | #buildings | #universities|
    | --------   | -----  | ----| ----|
    | Training | 50,218 | 701 | 33 |
    | Query_drone | 37,855 | 701 |  39 |
    | Query_satellite | 701 | 701 | 39|
    | Query_ground | 2,579 | 701 | 39|
    | Gallery_drone | 51,355 | 951 | 39|
    | Gallery_satellite |  951 | 951 | 39|
    | Gallery_ground | 2,921 | 793  | 39|
            - cameras: None.
    
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='university1652',
        targets='university1652',
        height=256,
        width=256,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    )
    """
    dataset_dir = 'places365_large'#'university1652' #

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        print(self.dataset_dir)
        if not os.path.isdir(self.dataset_dir):
            os.mkdir(self.dataset_dir)
            """gdown.download(
                self.dataset_url, self.dataset_dir + 'data.zip', quiet=False
            )
            os.system('unzip %s' % (self.dataset_dir + 'data.zip'))""" #Commented out assume downloaded places 365 dataset
        self.train_dir = osp.join(
            self.dataset_dir, 'train/' # img 1 to 4500
        )
        self.query_dir = osp.join(
            self.dataset_dir, 'test/query'#query_drone'# #each folder within this folder is a class on its own (4 digits) img 4901 to 5000
        )
        self.gallery_dir = osp.join(
            self.dataset_dir, 'test/gallery'#gallery_satellite'# #each folder within this folder is a class on its own (4 digits) img 4501 to 4900
        )

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        self.fake_camid = 0
        train = self.process_dir(self.train_dir, relabel=True, train=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(Places365, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False, train=False):
        IMG_EXTENSIONS = (
            '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff',
            '.webp'
        )
        if train:
            img_paths = glob.glob(osp.join(dir_path, '*/*'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*/*'))
        pid_container = set()
        for img_path in img_paths:
            if not img_path.lower().endswith(IMG_EXTENSIONS):
                continue
            pid = int(os.path.basename(os.path.dirname(img_path)))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        data = []
        # no camera for university
        for img_path in img_paths:
            if not img_path.lower().endswith(IMG_EXTENSIONS):
                continue
            pid = int(os.path.basename(os.path.dirname(img_path)))
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, self.fake_camid))
            self.fake_camid += 1
        return data
