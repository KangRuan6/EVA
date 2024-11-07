# coding:utf8
import os
from sklearn.model_selection import train_test_split
import torch
import cv2
import pickle
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
import random
from dataloaders.videomae_transforms import *
from torchvision import transforms
import src.utils.logging as logging

logger = logging.get_logger("visual_prompt")
from torch.utils.data import DataLoader


class HMDB51Dataset(Dataset):
    def __init__(self, cfg, split='train', clip_len=16, fpath_label=None):
        self.clip_len = clip_len
        self.split = split
        self.cfg = cfg
        self.crop_size = cfg.DATA.CROPSIZE
        self.name = cfg.DATA.NAME
        self.transform = transforms.Compose([
            GroupMultiScaleCrop(224, [1, .875, .75, .66]),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        ######################################################################### 补充的部分
        f = open(fpath_label)
        l = f.readlines()
        f.close()
        # print l
        fpaths = list()
        labels = list()
        for item in l:
            path = item.strip().split()[0].split('.')[0]  # Depending on your fpath_label file
            label = item.strip().split()[1]  # default for single label, while [1:] for single label
            label = int(label)
            fpaths.append(path)
            labels.append(label)

        self.root_folder = '/home/yqx/yqx_softlink/data/HMDB51/3fold/hmdb_frames/'
        self.fpaths = fpaths
        self.labels = labels

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        # 加载buffer
        try:
            labels = np.array(self.labels[index])
            frames_dir = self.root_folder + self.fpaths[index]
            prompt_dir = '/home/yqx/yqx_softlink/data/HMDB51/3fold/HMDB51Attr/' + self.fpaths[index]
            mask_feature, caption_feature, label_feature = self.load_attribute_pkl(prompt_dir)
            prompt_attribute = torch.cat((mask_feature, caption_feature, label_feature), dim=0)
            prompt_attribute = prompt_attribute.reshape(2, 768)
            buffer = self.load_frames(frames_dir)
            if buffer.shape[0] < self.clip_len:
                print(self.fnames[index])
            buffer = self.crop(buffer, self.clip_len, self.crop_size)

            # 加载attrribute

            if self.cfg.TRICKS.CROP:
                sampled_list = [Image.fromarray(np.uint8(buffer[vid, :, :, :])).convert('RGB') for vid in
                                range(buffer.shape[0])]
                process_data, _ = self.transform((sampled_list, None))
                buffer = process_data.view((-1, 3) + process_data.size()[-2:]).transpose(0, 1)
            else:
                if self.split == 'test':
                    buffer = self.randomflip(buffer)
                buffer = self.normalize(buffer)
                buffer = self.to_tensor(buffer)
            if not isinstance(buffer, torch.Tensor):
                if isinstance(buffer, np.ndarray):
                    buffer = torch.from_numpy(buffer)
                else:
                    buffer = torch.Tensor(buffer)
            # print(buffer.shape)
            sample = {
                "image": buffer,
                "label": torch.from_numpy(labels),
                "prompt_attribute": prompt_attribute.detach()
            }
            return sample
        except:
            # 判断一下边界，也可能-1
            return self.__getitem__(index + 1)

    def load_attribute_pkl(self, pkl_dir):
        # 找到目标文件夹
        pkl_files = os.listdir(pkl_dir)
        # 随机选取一个读入
        random_file = random.choice(pkl_files)
        # 拼接随机后地址
        random_key = os.path.join(pkl_dir, random_file)
        with open(random_key, 'rb') as f:
            data = pickle.load(f)
        # data = torch.load(random_key,map_location=torch.device('cpu'))
        # 地址中不包含pkl
        random_key = random_key.replace(".pkl", "")
        random_key = random_key.replace(self.cfg.PROMPT.ATTRIBUTE_DIR, "")
        return data['mask_feature'][random_key], data['caption_feature'][random_key], data['label_feature'][random_key]

    def attribute_process(self, mask_feature, caption_feature, label_feature):
        prompt_attribute = torch.Tensor()
        if self.cfg.PROMPT.ATTRIBUTE_MODE == 'concat-fusion-equal':
            prompt_attribute = torch.cat((mask_feature, caption_feature, label_feature), dim=0)
            prompt_attribute = prompt_attribute.reshape(2, 768)
        return prompt_attribute



    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
        if "train" not in self.split:
            raise ValueError("only getting training class distribution, " + "got split {} instead".format(self._split))
        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num
        id2counts = Counter(self.label_array)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self.label_array])
        mu = 0
        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)  ##取得某一个视频对应的所有图片
        frame_flag = np.array(cv2.imread(frames[0])).astype(np.float64)
        buffer = np.empty((frame_count, frame_flag.shape[0], frame_flag.shape[1], 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        if (buffer.shape[0] <= clip_len):
            print("该视频没有足够的帧数可供选择")
            time_index = 0
        else:
            time_index = np.random.randint(buffer.shape[0] - clip_len)
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


def get_data_loaders(cfg, logger, split_mode, dataset_name, clip_len):
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    logger.info("Loading {} data".format(split_mode))
    fpath_label = ('/home/yqx/yqx_softlink/data/HMDB51/3fold/labels/' + split_mode + '1.txt')

    dataset = HMDB51Dataset(cfg=cfg, split=split_mode, clip_len=clip_len, fpath_label=fpath_label)
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    data_loader = DataLoader(dataset,
                             shuffle=(False if sampler else True),
                             sampler=sampler,
                             num_workers=8,
                             pin_memory=cfg.DATA.PIN_MEMORY,
                             drop_last=drop_last,
                             batch_size=cfg.DATA.BATCH_SIZE)

    return data_loader


def get_hmdb51_loaders(cfg, logger):
    dataset_name = cfg.DATA.NAME
    train_loader = get_data_loaders(cfg, logger, split_mode='train',
                                    dataset_name=dataset_name, clip_len=16)
    val_loader = get_data_loaders(cfg, logger, split_mode='train',
                                  dataset_name=dataset_name, clip_len=16)
    test_loader = get_data_loaders(cfg, logger, split_mode='test',
                                   dataset_name=dataset_name, clip_len=16)
    return train_loader, test_loader, val_loader
