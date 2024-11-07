# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
from numpy.random import randint
import torchvision
from dataloaders.SSV2Transform import ToTorchFormatTensor, GroupScale, GroupRandomCrop
from dataloaders.SSV2Transform import *


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, cfg, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):
        self.cfg = cfg
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')
            # image_array = np.array(img)
            # print('image_array.shape',image_array.shape)
            return img

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]
        self.label_array = [item.label for item in self.video_list]
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        prompt_attribute = torch.Tensor()
        buffer, label = self.get(record, segment_indices)

        sample = {
            "image": buffer,
            "label": torch.tensor(label),
            "prompt_attribute": prompt_attribute.detach()
        }
        return sample

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.append(seg_imgs)
                if p < record.num_frames:
                    p += 1
        process_data = self.transform(images)
        buffer = process_data.view((-1, 3) + process_data.size()[-2:]).transpose(0, 1)
        return buffer, record.label

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
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

    def __len__(self):
        return len(self.video_list)


def get_dataloader(cfg, batach_size, num_worker, num_segments):
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(cfg, '/home/yqx/yqx_softlink/data/SomethingSomethingV2/something-something-v2-frames',
                   '/home/yqx/yqx_softlink/data/SomethingSomethingV2/train_videofolder.txt',
                   num_segments=num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl='{:06d}.jpg',
                   transform=torchvision.transforms.Compose([
                       GroupMultiScaleCrop(224, [1, .875, .75]),
                       GroupRandomHorizontalFlip(is_flow=False),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                   ),
                   dense_sample=False), batch_size=batach_size, shuffle=True,
        num_workers=num_worker, pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(cfg, '/home/yqx/yqx_softlink/data/SomethingSomethingV2/something-something-v2-frames',
                   '/home/yqx/yqx_softlink/data/SomethingSomethingV2/val_videofolder.txt',
                   num_segments=num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl='{:06d}.jpg',
                   transform=torchvision.transforms.Compose([
                       GroupMultiScaleCrop(224, [1, .875, .75]),
                       GroupRandomHorizontalFlip(is_flow=False),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                   ),
                   dense_sample=False), batch_size=batach_size, shuffle=True,
        num_workers=num_worker, pin_memory=True,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        TSNDataSet(cfg, '/home/yqx/yqx_softlink/data/SomethingSomethingV2/something-something-v2-frames',
                   '/home/yqx/yqx_softlink/data/SomethingSomethingV2/test_videofolder.txt',
                   num_segments=num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl='{:06d}.jpg',
                   transform=torchvision.transforms.Compose([
                       GroupMultiScaleCrop(224, [1, .875, .75]),
                       GroupRandomHorizontalFlip(is_flow=False),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                   ),
                   dense_sample=False), batch_size=batach_size, shuffle=True,
        num_workers=num_worker, pin_memory=True,
        drop_last=True)
    return train_loader, val_loader, test_loader
