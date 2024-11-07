from mmaction.utils import register_all_modules
import os.path as osp
from mmengine.fileio import list_from_file
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS
from mmengine.runner import Runner
import mmcv
import decord
import numpy as np
from mmcv.transforms import TRANSFORMS, BaseTransform, to_tensor
from mmaction.structures import ActionDataSample
import torch

register_all_modules(init_default_scope=True)


@TRANSFORMS.register_module()
class VideoInit(BaseTransform):
    def transform(self, results):
        container = decord.VideoReader(results['filename'])
        results['total_frames'] = len(container)
        results['video_reader'] = container
        return results


@TRANSFORMS.register_module()
class VideoSample(BaseTransform):
    def __init__(self, clip_len, num_clips, test_mode=False):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def transform(self, results):
        total_frames = results['total_frames']
        interval = total_frames // self.clip_len

        if self.test_mode:
            np.random.seed(42)

        inds_of_all_clips = []
        for i in range(self.num_clips):
            bids = np.arange(self.clip_len) * interval
            offset = np.random.randint(interval, size=bids.shape)
            inds = bids + offset
            inds_of_all_clips.append(inds)

        results['frame_inds'] = np.concatenate(inds_of_all_clips)
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips
        return results


@TRANSFORMS.register_module()
class VideoDecode(BaseTransform):
    def transform(self, results):
        frame_inds = results['frame_inds']
        container = results['video_reader']

        imgs = container.get_batch(frame_inds).asnumpy()
        imgs = list(imgs)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoResize(BaseTransform):
    def __init__(self, r_size):
        self.r_size = (np.inf, r_size)

    def transform(self, results):
        img_h, img_w = results['img_shape']
        new_w, new_h = mmcv.rescale_size((img_w, img_h), self.r_size)

        imgs = [mmcv.imresize(img, (new_w, new_h))
                for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoCrop(BaseTransform):
    def __init__(self, c_size):
        self.c_size = c_size

    def transform(self, results):
        img_h, img_w = results['img_shape']
        center_x, center_y = img_w // 2, img_h // 2
        x1, x2 = center_x - self.c_size // 2, center_x + self.c_size // 2
        y1, y2 = center_y - self.c_size // 2, center_y + self.c_size // 2
        imgs = [img[y1:y2, x1:x2] for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoFormat(BaseTransform):
    def transform(self, results):
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        imgs = results['imgs']

        # [num_clips*clip_len, H, W, C]
        imgs = np.array(imgs)
        # [num_clips, clip_len, H, W, C]
        imgs = imgs.reshape((num_clips, clip_len) + imgs.shape[1:])
        # [num_clips, C, clip_len, H, W]
        imgs = imgs.transpose(0, 4, 1, 2, 3)

        results['imgs'] = imgs
        return results


@TRANSFORMS.register_module()
class VideoPack(BaseTransform):
    def __init__(self, meta_keys=('img_shape', 'num_clips', 'clip_len')):
        self.meta_keys = meta_keys

    def transform(self, results):
        packed_results = dict()
        inputs = to_tensor(results['imgs'])
        data_sample = ActionDataSample().set_gt_labels(results['label'])
        metainfo = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(metainfo)
        packed_results['inputs'] = inputs
        packed_results['data_samples'] = data_sample
        return packed_results


@DATASETS.register_module()
class DatasetZelda(BaseDataset):
    def __init__(self, cfg, ann_file, pipeline, data_root, data_prefix=dict(video=''),
                 test_mode=False, modality='RGB', **kwargs):
        self.modality = modality
        self.cfg = cfg
        self.name = cfg.DATA.NAME

        super(DatasetZelda, self).__init__(ann_file=ann_file, pipeline=pipeline, data_root=data_root,
                                           data_prefix=data_prefix, test_mode=test_mode,
                                           **kwargs)

    def load_data_list(self):
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            filename, label = line_split
            label = int(label)
            filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label))
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        return data_info

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num


def K400_tiny_dataloader(cfg):
    batch_size = cfg.DATA.BATCH_SIZE
    train_pipeline_cfg = [
        dict(type='VideoInit'),
        dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
        dict(type='VideoDecode'),
        dict(type='VideoResize', r_size=256),
        dict(type='VideoCrop', c_size=cfg.DATA.CROPSIZE),
        dict(type='VideoFormat'),
        dict(type='VideoPack')
    ]

    val_pipeline_cfg = [
        dict(type='VideoInit'),
        dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=True),
        dict(type='VideoDecode'),
        dict(type='VideoResize', r_size=256),
        dict(type='VideoCrop', c_size=cfg.DATA.CROPSIZE),
        dict(type='VideoFormat'),
        dict(type='VideoPack')
    ]

    train_dataset_cfg = dict(
        type='DatasetZelda',
        cfg=cfg,
        ann_file='train_annotations.txt',
        pipeline=train_pipeline_cfg,
        data_root='/home/yqx/yqx_softlink/data/tiny-Kinetics-400',
        data_prefix=dict(video='train'))

    val_dataset_cfg = dict(
        type='DatasetZelda',
        cfg=cfg,
        ann_file='val_annotations.txt',
        pipeline=val_pipeline_cfg,
        data_root='/home/yqx/yqx_softlink/data/tiny-Kinetics-400',
        data_prefix=dict(video='val'))
    train_dataloader_cfg = dict(
        batch_size=batch_size,
        num_workers=0,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=train_dataset_cfg)
    val_dataloader_cfg = dict(
        batch_size=batch_size,
        num_workers=0,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=val_dataset_cfg)
    train_loader = Runner.build_dataloader(dataloader=train_dataloader_cfg)
    val_loader = Runner.build_dataloader(dataloader=val_dataloader_cfg)
    test_loader = Runner.build_dataloader(dataloader=val_dataloader_cfg)
    return train_loader, val_loader, test_loader
