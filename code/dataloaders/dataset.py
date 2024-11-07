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


class VideoDataset(Dataset):

    def __init__(self, cfg, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        self.root_dir = cfg.DATA.ROOT_DIR
        self.output_dir = cfg.DATA.OUTPUT_DIR
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.cfg = cfg
        self.resize_height = 224
        self.resize_width = 325
        self.crop_size = cfg.DATA.CROPSIZE
        self.name = cfg.DATA.NAME
        self.device = torch.device("cuda:{}".format(cfg.GPUS_id))
        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        if dataset == "ucf101":
            if not os.path.exists('dataloaders/ucf_labels.txt'):
                with open('ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id + 1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('hmdb_labels.txt'):
                with open('hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id + 1) + ' ' + label + '\n')
        self.dataset = dataset

        # GroupMultiScaleCrop数据增强
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        # normalize = GroupNormalize(self.input_mean, self.input_std)
        # self.train_augmentation = GroupMultiScaleCrop(224, [1, .875, .75, .66])
        self.transform = transforms.Compose([
            GroupMultiScaleCrop(224, [1, .875, .75, .66]),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(self.input_mean, self.input_std)
        ])

    def __len__(self):
        if self.cfg.TRICKS.FLIP:
            return 2 * len(self.fnames)
        else:
            return len(self.fnames)

    def __getitem__(self, index):
        flag = 0
        if self.cfg.TRICKS.FLIP:
            if self.split == 'train':
                flag = index % 2
                index = index // 2
            else:
                flag = 0
                index = index // 2
        # 加载attrribute
        prompt_attribute = torch.Tensor()
        if self.cfg.PROMPT.ATTRIBUTE:
            attribute_address = self.fnames[index].replace(self.cfg.DATA.OUTPUT_DIR, self.cfg.PROMPT.ATTRIBUTE_DIR)
            mask_feature, caption_feature, label_feature = self.load_attribute_pkl(attribute_address)
            prompt_attribute = self.attribute_process(mask_feature, caption_feature, label_feature)

        # 加载buffer
        buffer = self.load_frames(self.fnames[index])
        if buffer.shape[0] < self.clip_len:
            print(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])
        if self.cfg.TRICKS.CROP:
            # if self.split == 'test':
            #     buffer = self.randomflip(buffer)
            #     buffer = self.normalize(buffer)
            #     buffer = self.to_tensor(buffer)
            # if self.split == 'val':
            #     buffer = self.randomflip(buffer)
            #     buffer = self.normalize(buffer)
            #     buffer = self.to_tensor(buffer)
            # if self.split == 'train':
            sampled_list = [Image.fromarray(np.uint8(buffer[vid, :, :, :])).convert('RGB') for vid in
                            range(buffer.shape[0])]
            # print(sampled_list.shape)
            process_data, _ = self.transform((sampled_list, None))
            buffer = process_data.view((-1, 3) + process_data.size()[-2:]).transpose(0, 1)
            # T*C,H,W -> T,C,H,W -> C,T,H,W
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
        if self.cfg.TRICKS.FLIP:
            if flag:
                buffer = torch.flip(buffer, dims=[1])

        # # print(buffer.shape)
        # if self.cfg.MODEL.TYPE == 'CNN':
        #     if self.cfg.PROMPT.ATTRIBUTE:
        #         # print("开始处理")
        #         attr = torch.cat((mask_feature, mask_feature,
        #                           caption_feature, caption_feature, caption_feature,
        #                           label_feature, label_feature), dim=0).view(16, 224)
        #         for i in range(3):
        #             buffer[i, :, :, 0] = attr
        # print("buffer.shape", buffer.shape)
        sample = {
            "image": buffer,
            "label": torch.from_numpy(labels),
            "prompt_attribute": prompt_attribute.detach()
        }
        return sample

    def attribute_process(self, mask_feature, caption_feature, label_feature):
        if self.cfg.PROMPT.ATTRIBUTE_MODE == 'concat-fusion-equal':
            if self.cfg.MODEL.TYPE == 'videoMAE-h':
                prompt_attribute = torch.cat((mask_feature,
                                              caption_feature, caption_feature, caption_feature,
                                              label_feature), dim=0)
                prompt_attribute = prompt_attribute.reshape(2, 1280)
            else:
                prompt_attribute = torch.cat((mask_feature, caption_feature, label_feature), dim=0)
                prompt_attribute = prompt_attribute.reshape(2, 768)
        elif self.cfg.PROMPT.ATTRIBUTE_MODE == 'all-mask':
            # print("all-mask")
            prompt_attribute = torch.cat((mask_feature, mask_feature, mask_feature), dim=0)
            prompt_attribute = prompt_attribute.reshape(2, 768)
        elif self.cfg.PROMPT.ATTRIBUTE_MODE == 'all-cap':
            if self.cfg.MODEL.TYPE == 'videoMAE-h':
                prompt_attribute = torch.cat((caption_feature,
                                              caption_feature, caption_feature, caption_feature,
                                              caption_feature), dim=0)
                prompt_attribute = prompt_attribute.reshape(2, 1280)
            else:
                prompt_attribute = torch.cat((caption_feature, caption_feature, caption_feature), dim=0)
                prompt_attribute = prompt_attribute.reshape(2, 768)
        elif self.cfg.PROMPT.ATTRIBUTE_MODE == 'all-label':
            prompt_attribute = torch.cat((label_feature, label_feature, label_feature), dim=0)
            prompt_attribute = prompt_attribute.reshape(2, 768)
        elif self.cfg.PROMPT.ATTRIBUTE_MODE == '1-4-1-cap':
            prompt_attribute = torch.cat((label_feature, caption_feature, caption_feature,
                                          caption_feature, caption_feature, mask_feature), dim=0)
            prompt_attribute = prompt_attribute.reshape(4, 768)
        elif self.cfg.PROMPT.ATTRIBUTE_MODE == '1-4-1-lab':
            prompt_attribute = torch.cat((caption_feature, label_feature, label_feature,
                                          label_feature, label_feature, mask_feature), dim=0)
            prompt_attribute = prompt_attribute.reshape(4, 768)
        else:
            prompt_attribute = torch.cat((mask_feature, caption_feature, label_feature), dim=0)
            prompt_attribute = prompt_attribute.reshape(2, 768)

        return prompt_attribute

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
        random_key = random_key.replace(self.cfg.PROMPT.ATTRIBUTE_DIR + '/', "")
        return data['mask_feature'][random_key], data['caption_feature'][random_key], data['label_feature'][random_key]

    # 获取数据集类别
    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
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

    ## 检查是否已经做过预处理
    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                          sorted(
                                              os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[
                                              0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 224 or np.shape(image)[1] != 325:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    ## 数据集预处理
    def preprocess(self):
        ## 创建输出结果子文件夹
        print("output_dir is " + str(self.output_dir))
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        ## 划分train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)  ##file表示每一个视频文件夹
            video_files = [name for name in os.listdir(file_path)]  ##每一类视频中的视频文件
            ## train/val/test划分比例为0.64:0.16:0.2
            train_and_valid, test = train_test_split(video_files, test_size=0.3, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.1, random_state=42)

            ## 得到各个存储图片的子文件夹
            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            ## 处理视频，将其存储到对应的文件夹
            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    ## 读取视频，抽取帧
    def process_video(self, video, action_name, save_dir):
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ## 确保至少有16帧，默认抽帧频率为4，最低为1，如果视频总长度少于16帧，会报错
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

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
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    ## 从一个视频中随机抽帧，裁剪
    def crop(self, buffer, clip_len, crop_size):
        ## 随机选择时间切片参数
        if (buffer.shape[0] <= clip_len):
            print("该视频没有足够的帧数可供选择")
            time_index = 0
        else:
            time_index = np.random.randint(buffer.shape[0] - clip_len)

        ## 随机选择空间裁剪参数
        # height_index = np.random.randint(buffer.shape[1] - crop_size)
        height_index = 0
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer
