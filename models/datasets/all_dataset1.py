import io
import torch
import os
import glob
import pandas as pd
from PIL import Image
import numpy as np
# from datasets import load_dataset

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, mimic_data_dir, path_data_dir, txt_tokenizer, max_length=256):
        super(TrainDataset, self).__init__()
        self.mimic_data_dir = mimic_data_dir
        self.path_data_dir = path_data_dir

        self.all_images = []
        self.data_sources = []

        path_images = glob.glob(os.path.join(self.path_data_dir, '*.png'))[100000:200000]
        mimic_images = glob.glob(os.path.join(self.mimic_data_dir, '*.jpg'))[100000:200000]

        self.all_images.extend(path_images)
        self.data_sources.extend(['pathgen'] * len(path_images))

        self.all_images.extend(mimic_images)
        self.data_sources.extend(['mimic'] * len(mimic_images))
        #print(self.data[0])
        #print(self.path_images)
        self.txt_tokenizer = txt_tokenizer
        self.max_length = max_length
        
        #self.data = self._load_data()

    def _load_data(self):
        data = []
        for file in self.data_files:
            df = pd.read_parquet(file)
            data.extend(df[['image', 'reports']].values.tolist())
        return data

    def __len__(self):
        return len(self.all_images)

    def _vqgan_input_from(self, img: Image, target_image_size=512) -> torch.Tensor:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        tensor_img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float()
        )  # (Channels, Height, Width) format.

        # Add batch dimension.
        return tensor_img

    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        source = self.data_sources[idx]

        #print(image_path, source)

        if source == 'pathgen':
            report_path = image_path.replace('.png', '.txt')
        else:  # 'mimic'
            report_path = image_path.replace('.jpg', '.txt')

        with open(report_path, 'r', encoding='utf-8') as file:
            report = file.read()
        
        if source == 'mimic':
            report = 'The image is a radiograph of the chest, showing the thoracic cavity structures. ' + report
        # Convert image data
        image = Image.open(image_path).convert('RGB')

        width, height = image.size
        size = (width, height)

        pad_image = expand2square(image, (122, 116, 104))
        input_image = pad_image.resize((512, 512), Image.LANCZOS)
        image_tensor = self._vqgan_input_from(input_image)

        split_index = report.find('.')
        prompt = report[:split_index].strip()

        self.txt_tokenizer.padding_side = "right"
        prompt_out = self.txt_tokenizer(
            [prompt],
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        )
        prompt_out = prompt_out['input_ids'][0]
        prompt_idx = prompt_out.shape[-1]
        del prompt_out

        conversations = [report]
        txt_out = self.txt_tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        #pad_length = self.max_length - txt_ids.shape[-1]
        #mask_pad = torch.ones(size=(pad_length,))
        txt_ids = txt_out.input_ids[0]
        mask = txt_out.attention_mask[0]
        data = {'image': image_tensor, 'text': txt_ids, 'size': size, 'mask':mask, 'prompt_idx': prompt_idx}
        return data


class ValDataset(torch.utils.data.Dataset):    
    def __init__(self, mimic_data_dir, path_data_dir, txt_tokenizer, max_length=256):
        super(ValDataset, self).__init__()
        self.mimic_data_dir = mimic_data_dir
        self.path_data_dir = path_data_dir

        self.all_images = []
        self.data_sources = []
        path_images = glob.glob(os.path.join(self.path_data_dir, '*.png'))
        mimic_images = glob.glob(os.path.join(self.mimic_data_dir, '*.jpg'))

        self.all_images.extend(path_images)
        self.data_sources.extend(['pathgen'] * len(path_images))
        self.all_images.extend(mimic_images)
        self.data_sources.extend(['mimic'] * len(mimic_images))
        # print(self.data[0])
        self.txt_tokenizer = txt_tokenizer
        self.max_length = max_length

        # self.data = self._load_data()

    def _load_data(self):
        data = []
        for file in self.data_files:
            df = pd.read_parquet(file)
            data.extend(df[['image', 'reports']].values.tolist())
        return data

    def __len__(self):
        return len(self.all_images)

    def _vqgan_input_from(self, img: Image, target_image_size=512) -> torch.Tensor:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        tensor_img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float()
        )  # (Channels, Height, Width) format.

        # Add batch dimension.
        return tensor_img

    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        source = self.data_sources[idx]

        if source == 'pathgen':
            report_path = image_path.replace('.png', '.txt')
        else:  # 'mimic'
            report_path = image_path.replace('.jpg', '.txt')

        with open(report_path, 'r', encoding='utf-8') as file:
            report = file.read()
        
        if source == 'mimic': 
            report = 'The image is a radiograph of the chest, showing the thoracic cavity structures, ' + report

        # Convert image data
        image = Image.open(image_path).convert('RGB')

        width, height = image.size
        size = (width, height)

        pad_image = expand2square(image, (122, 116, 104))
        input_image = pad_image.resize((512, 512), Image.LANCZOS)
        image_tensor = self._vqgan_input_from(input_image)

        split_index = report.find('.')
        prompt = report[:split_index].strip()

        self.txt_tokenizer.padding_side = "right"
        prompt_out = self.txt_tokenizer(
            [prompt],
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        )
        prompt_out = prompt_out['input_ids'][0]
        prompt_idx = prompt_out.shape[-1]
        del prompt_out

        conversations = [report]
        txt_out = self.txt_tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # pad_length = self.max_length - txt_ids.shape[-1]
        # mask_pad = torch.ones(size=(pad_length,))
        txt_ids = txt_out.input_ids[0]
        mask = txt_out.attention_mask[0]
        data = {'image': image_tensor, 'text': txt_ids, 'size': size, 'mask': mask, 'prompt_idx': prompt_idx}
        return data

# from evaluation.chameleon import ImageTokenizer
# from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
#
# tokenizer = AutoTokenizer.from_pretrained(r'D:\CKPTS\liquid-7b',padding_side='left')
# print(tokenizer.model_max_length)
# ds = MyDataset(r'D:\CKPTS\liquid-7b\cc12m-train-0000', tokenizer)
# dl = DataLoader(ds, batch_size=2)
# vqgan_cfg_path = r"D:\CKPTS\vqgan\vqgan.yaml"
# vqgan_ckpt_path = r"D:\CKPTS\vqgan\vqgan.ckpt"
# image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0",)
#
# for data in dl:
#     img = data['image'].to('cuda')
#     print(img.shape)
#     _, _, [_, _, vq_code] = image_tokenizer._vq_model.encode(img)
#     vq_code = vq_code.view(img.shape[0], -1)
#     vq_code = vq_code + len(tokenizer)
#
#     print(vq_code.shape)

