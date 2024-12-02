import os
import re
import torch
import json
import argparse

import random
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LitDataset(Dataset):
    def __init__(self, img_path, vae_path, json_path, pose_map, transform=None, use_base=False, mode='train'):
        assert mode in ["train", "eval"]
        self.img_path = img_path
        self.transform = transform
        self.json_path = json_path
        if mode == "train":
            self.vae_path = vae_path
            self.use_base = use_base
        self.pose_map = pose_map
        self.data = self._load_data(mode=mode)
        self.mode = mode

    def _load_data(self, mode='train'):
        with open(self.json_path, "r") as f:
            json_dict = json.load(f)
        data = []
        if mode == 'train':
            for key in json_dict.keys():
                value = json_dict[key]
                obj, pose = key[:-20], key[-3:]
                src_condition, tgt_condition = int(value["src_light"]), int(value["tgt_light"])
                if self.use_base:
                    if f"{src_condition:03d}" != "013": continue
                src_lat_path = os.path.join(self.vae_path, f"{obj}_{src_condition:03d}_{pose}.pt")
                src_img_path = os.path.join(self.img_path, f"{obj}_{src_condition:03d}_{pose}.png")
                tgt_lat_path = os.path.join(self.vae_path, f"{obj}_{tgt_condition:03d}_{pose}.pt")
                tgt_img_path = os.path.join(self.img_path, f"{obj}_{tgt_condition:03d}_{pose}.png")
                data.append((src_img_path, tgt_img_path, src_lat_path, tgt_lat_path, src_condition, tgt_condition, self.pose_map[pose]))
        elif mode == 'eval':
            for key in json_dict.keys():
                value = json_dict[key]
                obj, pose = key[:-20], key[-3:]
                src_condition, tgt_condition = int(value["src_light"]), int(value["tgt_light"])
                src_img_path = os.path.join(self.img_path, f"{obj}_{src_condition:03d}_{pose}.png")
                tgt_img_path = os.path.join(self.img_path, f"{obj}_{tgt_condition:03d}_{pose}.png")

                #train_refs = glob(re.sub(f'{obj}', '*', re.sub(r'eval', 'train', src_img_path)))
                #ref_src_path = random.sample(train_refs, 1)[0]
                #ref_tgt_path = re.sub(f'{src_condition:03d}', f'{tgt_condition:03d}', ref_src_path)
                data.append((src_img_path, tgt_img_path,
                             #ref_src_path, ref_tgt_path,
                             src_condition, tgt_condition,
                             self.pose_map[pose], key.split("/")[-1]))
        else:
            raise ValueError('`mode` must be one of `train` or `eval`')

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            src_img_path, tgt_img_path, src_lat_path, tgt_lat_path, src_condition, tgt_condition, pose = self.data[idx]
            # 이미지를 로드하고 변환 적용
            src_img = Image.open(src_img_path)
            tgt_img = Image.open(tgt_img_path)

            src_lat = torch.load(src_lat_path, weights_only=True)
            tgt_lat = torch.load(tgt_lat_path, weights_only=True)

            if self.transform:
                src_img = self.transform(src_img)
                tgt_img = self.transform(tgt_img)

            item_dict = {
                "src_img": src_img,
                "tgt_img": tgt_img,
                "src_lat": src_lat,
                "tgt_lat": tgt_lat,
                "src_condition": torch.tensor(src_condition),
                "tgt_condition": torch.tensor(tgt_condition),
                "pose": torch.tensor(pose)
            }

        elif self.mode == 'eval':
            #src_img_path, tgt_img_path, src_ref_path, tgt_ref_path, src_condition, tgt_condition, pose = self.data[idx]
            src_img_path, tgt_img_path, src_condition, tgt_condition, pose, fname = self.data[idx]

            src_img = Image.open(src_img_path)
            tgt_img = Image.open(tgt_img_path)
            #src_ref = Image.open(src_ref_path)
            #tgt_ref = Image.open(tgt_ref_path)

            if self.transform:
                src_img = self.transform(src_img)
                tgt_img = self.transform(tgt_img)
                #src_ref = self.transform(src_ref)
                #tgt_ref = self.transform(tgt_ref)

            item_dict = {
                "src_img": src_img,
                "tgt_img": tgt_img,
                #"src_ref": src_ref,
                #"tgt_ref": tgt_ref,
                "src_condition": torch.tensor(src_condition),
                "tgt_condition": torch.tensor(tgt_condition),
                "pose": torch.tensor(pose),
                "name": fname
            }
        else:
            raise ValueError('`self.mode` must be one of `train` or `eval`')

        return item_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/workspace/dataset/train")
    parser.add_argument("--json_path", type=str, default="/workspace/dataset/preprocess/train.json", help="Load train.json or eval.json")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # 변환 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # NA3, NE7, CB5, CF8, NA7, CC7, CA2, NE1, NC3, CE2
    pose_map = {
        "NA3": 0, "NE7": 1, "CB5": 2, "CF8": 3, "NA7": 4, "CC7": 5, "CA2": 6, "NE1": 7, "NC3": 8, "CE2": 9
    }

    # 데이터셋 및 DataLoader 초기화
    dataset = LitDataset(data_path=args.data_path, json_path=args.json_path, pose_map=pose_map, transform=transform, use_base=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(args.json_path.split("/")[-1]=="train.json"))

    # 데이터 확인
    for batch in dataloader:
        print(batch["src_img"].shape)   # torch.Size([1, 3, 128, 128])
        print(batch["src_condition"])   # tensor([10])
        if "base_img" in batch.keys():
            print(batch["base_img"].shape)  # torch.Size([1, 3, 128, 128])
        print(batch["tgt_img"].shape)   # torch.Size([1, 3, 128, 128])
        print(batch["tgt_condition"])   # tensor([8])
        print(batch["pose"])            # tensor([1])
        break
