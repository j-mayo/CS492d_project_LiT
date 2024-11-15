import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import argparse

class LitDataset(Dataset):
    def __init__(self, data_path, train_json_path, pose_map, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.train_json_path = train_json_path
        self.pose_map = pose_map
        self.data = self._load_data()

    def _load_data(self):
        with open(self.train_json_path, "r") as f:
            train_json = json.load(f)
        data = []
        for key in train_json.keys():
            value = train_json[key]
            obj, pose = key[:-20], key[-3:]
            src_condition, tgt_condition = int(value["src_light"]), int(value["tgt_light"])
            src_img_path = os.path.join(self.data_path, f"{obj}_{src_condition:03d}_{pose}.png")
            tgt_img_path = os.path.join(self.data_path, f"{obj}_{tgt_condition:03d}_{pose}.png")
            data.append((src_img_path, src_condition, tgt_img_path, tgt_condition, self.pose_map[pose]))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_img_path, src_condition, tgt_img_path, tgt_condition, pose = self.data[idx]
        
        # 이미지를 로드하고 변환 적용
        src_img = Image.open(src_img_path)
        tgt_img = Image.open(tgt_img_path)

        if self.transform:
            src_img = self.transform(src_img)
            tgt_img = self.transform(tgt_img)
        
        return {
            "src_img": src_img,
            "src_condition": torch.tensor(src_condition),
            "tgt_img": tgt_img,
            "tgt_condition": torch.tensor(tgt_condition),
            "pose": torch.tensor(pose)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/workspace/dataset/train")
    parser.add_argument("--train_json_path", type=str, default="/workspace/dataset/preprocess/train.json")
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
    dataset = LitDataset(data_path=args.data_path, train_json_path=args.train_json_path, pose_map=pose_map, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 데이터 확인
    for batch in dataloader:
        print(batch["src_img"].shape)   # torch.Size([1, 3, 128, 128])
        print(batch["src_condition"])   # tensor([10])
        print(batch["tgt_img"].shape)   # torch.Size([1, 3, 128, 128])
        print(batch["tgt_condition"])   # tensor([8])
        print(batch["pose"])            # tensor([1])
        break