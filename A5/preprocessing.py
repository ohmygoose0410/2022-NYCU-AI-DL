import numpy as np
from pathlib import Path
import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse

def obtain_path(img_dir: str, mask_dir: str, target_path: str) -> None:
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    imgPaths = []
    maskPaths = []

    slide_id = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']

    for id in slide_id:
        for imgpath in tqdm(Path.joinpath(img_dir,id).glob('*.jpg'), desc=f'slide id:{id}'):
            imgPaths.append(str(imgpath))
            maskpath = Path.joinpath(mask_dir,id,f"{imgpath.stem}.png")
            if not maskpath.exists():
                print(f'The mask of the filename {imgpath.stem} was missing.')
                continue
            maskPaths.append(str(maskpath))

    data = {}
    data['imgPaths'] = imgPaths
    data['maskPaths'] = maskPaths
    with open(target_path, 'w', newline='') as jsonfile:
        json.dump(data, jsonfile)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_json_path: str, img_transforms: transforms.Compose, mask_transforms: transforms.Compose):
        with open(dataset_json_path, 'r') as jsonfile:
            data = json.load(jsonfile)
            self.imgPaths = data['imgPaths']
            self.maskPaths = data['maskPaths']
        
        self.img_transforms=img_transforms
        self.mask_transforms=mask_transforms
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        image = mpimg.imread(self.imgPaths[idx])
        mask = mpimg.imread(self.maskPaths[idx])
        if self.img_transforms is not None:
            image = self.img_transforms(image)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        return image, mask

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-orig_img_dir',type=str)
    parser.add_argument('-orig_msk_dir',type=str)
    parser.add_argument('-save_json',type=str)
    parser.add_argument('-save_samples',type=str)
    args = parser.parse_args()

    if args.save_samples != '':
        os.makedirs(args.save_samples, exist_ok=True)
        for sample in Path(args.save_samples).glob('*.*'):
            os.remove(str(sample))
    
    if not os.path.exists(args.save_json):
        obtain_path(img_dir=args.orig_img_dir, mask_dir=args.orig_msk_dir, target_path=str(args.save_json))

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    img_tsfm = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])
    mask_tsfm=transforms.Compose([transforms.ToTensor()])

    dataset = MyDataset(args.save_json, None, None)
    # dataset = MyDataset('./dataset.json', img_tsfm, mask_tsfm)
    # dataloader = torch.utils.data.DataLoader( dataset,
    #                                           batch_size=1,
    #                                           shuffle=False,
    #                                           num_workers=0,
    #                                           pin_memory=True)

    _iterator_ = iter(dataset)
    for i in range(5):
        data = next(_iterator_)
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original Image', fontdict={'fontsize': 8})
        ax.imshow(data[0])
        ax = fig.add_subplot(132)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original Mask', fontdict={'fontsize': 8})
        ax.imshow(data[1])
        ax = fig.add_subplot(133)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original Image +\n Original Mask', fontdict={'fontsize': 8})
        ax.imshow(data[0])
        ax.imshow(data[1], alpha=0.4)
        fig.savefig(os.path.join(args.save_samples, f'sample_{i}.jpg'), dpi=500)