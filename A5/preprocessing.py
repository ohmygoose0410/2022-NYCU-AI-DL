import numpy as np
from pathlib import Path
import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import random

def obtain_path(
    img_dir: str,
    mask_dir: str,
    target_path: str
) -> None:
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
    def __init__(self,
    dataset: dict,
    img_transforms,
    mask_transforms,
    dataset_type: str = 'train'
) -> None:
        dataset_types = ['train', 'test', 'valid']
        if dataset_type not in dataset_types:
            raise ValueError("Invalid dataset type. Expected one of: %s" % dataset_types)
        
        self.imgPaths = dataset[f'{dataset_type}_img']
        self.maskPaths = dataset[f'{dataset_type}_mask']
        
        self.img_transforms=img_transforms
        self.mask_transforms=mask_transforms
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        image = Image.open(self.imgPaths[idx])
        mask = Image.open(self.maskPaths[idx])
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transforms is not None:
            image = self.img_transforms(image)
            
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        return image, mask

def divide_dataset(
    dataset_json: str,
    lengths: list = [0.7,0.1,0.2],
    show: bool = False
) -> dict:

    # lengths -> [train_set_size, valid_set_size, test_set_size]
    with open(dataset_json) as jsonfile:
        data_load = json.load(jsonfile)
        imPaths = np.array(data_load['imgPaths'])
        maskPaths = np.array(data_load['maskPaths'])
    train_set_size = int(len(imPaths) * lengths[0])
    valid_set_size = int(len(imPaths) * lengths[1])
    test_set_size = len(imPaths) - train_set_size - valid_set_size

    dataset  = torch.utils.data.random_split(range(len(imPaths)),
                                            [train_set_size,valid_set_size,test_set_size],
                                            torch.Generator().manual_seed(42))

    data = {}
    data['train_img'] = imPaths[dataset[0]]
    data['train_mask'] = maskPaths[dataset[0]]
    data['valid_img'] = imPaths[dataset[1]]
    data['valid_mask'] = maskPaths[dataset[1]]
    data['test_img'] = imPaths[dataset[2]]
    data['test_mask'] = maskPaths[dataset[2]]

    if show:
        print("train_img_size: ", len(data['train_img']))
        print("train_mask_size: ", len(data['train_mask']))
        print("valid_img_size: ", len(data['valid_img']))
        print("valid_mask_size: ", len(data['valid_mask']))
        print("test_img_size: ", len(data['test_img']))
        print("test_mask_size: ", len(data['test_mask']))

    return data    

def GraphVisualization(dataset: MyDataset, model=None, col=5, target_dir: str="./"):
    rows = ['Images', 'Ground\nTruth\nMasks', 'Ground\nTruth\nFusions',
            'Prediction\nMasks', 'Prediction\nFusions', 'Prediction V.S.\nGround Truth']
    if model is None:
        fig, axes = plt.subplots(nrows=3, ncols=col, figsize=(10,10))

        for i in range(3):
            axes[i][0].annotate(rows[i], xy=(0, 0.5), xytext=(-30,60),
                                xycoords='axes points', textcoords='offset points',
                                size='large', ha='center', va='center')

        for i in range(5):
            filename = Path(dataset.imgPaths[i]).stem
            data = dataset.__getitem__(i)
            mask = np.array(data[1]).squeeze()
            invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                 std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                                 std = [ 1., 1., 1. ])])
            img = invTrans(data[0])
            img = np.array(img).transpose(1,2,0)

            axes[0][i].set_title(filename, {'fontsize': 8})
            axes[0][i].get_xaxis().set_visible(False)
            axes[0][i].get_yaxis().set_visible(False)
            axes[0][i].imshow(img)
            axes[1][i].get_xaxis().set_visible(False)
            axes[1][i].get_yaxis().set_visible(False)
            axes[1][i].imshow(mask, cmap='magma')
            axes[2][i].get_xaxis().set_visible(False)
            axes[2][i].get_yaxis().set_visible(False)
            axes[2][i].imshow(img)
            axes[2][i].imshow(mask, cmap='twilight', alpha=0.6)

        fig.tight_layout(h_pad=-25)
        plt.savefig(os.path.join(target_dir, 'sample.png'), dpi=500)
        plt.show()
    else:
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-orig_img_dir',type=str)
    parser.add_argument('-orig_msk_dir',type=str)
    parser.add_argument('-save_json',type=str)
    parser.add_argument('-save_samples',type=str)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    print(args.save_samples)
    if args.save_samples != '':
        os.makedirs(args.save_samples, exist_ok=True)
        for sample in Path(args.save_samples).glob('*.*'):
            os.remove(str(sample))
    
    if not os.path.exists(args.save_json):
        obtain_path(img_dir=args.orig_img_dir, mask_dir=args.orig_msk_dir, target_path=str(args.save_json))
    
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    img_tsfm=transforms.Compose([transforms.ToTensor(), 
                                # transforms.Normalize(channel_means, channel_stds),
                                transforms.RandomCrop((832, 832)), 
                                transforms.RandomRotation(90), 
                                transforms.RandomHorizontalFlip()])
    mask_tsfm=transforms.Compose([transforms.ToTensor(), 
                                # transforms.Normalize(channel_means, channel_stds),
                                transforms.RandomCrop((832, 832)), 
                                transforms.RandomRotation(90), 
                                transforms.RandomHorizontalFlip()])
    # dataset = MyDataset('./dataset.json', img_tsfm, mask_tsfm)
    # dataloader = torch.utils.data.DataLoader( dataset,
    #                                           batch_size=1,
    #                                           shuffle=False,
    #                                           num_workers=0,
    #                                           pin_memory=True)

    json_path = args.save_json
    dataset = divide_dataset(json_path, [0.7,0.1,0.2], True)

    train_set = MyDataset(dataset, img_tsfm, mask_tsfm, 'train')
    valid_set = MyDataset(dataset, None, None, 'valid')
    test_set = MyDataset(dataset, None, None, 'test')

    _iterator_ = iter(train_set)
    tsmf_flag = True
    for i in range(3):
        data = next(_iterator_)
        if tsmf_flag:
            img = np.moveaxis(data[0].numpy()*255, 0, -1).astype(np.uint8)
            mask = data[1].numpy().squeeze()
        else:
            img = data[0]
            mask = data[1]
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original Image', fontdict={'fontsize': 8})
        ax.imshow(img)
        ax = fig.add_subplot(132)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original Mask', fontdict={'fontsize': 8})
        ax.imshow(mask)
        ax = fig.add_subplot(133)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original Image +\n Original Mask', fontdict={'fontsize': 8})
        ax.imshow(img)
        ax.imshow(mask, alpha=0.4)
        fig.savefig(os.path.join(args.save_samples, f'sample_{i}.jpg'), dpi=500)