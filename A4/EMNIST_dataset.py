import gzip
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import json
from tqdm import tqdm
from glob import glob
from pathlib import Path

def char2index(*char_list):
    index_list = list()
    for char in char_list:
        assert 47 < ord(char) < 58 or \
               64 < ord(char) < 91 or \
              96 < ord(char) < 123, 'Invalid character.'
        if 47 < ord(char) < 58:
            index_list.append(ord(char) - 48)
        elif 64 < ord(char) < 91:
            index_list.append(ord(char) - 55)
        else:
            index_list.append(ord(char) - 61)
    return index_list


def dataset_filter(old_json_path: str, char_list: list=None, new_json_dir: str=None):
    spec_char = char2index(*char_list)
    new_labelList = list()
    new_imgPath = list()

    with open(old_json_path, 'r') as jsf:
        jsf = json.load(jsf)
        imageList = jsf['images']
        labelList = jsf['labels']

    for i in tqdm(range(len(labelList))):
        if labelList[i] in spec_char:
            new_imgPath.append(imageList[i])
            new_labelList.append(labelList[i])

    data = {'images': new_imgPath,
            'labels': new_labelList}

    json_filename = Path(old_json_path).stem
    new_json_path = os.path.join(*[new_json_dir,'new_{}{}'.format(json_filename,'.json')])
    with open(new_json_path, 'w', newline='') as jsonfile:
        json.dump(data, jsonfile)
        
def extract_gzip(img_filename: str, label_filename: str, target_dir: str, dataset_type: str='train'):
    dataset_types = ['train', 'test', 'valid']
    img_paths = list()

    if dataset_type not in dataset_types:
        raise ValueError("Invalid dataset type. Expected one of: %s" % dataset_types)
    os.makedirs(os.path.join(*[target_dir,'{}_images'.format(dataset_type)]), exist_ok=True)
    
    f_Img = gzip.open(img_filename, "rb")

    magic = f_Img.read(4)
    magic = int.from_bytes(magic, "big")

    num_img = f_Img.read(4)
    num_img = int.from_bytes(num_img, "big")

    row = f_Img.read(4)
    row = int.from_bytes(row, "big")

    col = f_Img.read(4)
    col = int.from_bytes(col, "big")

    image_arr = np.empty((row, col), "float32")

    for i in tqdm(range(num_img), ascii=True):
        for r in range(row):
            for c in range(col):
                image_arr[c, r] = int.from_bytes(f_Img.read(1), "big")
        image_arr = np.array(image_arr).astype(np.uint8).reshape(28,28)
        image = Image.fromarray(image_arr)
        img_path = os.path.join(*[target_dir, '{}_images'.format(dataset_type),
                                '{}.jpg'.format(str(i))])
        image.save(img_path)
        img_paths.append(img_path)
    f_Img.close()

    f_Label = gzip.open(label_filename, "rb")

    magic = f_Label.read(4)
    magic = int.from_bytes(magic, "big")

    nolab = f_Label.read(4)
    nolab = int.from_bytes(nolab, "big")

    labels = [f_Label.read(1) for i in range(nolab)]
    labels = [int.from_bytes(label, "big") for label in labels]
    f_Label.close()

    data = {'images': img_paths,
            'labels': labels}
    
    with open(os.path.join(*[target_dir,'{}_set.json'.format(dataset_type)]),
              'w', newline='') as jsonfile:
        json.dump(data, jsonfile)

def gz2jpg(gz_dataset_dir: str, mode: str, save_dataset_dir: str='.'):   
    print("Loading emnist")

    train_labels = os.path.join(*[gz_dataset_dir,"emnist-{}-train-labels-idx1-ubyte.gz".format(mode)])
    train_images = os.path.join(*[gz_dataset_dir,"emnist-{}-train-images-idx3-ubyte.gz".format(mode)])
    test_labels = os.path.join(*[gz_dataset_dir,"emnist-{}-test-labels-idx1-ubyte.gz".format(mode)])
    test_images = os.path.join(*[gz_dataset_dir,"emnist-{}-test-images-idx3-ubyte.gz".format(mode)])

    extract_gzip(test_images, test_labels, save_dataset_dir, 'test')
    extract_gzip(train_images, train_labels, save_dataset_dir, 'train')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms):
        with open(dataset_path, 'r') as jsf:
            jsf = json.load(jsf)
            self.imPath = jsf['images']
            self.labelList = jsf['labels']
        
        self.transforms=transforms
        
    def __len__(self):
        return len(self.imPath)

    def __getitem__(self, idx):
        image = Image.open(self.imPath[idx])
        label = self.labelList[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        label = torch.tensor(label)
        return image, label

if __name__ == '__main__':
    gzip_dir = './dataset/gzip'
    save_dataset_dir = './dataset/emnist-byclass_dataset'
    
    if glob(os.path.join(*[save_dataset_dir, '*.json'])) == []:
        gz2jpg(gz_dataset_dir=gzip_dir, mode='byclass', save_dataset_dir=save_dataset_dir) 
    
    #A->10 B->11 D->13 E->14 F->15 G->16 H->17 N->23 Q->26 R->27 T->29
    char_list = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T']
    if not (os.path.exists(os.path.join(*[save_dataset_dir, 'new_train_set.json']))):
        dataset_filter(old_json_path=os.path.join(*[save_dataset_dir,'train_set.json']),
                       char_list=char_list,
                       new_json_dir=save_dataset_dir)
    if not (os.path.exists(os.path.join(*[save_dataset_dir, 'new_test_set.json']))):
        dataset_filter(old_json_path=os.path.join(*[save_dataset_dir,'test_set.json']),
                       char_list=char_list,
                       new_json_dir=save_dataset_dir)

    train_dataloader = MyDataset(dataset_path=os.path.join(*[save_dataset_dir,'new_train_set.json']),
                                 transforms=None)
    test_dataloader = MyDataset(dataset_path=os.path.join(*[save_dataset_dir,'new_test_set.json']),
                                transforms=None)

    data_len = test_dataloader.__len__()
    print(f'{data_len=}')
    label_list = []
    for i in tqdm(range(data_len)):
        image, label = train_dataloader.__getitem__(i)
        label_list.append(label.item())
    print(sorted(set(label_list)))



    
        