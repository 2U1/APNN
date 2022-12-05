import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import base64
import os
import random
from PIL import Image
from io import BytesIO
from glob import glob
from tqdm import tqdm
import re

from collections import defaultdict

base_dir = "/home/workspace/dataset"

def convert_to_coco(json_paths, save_path, mode='train'):
    """
        only for train dataset
    """
    res = defaultdict(list)
    categories = {
        '01_ulcer': 1,
        '02_mass': 2,
        '04_lymph': 3,
        '05_bleeding': 4
    }
    
    n_id = 0
    for json_path in tqdm(json_paths):
        with open(json_path, 'r') as f:
            tmp = json.load(f)
            

        # [추가 ] train image 저장
        image = BytesIO(base64.b64decode(tmp['imageData']))
        image = Image.open(image).convert('RGB')

        if mode =='train':
            img_name = tmp['file_name'].split(".")[0]
            image.save(os.path.join(base_dir, "train_img", img_name +".jpg"))

        elif mode == 'valid':
            img_name = re.sub('train','valid',tmp['file_name'].split(".")[0])
            image.save(os.path.join(base_dir, "valid_img", img_name +".jpg"))

        elif mode == 'test':
            img_name = re.sub('train','test',tmp['file_name'].split(".")[0])
            image.save(os.path.join(base_dir, "test_img", img_name +".jpg"))
        
        image_id = int(tmp['file_name'].split('_')[-1][:6])
        res['images'].append({
            'id': image_id,
            'width': tmp['imageWidth'],
            'height': tmp['imageHeight'],
            'file_name': img_name +".jpg", # 변경
        })
        
        for shape in tmp['shapes']:
            box = np.array(shape['points']) # 추가
            x1, y1, x2, y2 = \
                    min(box[:, 0]), min(box[:, 1]), max(box[:, 0]), max(box[:, 1])
            
            w, h = x2 - x1, y2 - y1
            
            res['annotations'].append({
                'id': n_id,
                'image_id': image_id,
                'category_id': categories[shape['label']],
                'area': w * h,
                'bbox': [x1, y1, w, h],
                'iscrowd': 0,
            })
            n_id += 1
    
    for name, id in categories.items():
        res['categories'].append({
            'id': id,
            'name': name,
        })
        
    with open(save_path, 'w') as f:
        json.dump(res, f)

def convert_and_split():
    
    random.seed(10)

    train_files = glob(os.path.join(base_dir, 'train/*.json'))

    random.shuffle(train_files)

    # 8:2로 학습/검증 데이터 분리
    split_train = int(len(train_files)*0.8)
    split_test= int(len(train_files)*0.1)
    
    print("split_train :", split_train)
    print("split_test :", split_test)

    train_file = train_files[:split_train]
    valid_file = train_files[split_train:(split_train+split_test)]
    test_file = train_files[(split_train+split_test):]
    
    convert_to_coco(train_file, os.path.join(base_dir, 'train_annotations.json'), mode='train')
    convert_to_coco(valid_file, os.path.join(base_dir, 'valid_annotations.json'), mode='valid')
    convert_to_coco(test_file, os.path.join(base_dir, 'test_annotations.json'), mode='test')

# def convert_test():
#     test_files = sorted(glob(os.path.join(base_dir, 'test/*')))

#     test_json_list = []
#     for file in tqdm(test_files):
#         with open(file, "r") as json_file:
#             test_json_list.append(json.load(json_file))

#     save_path_dir = os.path.join(base_dir, "test_img")
    
#     if not os.path.exists(save_path_dir):
#         os.makedirs(save_path_dir)

#     for sample in tqdm(test_json_list):
        
#         image_id = sample['file_name'].split(".")[0]
#         image = BytesIO(base64.b64decode(sample['imageData']))
#         image = Image.open(image).convert('RGB')
        
#         image.save(os.path.join(base_dir, "test_img", image_id+".jpg"))


if __name__ == '__main__':
    train_path = os.path.join(base_dir, 'train_img')
    valid_path = os.path.join(base_dir, 'valid_img')
    test_path = os.path.join(base_dir, 'test_img')
    
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)


    convert_and_split()