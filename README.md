# Application and Practice in Neural Networks

This repository contains project for class Application and Practice in Neural Networks<br>

This project is for detecting disease through endoscopy image.
<br>

This code is based on mmdetection and used Faster-RCNN for baseline.

## Usage

### environments
- Ubuntu 20.04
- Python 3.8.5

### Requirements
- torch >= 1.8.0
- torchvision
- mmcv-full
- mmdet
- tqdm
- pandas

To use this code, please first install the 'mmcv-full' and 'mmdet' by following the official guideline guidelines ([`mmdet`](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)).


### Pre-trained Faster-RCNN model
please download the pre-trained via shell script
```shell
sh checkpoints/download.sh
```

### Preprocessing the dataset
The following dataset is preprocessed in COCO format , but if you are using the raw json file you can preprocess with the script
```shell
python tools/preprocessing.py
```

The final folder structure should look like this:
```none
FasterRCNN
├── ...
├── dataset
│   ├── test_img
│   │   ├── test_100002.jpg
│   │   ├── test_100003.jpg
│   │   ├── test_100035.jpg
│   │   ├── test_100047.jpg
│   │   ├── ...
│   ├── train
│   │   ├── train_100002.json
│   │   ├── trina_100003.json
│   │   ├── train_100035.json
│   │   ├── train_100047.json
│   │   ├── ...
│   ├── train_img
│   │   ├── train_100002.jpg
│   │   ├── train_100003.jpg
│   │   ├── train_100035.jpg
│   │   ├── train_100047.jpg
│   │   ├── ...
│   ├── valid_img
│   │   ├── valid_100002.jpg
│   │   ├── valid_100003.jpg
│   │   ├── valid_100035.jpg
│   │   ├── valid_100047.jpg
│   │   ├── ...
│   ├── class_id_info.csv
│   ├── train_annotations.json
│   ├── valid_annotations.json
├── ...
```

### Training
For convenience, provides and [annotated config file](configs/faster_rcnn_r50_1x_medi.py) of the detection model

A training job can be launched using:

```shell
sh dist_train.sh configs/faster_rcnn_r50_1x_medi.py 4
```

### Evaluating

The checkpoint will be sevaed automaticallt in work_dirs, else you set a directory for it.

```shell
sh dist_test configs/faster_rcnn_r50_1x_medi.py /path/to/checkpoint 4 --eval bbox
```