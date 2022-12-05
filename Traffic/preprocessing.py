import numpy as np
import pickle
from PIL import Image
import os
from tqdm import tqdm


base_dir = './data'


def load_data():
    data_dir = './'
    with open(os.path.join(data_dir, 'input', 'data2.pickle'), 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # dictionary type

    with open(os.path.join(data_dir, 'y_train.pickle'), 'rb') as fr:
        data['y_train'] = pickle.load(fr)

    with open(os.path.join(data_dir, 'y_validation.pickle'), 'rb') as fr:
        data['y_validation'] = pickle.load(fr)

    with open(os.path.join(data_dir, 'x_train.pickle'), 'rb') as fr:
        data['x_train'] = pickle.load(fr)

    with open(os.path.join(data_dir, 'x_validation.pickle'), 'rb') as fr:
        data['x_validation'] = pickle.load(fr)

    with open(os.path.join(data_dir, 'x_test.pickle'), 'rb') as fr:
        data['x_test'] = pickle.load(fr)

    return data


def convert_to_custom(annotation_path, image_path, mode='train'):
    data = load_data()
    if mode == 'train':
        print("Converting training data...")
        x = data['x_train']
        y = data['y_train']

        with open(os.path.join(annotation_path, 'train.txt'), 'w') as f:
            for idx, image in enumerate(tqdm(x)):
                low, high = np.min(image), np.max(image)
                image = (image - low) / (high - low) * 255
                img = Image.fromarray(np.uint8(image))
                index = str(idx+1).zfill(5)
                img.save(os.path.join(image_path, f'{index}.png'))

                label_id = y[idx].argmax()

                f.write(f'{index}.png {label_id}\n')
        f.close()

    elif mode == 'validation':
        print("Converting validation data...")
        x = data['x_validation']
        y = data['y_validation']

        with open(os.path.join(annotation_path, 'val.txt'), 'w') as f:
            for idx, image in enumerate(tqdm(x)):
                low, high = np.min(image), np.max(image)
                image = (image - low) / (high - low) * 255
                img = Image.fromarray(np.uint8(image))
                index = str(idx+1).zfill(4)
                img.save(os.path.join(image_path, f'{index}.png'))

                label_id = y[idx].argmax()

                f.write(f'{index}.png {label_id}\n')

    elif mode == 'test':
        print("Converting test data...")
        x = data['x_test']
        y = None

        for idx, image in enumerate(tqdm(x)):
            low, high = np.min(image), np.max(image)
            image = (image - low) / (high - low) * 255
            img = Image.fromarray(np.uint8(image))
            index = str(idx+1).zfill(5)
            img.save(os.path.join(image_path, f'{index}.png'))

    else:
        raise ValueError('mode should be train, val or test')


if __name__ == '__main__':
    train_path = os.path.join(base_dir, 'train')
    validation_path = os.path.join(base_dir, 'val')
    test_path = os.path.join(base_dir, 'test')

    annotation_path = os.path.join(base_dir, 'annotations')

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)

    convert_to_custom(annotation_path, train_path, mode='train')
    convert_to_custom(annotation_path, validation_path, mode='validation')
    convert_to_custom(annotation_path, test_path, mode='test')
