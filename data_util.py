import os
import pandas as pd
import multiprocessing
import tensorflow as tf
import numpy as np
import cv2 as cv

def image_read(img_link, img_size):
    image_process = cv.imread(img_link)
    image_process = cv.resize(image_process, (img_size, img_size), \
                              interpolation=cv.INTER_LINEAR)

    img1 = cv.cvtColor(image_process, cv.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float32)
    return img1

def image_processing_worker(path_label):
    path, label = path_label
    img_read = image_read(path, 64)
    img = [(img_read, label)]
    return img

def createTrainingDataset(dataset_path, batch_size):
    def createListDir(file_path):

        listDir = [[f'{file_path}/{path}/{file}' for file in os.listdir(f'{file_path}/{path}')] \
                   for path in os.listdir(file_path)]
        flat_list = [num for sublist in listDir for num in sublist]
        flat_list = [(f, int(f.split('/')[-2])) for f in flat_list]
        return flat_list

    listDir = createListDir(dataset_path)
    train_df = pd.DataFrame(listDir, columns =['file_path', 'label'])
    data = [tuple(row) for row in train_df.itertuples(index=False)]
    
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    images_label = pool.map(image_processing_worker, data)
    pool.close()
    pool.join()
    
    images_label = [item for sublist in images_label for item in sublist]

    train_images = [t[0] for t in images_label]
    train_labels = [t[1] for t in images_label]
    num_classes = max(train_labels) + 1
    train_labels = tf.one_hot(train_labels, depth=num_classes)
    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    
    def load_and_preprocess_image(img, label):
        image = tf.convert_to_tensor(img, dtype=tf.float32)
        image = tf.expand_dims(image, axis=2)
        return image, label
    
    dataset = dataset.map(load_and_preprocess_image)
    dataset = dataset.shuffle(buffer_size=len(train_images)).batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print('Training dataset created!')
    return dataset, num_classes

def createValidationDataset(dataset_path, num_classes):
    image_folder = f'{dataset_path}/images'
    csv_file = f'{dataset_path}/labels.csv'
    dataframe = pd.read_csv(csv_file)
    image_paths = [os.path.join(image_folder, f'{image_name}.jpg') for image_name in dataframe['image_name']]
    validation_images = [image_read(path, 64).astype(np.float32) for path in image_paths]
    labels = dataframe['label']
    labels = tf.one_hot(labels, depth=num_classes)
    dataset = tf.data.Dataset.from_tensor_slices((validation_images, labels))

    def load_and_preprocess_image(img, label):
        image = tf.convert_to_tensor(img, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        image = tf.expand_dims(image, axis=3)
        label = tf.expand_dims(label, axis=0)
        return image, label
    dataset = dataset.map(load_and_preprocess_image)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print('Validation dataset created!')
    return dataset