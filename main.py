import numpy as np
import pandas as pd
from model.SimCLR import ContrastiveModel
import keras
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from keras.src.losses import CategoricalCrossentropy
from keras.src.metrics import CategoricalAccuracy
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.optimizers.schedules import InverseTimeDecay
from keras.src.optimizers import Adam
from keras.src.layers import Dense
from data_util import createTrainingDataset, createValidationDataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the accuracy of a model')
    parser.add_argument('--epoch_pretrain', type=str, required=False, default=100, help='Number of epoches in pretraining model')
    parser.add_argument('--epoch_finetune', type=str, required=False, default=100, help='Number of epoches in finetuning model')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory to training and validation dataset')
    parser.add_argument('--batch_size', type=str, required=False, default=1024, help='Batch size of the training dataset')
    
    return parser.parse_args()

def export_result(model, val_dataset, val_path, output_path):
    dataframe = pd.read_csv(val_path)
    prediction = model.predict(val_dataset)
    predicted_classes = np.argmax(prediction, axis=1)
    df = pd.DataFrame({'image_name': dataframe['image_name'], 'label': predicted_classes})
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    conf = parse_args()
    train_directory = f'{conf.root_dir}/train'
    validation_directory = f'{conf.root_dir}/val'

    IMAGE_SIZE = 64
    NUM_CLASSES = 2139

    training_dataset, NUM_CLASSES = createTrainingDataset(train_directory, conf.batch_size)
    validation_dataset = createValidationDataset(validation_directory, NUM_CLASSES)

    checkpoint_callback = ModelCheckpoint(
        f'cp-epoch{conf.epoch_finetune}.weights.h5',
        verbose=1,
        monitor='val_accuracy',
        save_weights_only=True
    )
    early_stopping = EarlyStopping(patience=15)
    initial_learning_rate = 0.1
    decay_steps = 1.0
    decay_rate = 0.5
    learning_rate_fn = InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)

    pretraining_model = ContrastiveModel(num_classes=NUM_CLASSES)
    pretraining_model.compile(
        contrastive_optimizer=Adam(),
        probe_optimizer=Adam(),
    )
    with tf.device('/gpu:0'):
        history = pretraining_model.fit(
            training_dataset, epochs=conf.epoch_pretrain, validation_data=validation_dataset
        )

    finetuning_model = keras.Sequential(
        [
            pretraining_model.encoder,
            Dense(NUM_CLASSES, activation='softmax'),
        ],
        name="finetuning_model",
    )

    finetuning_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_fn),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy(name="accuracy")],
    )
    with tf.device('/gpu:0'):
        finetuning_history = finetuning_model.fit(
            training_dataset, epochs=conf.epoch_finetune, validation_data=validation_dataset,
            verbose=1,
            callbacks=[checkpoint_callback, early_stopping]
        )

    finetuning_model.evaluate(validation_dataset)
    export_result(finetuning_model, validation_dataset, f"{validation_directory}/labels.csv", "output.csv")