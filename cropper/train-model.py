# Written by John Nelson (jcnelson2185@gmail.com)

import json
import os
import time

from PIL import Image, ImageDraw
from keras_preprocessing.image.utils import validate_filename
from tensorflow import keras
import pandas as pd
import random
import multiprocessing
import numpy as np

# keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Conv2D, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# directory to store images
image_output_dir = "data/right-extra-oral"

# coco data path
coco_data_path = "formatted-coco.json"

NUMBER_OF_OUTPUTS = 12

TARGET_SIZE = (512, 512)


def load_coco_data():
    with open(coco_data_path) as json_file:
        return json.load(json_file)


def visualize_image(coco_item):
    im = Image.open(coco_item['system_path'])
    im.thumbnail(TARGET_SIZE, Image.ANTIALIAS)  # resize
    r = 2
    image_size = im.size
    draw = ImageDraw.Draw(im)
    for point in coco_item["annotations"]:
        center = (point['x'] * image_size[0],
                  point['y'] * image_size[1])
        draw.ellipse((center[0] - r, center[1] - r,
                     center[0] + r, center[1] + r), 'red')
    # Draw annotations

    for point in coco_item["predicted-annotations"]:
        r = 1
        center = (point['x'] * image_size[0],
                  point['y'] * image_size[1])
        draw.ellipse((center[0] - r, center[1] - r,
                     center[0] + r, center[1] + r), 'green')
    im.show()


def split_training_validation_set(input, seed=14354):
    training_split = 0.8
    # loop through each key, shuffle, then create training and validation sets
    random.seed(seed)

    random.shuffle(input)
    training_length = int(len(input) * training_split)

    return input[:training_length], input[training_length:]


def build_model(n_outputs):
    model = keras.Sequential(
        [
            InputLayer(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.8),
            Dense(n_outputs),
        ]
    )

    METRICS = [
        keras.metrics.MeanAbsoluteError(
            name="mean_absolute_error", dtype=None),
    ]

    model.compile(optimizer='adam',
                  loss='mae',
                  metrics=METRICS)

    model.summary()

    return model


def train_model(train_generator, validation_generator):
    conv_layer = 3
    filter_size = 64
    dense_layer = 1
    NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
        conv_layer, filter_size, dense_layer, int(time.time()))
    print(NAME)

    tensorboard = TensorBoard(
        log_dir="logs/{}".format(NAME), update_freq="epoch", profile_batch=0)

    # setup model checkpoint
    # Recognize model checkpoint now
    filepath = 'trained_models/right-extra-oral.hdf5'
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor="val_mean_absolute_error",
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    model = build_model(NUMBER_OF_OUTPUTS)

    model.fit(train_generator,
              validation_data=validation_generator,
              epochs=50,
              workers=multiprocessing.cpu_count(),
              callbacks=[tensorboard, checkpoint],
              shuffle=True
              )


def test_model(validation_df, validation_datagen, validation_generator):
    model = keras.models.load_model("trained_models/right-extra-oral.hdf5")
    model.evaluate(validation_generator)
    predictions = model.predict(validation_generator)

    print(predictions)


def debug_validation(validation_df):
    model = keras.models.load_model("trained_models/right-extra-oral.hdf5")
    validation_datagen = ImageDataGenerator(rescale=1./255)
    y_col_vals = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2',
                  'x3', 'y3', 'x4', 'y4', 'x5', 'y5']
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory=None,
        x_col="system_path",
        class_mode="multi_output",
        target_size=TARGET_SIZE,
        y_col=y_col_vals,
        validate_filenames=False,
        shuffle=False
    )

    predictions = model.predict(validation_generator)

    coco_items = []

    for idx in range(0, len(predictions)):
        coco_item = {"system_path": validation_df.iloc[idx, :]['system_path'], "annotations": [
        ], "predicted-annotations": []}
        known_coordinates = []
        for item in y_col_vals:
            known_coordinates.append(validation_df.iloc[idx, :][item])
        for coord in range(0, 6):
            coco_item["annotations"].append(
                {"x": float(known_coordinates[coord * 2]), "y": float(known_coordinates[coord * 2 + 1])})
            coco_item["predicted-annotations"].append(
                {"x": float(predictions[idx][coord * 2]),
                    "y": float(predictions[idx][coord * 2 + 1])}
            )
        coco_items.append(coco_item)

    print(json.dumps(coco_items, indent=4))
    # Now loop through the predictions and pair them appropraitely

    # model = keras.models.load_model("trained_models/right-extra-oral.hdf5")
    # predictions = model.predict(validation_generator)
    # for idx in validation_generator:
    #     print(idx)
    # return
    # Now for each prediction, build the cocoitem


def test_image(path_to_image):
    # build array
    model = keras.models.load_model("trained_models/right-extra-oral.hdf5")
    fake = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2',
            'x3', 'y3', 'x4', 'y4', 'x5', 'y5']
    input = {"system_path": path_to_image}
    for x in fake:
        input[x] = 0.0

    df = pd.DataFrame([input])
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col="system_path",
        class_mode="multi_output",
        target_size=TARGET_SIZE,
        y_col=fake,
        validate_filenames=False
    )
    predictions = model.predict(validation_generator)

    print(predictions)

    # Built coco item so we can display
    coco_item = {
        "system_path": path_to_image,
        "annotations": []
    }
    for idx in range(0, 6):
        coco_item["annotations"].append(
            {"x": predictions[0][idx * 2], "y": predictions[0][idx * 2 + 1]})

    visualize_image(coco_item)


if __name__ == "__main__":
    # Okay, load the coco data
    coco_data = load_coco_data()

    # update the system path of all images
    base_path = os.path.dirname(os.path.abspath(__file__))
    for item in coco_data:
        item['system_path'] = os.path.join(base_path, item['system_path'])

    # lets visualize some of our images to make sure things are plotting right
    # visualize_image(coco_data[2])

    # We need to build our dataframe where the input is the filepath
    # and the output will be the 6 coordinates
    # rebuild the dataframe
    all_df = []
    for item in coco_data:
        new_df = {"system_path": item['system_path']}
        idx = 0
        for annotation in item['annotations']:
            new_df['x' + str(idx)] = annotation['x']
            new_df['y' + str(idx)] = annotation['y']
            idx += 1

        all_df.append(new_df)

    # split into training set and validation set
    training_set, validation_set = split_training_validation_set(all_df)

    # convert to dataframe
    training_df = pd.DataFrame(training_set)
    validation_df = pd.DataFrame(validation_set)

    # Now keras stuff
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1./255
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    print("Building training generator...")
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=training_df,
        directory=None,
        x_col="system_path",
        class_mode="multi_output",
        target_size=TARGET_SIZE,
        y_col=['x0', 'y0', 'x1', 'y1', 'x2', 'y2',
               'x3', 'y3', 'x4', 'y4', 'x5', 'y5'],
        validate_filenames=False
    )

    print("Building validation generator")
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory=None,
        x_col="system_path",
        class_mode="multi_output",
        target_size=TARGET_SIZE,
        y_col=['x0', 'y0', 'x1', 'y1', 'x2', 'y2',
               'x3', 'y3', 'x4', 'y4', 'x5', 'y5'],
        validate_filenames=False
    )

    print("Generators are finished build...")

    # train_model(train_generator, validation_generator)

    # test_model(validation_df, validation_datagen, validation_generator)

    # test_image("/Users/jcnelson/Desktop/test/142826601.V20.jpeg")
    debug_validation(validation_df)
