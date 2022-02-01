# First, lets get a list of all files in the data directory


import time
import sys
from tensorflow import keras
import tensorflow as tf
import pathlib
import json

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, GlobalAveragePooling3D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Conv2D, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import pickle  # used for caching
import random
import pandas as pd
from PIL import Image
import numpy as np

import multiprocessing

path = 'data/'

images_list = {}
for i in range(1, 11):
    images_list[i] = []

abs_dir = pathlib.Path(__file__).parent.absolute()

broken = []
count = 0

# This is to rename everything
for root, directories, files in os.walk(path, topdown=False):
    for name in files:
        count = count + 1
        try:
            old_name = str(os.path.join(abs_dir, root, name))
            new_name = str(
                os.path.join(abs_dir, root, name))
            if ".jpeg" not in str(name):
                print("Need to update:", name)
                os.rename(old_name, new_name)

            img = Image.open(new_name)
            img.verify()

            classification = int(root.split("/")[-1])
            images_list[classification].append(
                {"f": name, "p": str(os.path.join(abs_dir, root, name)), "t": str(classification)})
        except NameError as error:
            print(error)
        except FileNotFoundError as error:
            print(error)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("Unable to handle file: ", os.path.join(
                abs_dir, root, name))
            broken.append(name)

print("Found", len(broken), "broken items out of a total",
      count, "leaving", count - len(broken), "valid")
# Now, lets print some debug informatiton
for key in images_list:
    print(key, len(images_list[key]))
# if len(broken) > 1:
#     quit()


# for root, directories, files in os.walk(path, topdown=False):
#     for name in files:
#         try:

#             classification = int(root.split("/")[-1])
#             images_list[classification].append(
#                 {"f": name, "p": str(os.path.join(abs_dir, root, name)), "t": str(classification)})
#         except:
#             print("Unable to handle file: ", name)

# Now, lets separate everything into a training set and validation set
training_split = 0.8

training_set = []
validation_set = []
target_size = (300, 300)

# loop through each key, shuffle, then create training and validation sets
random.seed(14354)

for key in images_list:
    random.shuffle(images_list[key])
    training_length = int(len(images_list[key]) * training_split)

    training_set.extend(images_list[key][:training_length])
    validation_set.extend(images_list[key][training_length:])

# now, we need to prepare this as a data frame
training_df = pd.DataFrame(training_set)
validation_df = pd.DataFrame(validation_set)

print(training_df.at[0, 'p'])
print(training_df)

training_df.to_csv("training-df.csv")
validation_df.to_csv("validation-df.csv")


# Now, keras sttuff

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True,
    rescale=1./255)


test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=training_df,
    directory=None,
    x_col="p",
    y_col="t",
    class_mode='categorical',
    target_size=target_size,
    validate_filenames=False
    # save_to_dir="build/"
)

validation_generator = test_datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory=None,
    x_col="p",
    y_col="t",
    class_mode='categorical',
    target_size=target_size,
    validate_filenames=False
)

# # now, build the model
# if os.name == 'nt':
#     print("--------- USING GPU ACCELERATION ----------")
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)


def test_model():

    class_names = [str(idx) for idx in range(1, 11)]
    print("Class names:", class_names)

    # Combine training df and testing df
    combined_df = pd.concat([training_df, validation_df])
    combined_datagen = ImageDataGenerator(rescale=1.0/255)
    combined_generator = combined_datagen.flow_from_dataframe(
        dataframe=combined_df,
        directory=None,
        x_col="p",
        y_col="t",
        class_mode="categorical",
        classes=class_names,
        target_size=target_size,
        validate_filenames=False,
        shuffle=False,
    )

    class_indices = combined_generator.class_indices
    print(class_indices)

    class_from_index = [0] * 10
    for key in class_indices:
        class_from_index[class_indices[key]] = key

    # predict results
    model = keras.models.load_model(
        'trained_models/best_model-image-classifier.hdf5')
    model.evaluate(combined_generator)

    predictions = model.predict(combined_generator)

    count = 0
    correct = 0
    for idx in range(0, len(predictions)):
        expected = combined_df.iloc[idx, :]['t']  # YES
        actual = class_from_index[np.argmax(predictions[idx])]  # YES
        # print(type(expected), expected, type(actual), actual)
        count += 1
        if(int(expected) == int(actual)):
            correct += 1

    print("Found", correct, "correct out of", count)

    # We need to keep track of the results in each category
    # so we know how 1s were interpreted (i.e. histogram)
    # the lookup key will
    # key will be the known category
    # value will be a map of predicted values (hashmap), where the key
    # is the str predicted value
    category_store = {}
    for idx in range(1, 11):
        category_store[str(idx)] = np.zeros(10)

    # Now, we loop through the predictions and update the category store
    for idx in range(0, len(predictions)):
        expected = combined_df.iloc[idx, :]['t']  # String
        pred_index = np.argmax(predictions[idx])  # array index
        pred_value = class_from_index[pred_index]  # String

        new_index = class_names.index(pred_value)

        # this returns the class_names INDEX!!!
        category_store[expected][new_index] += 1
    # Now, lets reformat category store so it is more usable
    for key in category_store:
        item = category_store[key]
        result = {
            'data': item.astype(int).tolist(),
            'count': int(item.astype(int).sum())
        }
        result['accuracy'] = item[class_names.index(key)] / result['count']
        category_store[key] = result

    # Now build summary
    summary = {}
    for key in category_store:
        summary[key] = {
            'accuracy': category_store[key]['accuracy'],
            'count': category_store[key]['count']
        }

    category_store['summary'] = summary

    print(json.dumps(category_store, indent=4))

    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=np.inf)
    # yep

    with open('output.json', 'w') as f:
        f.write(json.dumps(category_store, indent=4))


def train_model():
    # Lets save out the value mapping
    np.save('class_indices', train_generator.class_indices)

    # Lets build class_weights
    # We need to be sure to remap the weights to the correct values (i.e. following class_indices)
    val_counts = training_df['t'].value_counts()
    class_weights = {}
    for name, val in val_counts.items():
        class_weights[name] = val
    max_val = val_counts.max()
    for key in class_weights:
        class_weights[key] = max_val / class_weights[key]
    reordered_class_weights = {}
    for key in train_generator.class_indices:
        reordered_class_weights[train_generator.class_indices[key]
                                ] = class_weights[key]
    print("class weights:", reordered_class_weights)

    # for dense_layer in dense_layers:
    #     for filter_size in filter_sizes:
    #         for conv_layer in conv_layers:
    conv_layer = 3
    filter_size = 64
    dense_layer = 1
    NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
        conv_layer, filter_size, dense_layer, int(time.time()))
    print(NAME)

    tensorboard = TensorBoard(
        log_dir="logs/{}".format(NAME), update_freq="epoch", profile_batch=0)

    # Recognize model checkpoint now
    filepath = 'trained_models/best_model-image-classifier.hdf5'
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor="val_accuracy",
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    model = keras.Sequential(
        [
            InputLayer(input_shape=(target_size[0], target_size[1], 3)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.8),
            Dense(10, activation="softmax"),
        ]
    )

    METRICS = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision')
    ]

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=METRICS)

    model.summary()

    model.fit(train_generator,
              validation_data=validation_generator,
              epochs=50,
              workers=multiprocessing.cpu_count(),
              callbacks=[tensorboard, checkpoint],
              shuffle=True,
              class_weight=reordered_class_weights
              )


if __name__ == '__main__':

    train_model()

    test_model()
    # At this point, we should have our
