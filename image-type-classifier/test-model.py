from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import json

# now, build the model
if os.name == 'nt':
    print("--------- USING GPU ACCELERATION ----------")
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = keras.models.load_model(
    'trained_models/best_model-image-classifier.hdf5')

model.summary()


np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

category_mapper = [0, 2, 3, 4, 5, 6, 7, 8, 9, 1]

category_store = {}

for folder_idx in range(1, 11):
    results = np.zeros(10)
    # load all images in a folder
    # This is to rename everything
    for root, directories, files in os.walk("data/" + str(folder_idx) + "/", topdown=False):
        for name in files:
            try:
                # let's load a test file
                img = keras.preprocessing.image.load_img(
                    os.path.join(root, name)
                )

                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create a batch

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                results = np.add(results, predictions[0])

                print(results)
            except:
                print("Unable to load image:", name)

    # Remap categories for easier understanding
    remapped_counts = results[category_mapper]

    # convert all output to integers
    remapped_counts = remapped_counts.astype(int)

    # Time to calculate some statistics
    count = int(remapped_counts.sum())

    # compute accuracy
    accuracy = remapped_counts[folder_idx - 1] / (count + 0.0)

    print(results)
    print(remapped_counts)
    print(accuracy)

    # store results
    category_store[str(folder_idx)] = (
        {'data': remapped_counts.tolist(), 'accuracy': accuracy, 'count': count})


# build summary
summary = {}
for item in range(1, 11):
    summary[str(item)] = category_store[str(item)]['accuracy']

category_store['summary'] = summary

print("----- Final results -----")
print(json.dumps(category_store))

with open('output.json', 'w') as f:
    f.write(json.dumps(category_store))
