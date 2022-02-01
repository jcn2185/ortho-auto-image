import os
import tensorflow as tf
from tensorflow import keras
print(keras.__version__)

print(tf.__version__)


if os.name == 'nt':
    print("--------- USING GPU ACCELERATION ----------")
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
