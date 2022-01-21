# https://www.tensorflow.org/tutorials/images/classification
#%%
from statistics import mode
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import glob
import os
import random
import cv2

#%%
batch_size = 32
img_height = 480
img_width = 720

data_dir = "/Users/aruth3/Documents/repos/dumb_way_active_learning/PetImages"

# %%
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names

# %%
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# %%
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %%
# Standardize the Data
# NOTE: Not needed to re-do
normalization_layer = layers.Rescaling(1.0 / 255)
# %%
num_classes = len(class_names)

model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

# %%
# NOTE: Not needed to re-do
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# %%
# NOTE: Not needed to re-do
model.summary()

# %%
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# %%
# Save the model:
model.save("latest_model")

# %%
# Convert to TFLite
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("latest_model")
# path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# %%
# Dirty coding the other side
# Possible Name: Generate Label Queries
new_model = tf.keras.models.load_model("latest_model")

unlabel_dir = "/Users/aruth3/Documents/repos/dumb_way_active_learning/Pet_Unlabeled"

# %%
filelist = glob.glob(unlabel_dir + "/*.jpg")

saved_predictions = []

# x = np.array([np.array(cv2.imread(fname)) for fname in filelist])

for i, e in enumerate(filelist):
    if i >= 10000:
        break
    img = tf.keras.utils.load_img(e, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    conf = 100 * np.max(score)

    print(
        "{}/{} This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            i, len(filelist), class_names[np.argmax(score)], 100 * np.max(score)
        )
    )

    if conf < 60.0:
        saved_predictions.append(i)

# %%
temp_to_label = "/Users/aruth3/Documents/repos/dumb_way_active_learning/to_label"
num_label = 5
please_label = random.choices(saved_predictions, k=num_label)
for i in please_label:
    filename = filelist[i].split("/")[-1]
    os.rename(filelist[i], temp_to_label + "/" + filename)

# First Round 1760
# Should categorize all 10's place bins to understand better exactly how confidence
# rose given the active learning.
# Second Round 1131
# Third Round 813

# %%
# Some TFLite Inference
import tflite_runtime.interpreter as tflite
