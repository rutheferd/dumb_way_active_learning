# https://www.tensorflow.org/tutorials/images/classification
#%%
from http.client import LineTooLong
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
import pandas as pd

# %%
def load_data(path=None):
    # Load Training Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


# %%
# Standardize the Data
# NOTE: Not needed to re-do
# NOTE: Can I save a model without data??
def build_model(class_names):
    normalization_layer = layers.Rescaling(1.0 / 255)

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
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


# %%
def train(model, train_ds, val_ds):
    epochs = 10
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return model, history


#%%
print(history.history)


# %%
def save_convert(model):
    # Save the model:
    model.save("latest_model")
    # Convert to TFLite
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model("latest_model")
    # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)


# %%
def run_inference(class_names, in_data, ext="png", num=None):
    # Dirty coding the other side
    # Possible Name: Generate Label Queries
    new_model = tf.keras.models.load_model("latest_model")

    filelist = glob.glob(in_data + "/*.{}".format(ext))

    saved_predictions = []
    confidence_distribution = []

    # x = np.array([np.array(cv2.imread(fname)) for fname in filelist])

    for i, e in enumerate(filelist):
        if num is not None:
            if i >= num:
                break
        try:
            img = tf.keras.utils.load_img(e, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)

            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = new_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            conf = 100 * np.max(score)
            confidence_distribution.append(conf)

            print(
                "{}/{} This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                    i, len(filelist), class_names[np.argmax(score)], 100 * np.max(score)
                )
            )

            if conf < 60.0:
                saved_predictions.append(i)
        except:
            # Dangerous, need to figure out how to define error
            # UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x7fb12c729eb8>
            continue

    return filelist, saved_predictions, len(saved_predictions), confidence_distribution


# %%
def distribute_data(saved_predictions, filelist):
    temp_to_label = "../data/to_label"
    num_label = 5
    please_label = random.sample(saved_predictions, num_label)
    for i in please_label:
        filename = filelist[i].split("/")[-1]
        os.rename(filelist[i], temp_to_label + "/" + filename)


# %%
# Get Stats
# NOTE: Very image dataset dependent!
def get_stats(history, data_dir, ext):
    list = glob.glob(data_dir + "/**/*.{}".format(ext))
    num_labeled = len(list)
    hist = history.history
    avg_acc = np.mean(hist["accuracy"])
    avg_vacc = np.mean(hist["val_accuracy"])
    max_acc = np.max(hist["accuracy"])
    max_vacc = np.max(hist["val_accuracy"])
    return [num_labeled, avg_acc, avg_vacc, max_acc, max_vacc]


# %%
# Save Stats
# FIXME: Should make this kwargs and loop through stats.
# FIXME: Need number of saved predictions
# FIXME: Will likely remove df return in favor of a function to generate the plots
#        frome the saved stats file.
def save_stats(stat_list, stat_file):
    """This saves the statistics into the stats.csv file. That file's headers
       must be compliant with the list input.

    Args:
        stat_list (list): list of stats provided by the get_stats function.
        stat_file (path): path of the stat file to load from/save to.

    Returns:
        dataframe: dataframe with updated stats for the plotting method.
    """
    # Load Stats
    df = pd.read_csv(stat_file, index_col=0)
    # append run stats
    line_item = np.array(stat_list)
    df.loc[len(df)] = line_item
    df.to_csv(stat_file)

    return df


# First Round 1760
# Should categorize all 10's place bins to understand better exactly how confidence
# rose given the active learning.
# Second Round 1131
# Third Round 813


# %%
batch_size = 32
img_height = 480
img_width = 720

data_dir = "/Users/aruth3/Documents/repos/dumb_way_active_learning/PetImages"
unlabel_dir = "/Users/aruth3/Documents/repos/dumb_way_active_learning/Pet_Unlabeled"
stat_file = "/Users/aruth3/Documents/repos/dumb_way_active_learning/stats.csv"

# %%
# Load Data
# FIXME: Need a Test Dataset
train_ds, val_ds, class_names = load_data(data_dir)

# %%
# Build Model
model = build_model(class_names)

# %%
# Train Model
model, hist = train(model, train_ds, val_ds)


# %%
# TODO: Need to add git commit for automation of loading new
# model onto the TPU or the Drone
# FIXME: Move save convert to after the statistical analysis
# decide if we should save the model.
save_convert(model)

# %%
# FIXME: Need to shuffle the data here.
filelist, saved_predictions, num_saved, confidences = run_inference(
    class_names, unlabel_dir, "jpg"
)

# %%
# Distribute Samples
# FIXME: Sample larger than population or is negative
distribute_data(saved_predictions, filelist)

# %%
stat_list = get_stats(hist, data_dir, "jpg")
plot_stat = save_stats(stat_list, stat_file)

# %%
acc_plot = plot_stat.plot.line(x="num_labelled", y=["avg_vacc", "max_vacc"])
fig = acc_plot.get_figure()
fig.savefig("accuracy.pdf")

# %%
def run_request():
    train_ds, val_ds, class_names = load_data(data_dir)
    model = build_model(class_names)
    model, hist = train(model, train_ds, val_ds)
    save_convert(model)
    filelist, saved_predictions, num_saved, confidences = run_inference(
        class_names, unlabel_dir, "jpg"
    )
    # FIXME: Add this to a method
    plt.hist(confidences, bins="auto")
    distribute_data(saved_predictions, filelist)
    stat_list = get_stats(hist, data_dir, "jpg")
    plot_stat = save_stats(stat_list, stat_file)
    acc_plot = plot_stat.plot.line(x="num_labelled", y=["avg_vacc", "max_vacc"])
    fig = acc_plot.get_figure()
    fig.savefig("accuracy.pdf")


# %%
run_request()
# %%
