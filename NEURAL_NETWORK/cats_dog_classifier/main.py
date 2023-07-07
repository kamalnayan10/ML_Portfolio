import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import cv2


pickle_in = open("NEURAL_NETWORK/cats_dog_classifier/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("NEURAL_NETWORK/cats_dog_classifier/y.pickle", "rb")
y = pickle.load(pickle_in)
y = np.array(y)

X = X / 255

y = y / max(y)

dense_layers = [0]
layer_sizes = [64]

conv_layers = [3]

# tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         for conv_layer in conv_layers:
#             NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"

#             tensorboard = TensorBoard(
#                 log_dir=f"D:/PROGRAMMING/PYTHON/SentDexML/NEURAL_NETWORK/cats_dog_classifier/logs/{NAME}"
#             )

#             model = Sequential()

#             model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
#             model.add(Activation("relu"))
#             model.add(MaxPooling2D(pool_size=(2, 2)))

#             for l in range(conv_layer - 1):
#                 model.add(Conv2D(layer_size, (3, 3)))
#                 model.add(Activation("relu"))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))

#             model.add(Flatten())
#             for l in range(dense_layer):
#                 model.add(Dense(layer_size))
#                 model.add(Activation("relu"))

#             model.add(Dense(1))
#             model.add(Activation("sigmoid"))

#             model.compile(
#                 loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
#             )

#             model.fit(
#                 X,
#                 y,
#                 batch_size=32,
#                 validation_split=0.1,
#                 epochs=10,
#                 callbacks=[tensorboard],
#             )
# model.save("64x3CNN.model")

model = tf.keras.models.load_model("64x3CNN.model")
CATEGORIES = ["Dog", "Cat"]


def prepare(filepath):
    IMG_SIZE = 50

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3CNN.model")

prediction = model.predict(
    [
        prepare(
            "D:/PROGRAMMING/PYTHON/SentDexML/NEURAL_NETWORK/cats_dog_classifier/cat.jpg"
        )
    ]
)

print(f"\npredcition of CNN = {CATEGORIES[int(prediction[0][0])]}")
