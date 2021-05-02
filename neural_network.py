import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPOCHS = 50
BATCH_SIZE = 2048

print('Loading data')
(features_train, labels_train), (features_test, labels_test) = keras.datasets.mnist.load_data()
print('Finished loading data')

datagen = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1
)

input_shape = (28, 28, 1)
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

features_train = np.expand_dims(features_train / 255, -1)
labels_train = np.expand_dims(labels_train, -1)
features_test = np.expand_dims(features_test / 255, -1)
labels_test = np.expand_dims(labels_test, -1)

print("Start Training")

batch = 0
for x_batch, y_batch in datagen.flow(features_train, labels_train, batch_size=BATCH_SIZE):
    model.fit(x_batch, y_batch, validation_data=(features_test, labels_test))
    batch += 1
    if batch > len(features_train) // BATCH_SIZE:
        break
print('Finished training')

print(model.evaluate(datagen.flow(features_test, labels_test)))
model.save('neural_network_model.h5')
