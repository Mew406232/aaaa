import tensorflow as tf
import os
import numpy as np
import pathlib
import PIL
import PIL.Image
#import matplotlib.pyplot as plt
"""test os
#temp=os.listdir("./hw1/HW2_dataset/training")
#print(temp)
#print(len(temp))
#print(temp[1])
"""
train_data_dir = pathlib.Path("./hw1/HW2_dataset/training").with_suffix('')
test_data_dir=pathlib.Path("./hw1/HW2_dataset/testing").with_suffix('')
print(train_data_dir)
#image_count = len(list(train_data_dir.glob('*/*.jpg')))
#print(image_count)

#image_count = len(list(train_data_dir.glob('Alfred_Sisley/*.jpg')))
#print(image_count)

#display test
#im = list(train_data_dir.glob('Alfred_Sisley/*'))
#im_test=PIL.Image.open(str(im[0]))
#print(im_test)
#im_test.show


#training configuration
batch_size = 16
img_height = 512
img_width = 512
#---------------------------------------------------------------------------------------------------------------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_data_dir,
  validation_split=0.2,
  labels='inferred',
  label_mode='int',
  class_names=None,
  subset="training",
  seed=123,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=batch_size,
  interpolation='bilinear'
  )
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(train_data_dir,
  validation_split=0.2,
  labels='inferred',
  label_mode='int',
  class_names=None,
  subset="validation",
  seed=123,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=batch_size,
  interpolation='bilinear'
  )

#tf.keras.preprocessing.image_dataset_from_directory
#tf.keras.utils.image_dataset_from_directory
class_names = train_ds.class_names
print(class_names)

#normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

num_classes = len(class_names)
print(num_classes)
#os.system("pause")


model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

#DATASET_SIZE=len(list(train_data_dir.glob('*/*.jpg')))
#print(DATASET_SIZE)
#train_size = int(0.8 * DATASET_SIZE)
#val_size = int(0.2 * DATASET_SIZE)
#full_dataset =train_ds.shuffle(DATASET_SIZE)
#train_dataset = full_dataset.take(train_size)
#val_dataset = full_dataset.skip(train_size)



model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


model.fit(
  train_ds,
  validation_data=valid_ds,
  epochs=10
)

os.system("pause")










"""
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

"""