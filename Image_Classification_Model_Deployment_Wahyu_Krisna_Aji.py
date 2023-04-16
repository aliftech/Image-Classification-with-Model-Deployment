import os
import zipfile
import glob

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

from keras.preprocessing import image
import pandas as pd
# from google.colab import files
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage


file_name = 'archive.zip'
extract_file = zipfile.ZipFile(file_name, 'r')
extract_file.extractall('/alas-kaki')
extract_file.close()

main_repo = "/alas-kaki/Shoe vs Sandal vs Boot Dataset"

list_main_repo = os.listdir(main_repo)
list_main_repo

sandal_repo = os.path.join(main_repo, 'Sandal')
shoe_repo = os.path.join(main_repo, 'Shoe')
boot_repo = os.path.join(main_repo, 'Boot')

total_sandal = len(os.listdir(sandal_repo))
total_shoe = len(os.listdir(shoe_repo))
total_boot = len(os.listdir(boot_repo))

print("Total Sandal    : ", total_sandal)
print("Total Shoe   : ", total_shoe)
print("Total Boot   : ", total_boot)

training_data = ImageDataGenerator(
    rotation_range = 30,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest",
    rescale = 1./255,
    validation_split = 0.2
)

validation_data = ImageDataGenerator(
    rotation_range = 30,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest",
    rescale = 1./255,
    validation_split = 0.2
)

generator_training_data = training_data.flow_from_directory(
    main_repo,
    target_size = (136,102),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 128,
    shuffle = True,
    subset = "training"
)

generator_validation_data = validation_data.flow_from_directory(
    main_repo,
    target_size = (136,102),
    color_mode = "rgb",
    class_mode = "categorical",
    batch_size = 128,
    shuffle = False,
    subset = "validation"
)

Image_Model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(136,102, 3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.5),  
    Conv2D(64, (3,3), activation='relu'), 
    MaxPooling2D(2,2),
    Dropout(0.5),  
    Flatten(), 
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(3, activation='softmax')
])

Adam(learning_rate=0.00146, name='Adam')
Image_Model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])


class newCallbacks(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.9):
      print("Akurasi telah tercapai!")
      self.model.stop_training = True

cb = newCallbacks()

Image_Model.summary()

model_hist = Image_Model.fit(
    generator_training_data,
    validation_data=generator_validation_data,
    epochs=50,
    verbose=2,
    batch_size=128,
    callbacks=[cb]
)

plt.plot(model_hist.history['accuracy'])
plt.plot(model_hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

# Membuat model dengan format pb
export_repo = 'saved_model/'
tf.saved_model.save(Image_Model, export_repo)
 
# Konversi model pb ke tflite
model_converter = tf.lite.TFLiteConverter.from_saved_model(export_repo)
tflite_converted_model = model_converter.convert()
 
with open('sandal.tflite', 'wb') as f:
  f.write(tflite_converted_model)