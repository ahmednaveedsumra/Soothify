import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import utils
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.utils import load_img
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint,EarlyStopping
import scipy
es = EarlyStopping(monitor="val_accuracy",min_delta=0.01,patience=5,verbose=1,mode='auto')
mc = ModelCheckpoint(filepath=r"best_model.h5",monitor="val_accuracy",verbose=1,save_best_only=True,mode='auto')
callback = [es,mc]
train_datagen = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, horizontal_flip=True, rescale=1./255)
train_data = train_datagen.flow_from_directory(directory=r"C:\Users\naima\Desktop\FYP Finall\train", target_size=(224,224),batch_size=32)
train_data.class_indices
val_datagen = ImageDataGenerator(rescale=1/255)
val_data = val_datagen.flow_from_directory(directory=r"C:\Users\naima\Desktop\FYP Finall\train", target_size=(224,224),batch_size=32)


base_model = MobileNet(input_shape=(224,224,3), include_top= False )

for layer in base_model.layers:
    layer.trainable = False


x = Flatten()(base_model.output)
x = Dense(units=7 , activation='relu' )(x)

# creating our model.
model = Model(base_model.input, x)
model.compile(optimizer='adam', loss= categorical_crossentropy, metrics=['accuracy'])
hist = model.fit_generator(train_data,
                           steps_per_epoch=10,
                           epochs=1000,
                           validation_data= val_data,
                           validation_steps=8,
                           callbacks=[es,mc])



