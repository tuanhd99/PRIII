import os
import cv2
import numpy as np
import pickle
import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
def load_data():
    file = open('car1.data', 'rb')
    (pixels, labels) = pickle.load(file)
    file.close()
    print(pixels.shape)
    print(labels.shape)
    print('px is :')
    print(pixels)
    print('lb is :')
    print(labels)
    return pixels, labels
X,y = load_data()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100)
print(X_train.shape)
print(y_train.shape)
#dùng vgg16
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
# Dong bang cac layer
for layer in model_vgg16_conv.layers:
    layer.trainable = False
#tao model với input là ảnh,lấy output của VGG16 và làm input của các layer FC thêm vào
input = Input(shape=(128, 128, 3), name='image_input')
output_vgg16_conv = model_vgg16_conv(input)

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(11, activation='softmax', name='predictions')(x)

my_model = Model(inputs=input, outputs=x)
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
vggmodel = my_model                      
#checkpoint
filepath="weights-final-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
aug = ImageDataGenerator(
    rescale=1./255, #chuyen doi ti le,vì 255 là giá trị pixel tối đa. rescale 1./255 là chuyển đổi mọi giá trị pixel từ phạm vi [0,255] -> [0,1]
	width_shift_range=0.1,
    height_shift_range=0.1,
	horizontal_flip=True,
    brightness_range=[0.2,1.5], fill_mode="nearest")
    # chuyen trai,phai,do sang,lat ngang

aug_val = ImageDataGenerator(rescale=1./255)

vgghist=vggmodel.fit_generator(aug.flow(X_train, y_train, batch_size=64),
                               epochs=50,steps_per_epoch=len(X_train)//64,
                               validation_data=aug_val.flow(X_test,y_test,
                               batch_size=64), shuffle=True,
                               callbacks=callbacks_list) 
vggmodel.save("vggmodel.h5")


