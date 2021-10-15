import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

path = 'train'
def save_data(path):
    pixels = []
    labels = []
    for image in os.listdir(path):
        linkImage = path + '/' + image
        print('processing...', image)
        pixels.append(cv2.resize(cv2.imread(linkImage), dsize=(128, 128)))
        category = image.split('_')[0]
        labels.append(category)
    pixels = np.array(pixels)
    labels = np.array(labels)
    print(pixels)
    print(labels)
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    #Phù hợp với bộ mã hóa nhãn và trả về các nhãn được mã hóa.
    print('one-hot')
    print(labels)
    print(labels.shape) 
    file = open('car.data', 'wb')
    pickle.dump((pixels, labels), file)
    file.close()
    print(file)
save_data(path)
