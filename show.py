from sklearn.metrics import accuracy_score 
import numpy as np
import pickle
from keras.models import Model
from keras.models import load_model
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from PIL import Image as im
def load_data():
    file = open('car1.data', 'rb')
    (pixels, labels) = pickle.load(file)
    file.close()
    print(pixels.shape)
    print(labels.shape)
    return pixels, labels
X,y = load_data()  
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100)
my_model = load_model("vggmodel.h5")
my_model.load_weights("weights1-final1-05-0.75.hdf5")
y_hat = my_model.predict(X_test) #vector du doan dau ra
print(y_hat)
y_pred = np.argmax(y_hat, axis=1) #vector du doan dau ra voi moi phan tu la class du doan cua mot diem du lieu trong tap kiem thu duoi dang mang
print(y_pred)
y_test =  np.argmax(y_test, axis=1) #vector class that cua du lieu
print(y_test)
# Tính accuracy:
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
#Tính confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
compare= y_test==y_pred
print(compare)




