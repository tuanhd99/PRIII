from threading import main_thread
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
import cv2
import os
def load_data():
    file = open('car.data', 'rb')
    (pixels, labels) = pickle.load(file)
    file.close()
    print(pixels.shape)
    print(labels.shape)
    return pixels, labels
X,y = load_data()  
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100)
my_model = load_model("vggmodel.h5")
my_model.load_weights("weights-final-end-epoch-42-ac-0.99.hdf5")
y_hat = my_model.predict(X_test) #vector du doan dau ra
#print(y_hat)
y_pred = np.argmax(y_hat, axis=1) #vector du doan dau ra voi moi phan tu la class du doan cua mot diem du lieu trong tap kiem thu duoi dang mang
print(y_pred)
y_test_label =  np.argmax(y_test, axis=1) #vector class that cua du lieu
print(y_test_label)
# Tính accuracy:
accuracy = accuracy_score(y_test_label, y_pred)
print('Accuracy: %f '% accuracy)
# Tính confusion matrix
matrix = confusion_matrix(y_test_label, y_pred)
print(matrix)
'''
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(matrix, index = [i for i in "ABCDEFGHIJK"],
                  columns = [i for i in "ABCDEFGHIJK"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
'''
#so sanh

compare = y_test_label==y_pred
print(y_test_label.shape)
print(y_pred.shape)
print(compare.shape)
if not os.path.exists('imger'):
    os.mkdir('imger')
for i, n in enumerate(compare):
    
    if str(n) == 'False':
        print(i)
        px = X_test[i]# đây là ảnh sai
        cv2.imwrite('imger/'+str(i)+'.jpg', px)
        cv2.waitKey(1)


        


