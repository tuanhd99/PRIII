import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.models import load_model

cap = cv2.VideoCapture('Aircraft.mp4')

class_name = ['airplane','bike', 'bus', 'car', 'cargo', 'container', 'cyclo', 'excavator', 'motorcycle', 'sailboat', 'truck'] 
#load model đã train 
my_model = load_model("vggmodel.h5")
my_model.load_weights("weights-final-end-epoch-42-ac-0.99.hdf5")

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    image = frame.copy()
    imageshow = cv2.resize(image, dsize=(500,500))
 
    image = cv2.resize(image, dsize=(128, 128))

    image = image.astype('float')*1./255
    
    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    print(predict)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0],axis=0))
    if (np.max(predict) >= 0.85):
        # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (100, 100)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(imageshow, class_name[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('predict', imageshow)

    if cv2.waitKey(1) == ord('q'):
        break