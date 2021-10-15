from numpy import loadtxt
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
dataset = loadtxt('data.csv',delimiter=',')
print(dataset)
X = dataset[:,0:8]
y = dataset[:,8]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,test_size=0.2)
X_train, X_val ,y_train, y_val = train_test_split(X_train_val,y_train_val,test_size=0.2)
model = Sequential()
model.add(Dense(16,input_dim= 8,activation = 'relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100,batch_size=8,validation_data=(X_val,y_val))
