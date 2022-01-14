from tkinter.tix import Y_REGION
import pandas as pd
import numpy as np
#load data
#train
train_data = pd.read_csv("tr_data.csv") 
#train_data =train_data[~np.isnan(train_data)]
train_data = train_data.drop(index=0)
train_data = train_data.values.reshape(10751, 120,1,1)
print("train_data_shape:",train_data.shape)
#print("len to train_data",len(train_data))
X_train =train_data
#test
test_data = pd.read_csv("tst_data.csv") 
test_data = test_data.drop(index=0)
test_data = test_data.values.reshape(2575, 120,1,1)
#test_data =test_data[~np.isnan(test_data)]
print("test_data_shape:",test_data.shape)
#print("len to test_data",len(test_data))
X_test =test_data

#train_label
train_lab = pd.read_csv("tr_label.csv")
train_lab = train_lab.values.reshape(-1,1)
print(train_lab.shape)

#test_label
test_lab = pd.read_csv("tst_label.csv")
test_lab = test_lab.values.reshape(-1,1)
print(test_lab.shape)

############ shuffle ###########
index1= [i for i in range(len(X_train))]
np.random.shuffle(index1)
X_train=X_train[index1]
train_lab = train_lab[index1]

index2= [i for i in range(len(X_test))]
np.random.shuffle(index2)
X_test=X_test[index2]
test_lab =test_lab[index2]
####################################
##############std###################
X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)
####################
#load tensorflow/keras 
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sklearn.metrics as metrics
#y_trainOneHot= pd.get_dummies(train_lab)
#y_testOneHot= pd.get_dummies(test_lab)
y_trainOneHot = np_utils.to_categorical(train_lab)
y_testOneHot = np_utils.to_categorical(test_lab)
#print(y_testOneHot[0].shape)
input_shape = (120,1,1)

def model_by_self(input_shape=input_shape):
    from tensorflow.keras.layers import PReLU,LeakyReLU
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D,BatchNormalization
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', input_shape=(120,1,1),activation = 'relu'))
    #model.add(LeakyReLU(alpha=.001))   # add an advanced activation
    model.add(BatchNormalization())
    #model.add(PReLU())
    #model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same',activation = 'relu'))
    model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=.001))   # add an advanced activation
    #model.add(PReLU())
    #model.add(Dropout(rate=0.2))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',activation = 'relu'))
    model.add(BatchNormalization())
    #model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    #model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))   
    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU(alpha=.001))   # add an advanced activation    
    #model.add(PReLU())
    model.add(Dropout(rate=0.25))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dropout(rate=0.2))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(rate=0.25))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(rate=0.25))
    model.add(Dense(15, activation='softmax'))
    return model

model = model_by_self(input_shape=input_shape)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy']) 
#summary
model.summary()

epochs = 30
batch_size = 64
steps_per_epoch = int(len(X_train)*2/batch_size)
samples_per_epoch=(len(X_train)*2)
##########Tensorboard#####
#from keras.callbacks import TensorBoard
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#import tensorflow as tf
#import datetime
#log_dir = "logs" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=True, write_images=True)
############################
history = model.fit(x=X_train, y=y_trainOneHot, batch_size=8, epochs=100, verbose=1, validation_split=0.2)
scores = model.evaluate(X_test,y_testOneHot,verbose=1)
print("scores:",scores[1])
prediction= model.predict_classes(X_test)
cnf_matrix = metrics.confusion_matrix(test_lab, prediction)

############acc / loss###########
def plt_loss(history):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left') 
    #plt.show()
    # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left') 
    plt.show()
plt_loss(history)
