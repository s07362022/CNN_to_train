import os
import sys
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
##set input image size
input_shape = (416,416,3)
IMAGE_SIZE = 416
#resize
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #get size
    h, w, _ = image.shape
    
    #adj(w,h)
    longest_edge = max(h, w)    
    
    #size = n*n 
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 

    BLACK = [0, 0, 0]   
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    return cv2.resize(constant, (height, width))

#load_data
A = "D:\\dataset\\onlyimg_10281103\\stand_img\\"
B = "D:\\dataset\\onlyimg_10281103\\img_fall\\"


test_final = "D:\\dataset\\onlyimg_10281103\\test\\"

images = []
labels = []
dir_counts = 0
def a (A=A,images=images,labels=labels):
    for i in os.listdir(A):
        img1 = cv2.imread(A+"/"+i)
        #img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
        img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
        images.append(img1)
        labels.append(dir_counts)
        if len(labels) == 600:
            break    
    print("A already read")
    return(images,labels)

def b (B=B,images=images,labels=labels):
    BC = 0
    for i in os.listdir(B):
        img2 = cv2.imread(B+"/"+i)
        #img2 = cv2.resize(img2,(IMAGE_SIZE,IMAGE_SIZE))
        img2 = resize_image(img2, IMAGE_SIZE, IMAGE_SIZE)
        images.append(img2)
        labels.append(dir_counts+1)
        BC = BC+1
        if BC == 600:
            break
    print("B already read")
    return(images,labels)


a(A,images,labels)
b(B,images,labels)
###################

####################

label = np.array(labels)
#########Train/Test############
X_train_img,X_test_img,y_train_label,y_test_label =  train_test_split(images, label,test_size=0.4,random_state=42 )#
X_train = np.array(X_train_img, dtype=np.float32)
X_test = np.array(X_test_img, dtype=np.float32)
#print("X_train.shape",X_train.shape)
x_train_std = X_train/255.0
x_test_std  =  X_test/255.0
y_trainOneHot = np_utils.to_categorical(y_train_label)
y_testOneHot = np_utils.to_categorical(y_test_label)
print("x_train_std.shape",x_train_std.shape)
print("y_train_label",y_train_label.shape)
################################

################################
test_final1 = []
####test#######
for i in os.listdir(test_final):
    img6 = cv2.imread(jw_face+"/"+i)
    img6 = resize_image(img6, IMAGE_SIZE, IMAGE_SIZE)
    test_final1.append(img6)
# Set the augmentation parameters and fit the training data
############### change here #################
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    fill_mode="constant",
    cval=0,
    horizontal_flip=True,
    zca_whitening=False,
    brightness_range=[0.5,1.5]
)#brightness_range=[0.5,1.5]
############### change here #################
datagen.fit(x_train_std)



#Model
def mobile_model():
    from tensorflow.keras.applications import MobileNet

    base_model = MobileNet(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(26, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def resnet_50_model():
    from tensorflow.keras.applications import ResNet50
    base_model = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def model_by_self(input_shape=input_shape):
    from keras.layers.advanced_activations import PReLU
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(7, 7), padding='same', input_shape=input_shape))
    #model.add(LeakyReLU(alpha=.001))   # add an advanced activation
    model.add(PReLU())
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=input_shape))
    #model.add(LeakyReLU(alpha=.001))   # add an advanced activation
    model.add(PReLU())
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5)))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dropout(rate=0.25))   
    model.add(Dense(1024, activation='relu'))
    #model.add(LeakyReLU(alpha=.001))   # add an advanced activation    
    #model.add(PReLU())
    model.add(Dropout(rate=0.35))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.35))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.25))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(rate=0.25))
    model.add(Dense(2, activation='softmax'))
    return model

# Compile the model
#model=mobile_model()
#model=resnet_50_model()
model = model_by_self(input_shape=input_shape)

'''
model.compile(
      optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
      loss="binary_crossentropy",
      metrics=["accuracy"],
)
'''
from keras import optimizers
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
#summary
#model.summary()

# Set the epochs and batch size, then train the model
############### change here #################
epochs = 20
batch_size = 5
steps_per_epoch = int(len(x_train_std)*2/batch_size)
samples_per_epoch=(len(x_train_std)*2)
############### change here #################

history = model.fit_generator(
    datagen.flow(x_train_std, y_trainOneHot, batch_size=batch_size),
    epochs=epochs,
    validation_data=(x_test_std, y_testOneHot),
    steps_per_epoch=steps_per_epoch,
    
)#fit_generator
print(samples_per_epoch)

#evn
scores = model.evaluate(x_test_std,y_testOneHot,verbose=1)
print("scores:",scores[1])

# Plot loss and accuracy
def plt_loss(history):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
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


#############################Predict
#predict_y = model.predict(x_test_std)
prediction= model.predict_classes(test_final)# x_test_std


#confusion_matrix
'''
def cof_matr_premodel(predict_y=predict_y):
    predict_y[predict_y >= 0.5] = 1
    predict_y[predict_y < 0.5] = 0
    print(confusion_matrix(y_testOneHot.argmax(axis=1), predict_y.argmax(axis=1), labels=[1, 0]))

    y=y_testOneHot.argmax(axis=1)
    p_y=predict_y.argmax(axis=1)
    # Calculate the sensitivity and specificity
    TP = confusion_matrix(y, p_y, labels=[1, 0])[0, 0]
    FP = confusion_matrix(y, p_y, labels=[1, 0])[1, 0]
    FN = confusion_matrix(y, p_y, labels=[1, 0])[0, 1]
    TN = confusion_matrix(y, p_y, labels=[1, 0])[1, 1]
    print("True positive: {}".format(TP))
    print("False positive: {}".format(FP))
    print("False negative: {}".format(FN))
    print("True negative: {}".format(TN))
    ############################
    sensitivity = TP/(FN+TP)
    specificity = TN/(TN+FP)
    ################################
    print("Sensitivity: {}".format(sensitivity))
    print("Specificity: {}".format(specificity))
'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#print(metrics.classification_report(y_test_label, prediction))
try:
    import itertools
    import sklearn.metrics as metrics
    cnf_matrix = metrics.confusion_matrix(y_test_label, prediction)#y_test_label
    target_names = ['stall', 'fall']
    #plot_confusion_matrix(predict_y)
except:
    pass
#draw label vs predict
def plot_image(image,labels,prediction,idx,num=26):  
    fig = plt.gcf() 
    fig.set_size_inches(20, 24) 
    if num>10: 
        num=10 
    for i in range(0, num): 
        ax = plt.subplot(2,5, 1+i) 
        ax.imshow(image[idx], cmap='binary') 
        title = "label=" +str(labels[idx]) 
        if len(prediction)>0: 
            title+=",perdict="+str(prediction[idx]) 
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([]) 
        idx+=1 
    plt.show() 
#plot_image(x_test_std,y_test_label,predict_y.argmax(axis=1),idx=10)
plot_image(x_test_std,y_test_label,prediction,idx=10)

# Plot the ROC curve of the test results
def plt_auc(y_test_label,predict_y):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')

    fpr, tpr, _ = roc_curve(y_test_label, predict_y)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = {}'.format(roc_auc))

    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
try:
    plt_auc(y,predict_y.argmax(axis=1))
except:
    pass

try:
    target_names = ['stall', 'fall']
    plot_confusion_matrix(cnf_matrix, classes=target_names)
   # plot_image(x_test_std,y_test_label,prediction,idx=10)
except:
    pass
