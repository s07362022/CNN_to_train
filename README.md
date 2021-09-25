# CNN_to_train
##resize the photo

if hight and weight are least then what we set shape, we add black ground to over (set_xshape - img_xshape).

![resize_image](https://ibb.co/74LHnz7)

```python
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
```


###Use CNN (MobileNet)to train photo

we use pretrain model (MobileNet) to effetively train our model.

but, the net layer may over our expected.

we have to try more and more times.(model by self) 

```python
#Model
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
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
```


####Plot the ROC curve of the test results

we have to know if the model could efficient by search.
so, we use ROC curve to look easily.

![ROC curve](https://ibb.co/DR6rPJ5)

```python
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
plt_auc(y,predict_y.argmax(axis=1))
```



#####Calculate the sensitivity and specificity

the the sensitivity and specificity are very important.

```python
sensitivity = TP/(FN+TP)
specificity = TN/(TN+FP)
```

######Plot confusion_matrix

![confusion matrix](https://ibb.co/hBDZgX5)
