import numpy as np 
import cv2 
import os
'''
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

images = []
labels = []

img1 = "D:\\dataset\\yolo4\\face_dataset\\face_img"
lab1 = "0"
def img_load(img=img1,images=images,labels=labels):
    m=0
    for i in os.listdir(img):
        m+=1
        if int(m) >= 1000:
            break
        img2 = cv2.imread(img1+"/"+i)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        images.append(img2)
        labels.append("0")
    return images , labels

img_load(img1,images,labels)

images = np.array(images)
labels = np.array(labels)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0,
    zoom_range=0,
    fill_mode="constant",
    cval=0,
    horizontal_flip=True,
    zca_whitening=False,
    brightness_range=[0.5,1.5],
)

count = 0 
for batch in datagen.flow(images,batch_size=10,save_to_dir="D:\\dataset\\yolo4\\face_dataset\\test1\\",save_prefix='linear',save_format="jpg"):
    count+=1
    if count >100:
        break
'''
images5=[]
img5 = "D:\\dataset\\yolo4\\face_dataset\\test1"
img6 = "D:\\dataset\\yolo4\\face_dataset\\test2\\"
#lab1 = "0"
def img_load(img=img5,images=images5):
    m=0
    for i in os.listdir(img):
        m+=1
        if int(m) >= 1000:
            print("already 1000 imgs")
            break
        img2 = cv2.imread(img+"/"+i)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        images5.append(img2)
        #labels.append("0")
    return images 
img_load(img5,images5)

#images2 = []
#def sharpen(img, sigma=100):    
    # sigma = 5、15、25
    #blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    #usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    #cv2.imshow("t",usm)
    #cv2.imshow("o",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #return usm



def ke(image):
    kernel = np.array([[0,-1,0],[0,3,0],[0,-1,0]],np.float32)
    dst = cv2.filter2D(image,-1,kernel=kernel)
    return dst


for k in range (len(images5)):
    new = ke(image=images5[k])
    s = str(k)
    cv2.imwrite(img6+s+".jpg",new)
