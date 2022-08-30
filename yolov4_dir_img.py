import cv2
from matplotlib.pyplot import box
import numpy as np
import os
import shutil

#讀取模型與訓練權重
def initNet():
    CONFIG = 'F:\\workspace2\\ncku\\crypt\\yolo_weight\\yolov4-custom3087.cfg'
    WEIGHT = 'F:\\workspace2\\ncku\\crypt\\yolo_weight\\yolov4-custom_best3080.weights'
    # WEIGHT = './train_finished/yolov4-tiny-myobj_last.weights'
    net = cv2.dnn.readNet(CONFIG, WEIGHT)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255.0)
    model.setInputSwapRB(True)
    # print(model)
    return model

#物件偵測
def nnProcess(image, model):
    classes, confs, boxes = model.detect(image, 0.1, 0.1)
    return classes, confs, boxes

#框選偵測到的物件，並裁減
def drawBox(image, classes, confs, boxes):
    new_image = image.copy()
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        if x - 18 < 0:
            x = 18
        if y - 18 < 0:
            y = 18
        cv2.rectangle(new_image, (x - 18, y - 18), (x + w + 20, y + h + 24), (0, 255, 0), 3)
    return new_image

# 裁減圖片
def cut_img(image, classes, confs, boxes):
    cut_img_list = []
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        if x - 31 < 0:
            x = 31
        if y - 40 < 0:
            y = 41
        #cut_img = image[y - 30:y + h + 30, x - 18:x + w + 25]
        cut_img = image[y - 120: y + h + 90, x - 10: x + w + 10]
        cut_img_list.append(cut_img)
    return cut_img_list[0]

# 儲存已完成前處理之圖檔(中文路徑)
def saveClassify(image, output):
    cv2.imencode(ext='.tiff', img=image)[1].tofile(output)

if __name__ == '__main__':
    source = 'G:\\nuck\\ncku\\data_2_01\\'
    savepath01='G:\\nuck\\ncku\\data_2_detected\\'
    # source = './public_training_data/public_testing_data/'
    files = os.listdir(source)
    print('※ 資料夾共有 {} 張圖檔'.format(len(files)))
    print('※ 開始執行YOLOV4物件偵測...')
    model = initNet()
    success = fail = uptwo = 0
    number = 1
    crypt_num = 0
    for file in files:
        print(' ▼ 第{}張'.format(number)," 名稱: {}".format(file))
        #img = cv2.imdecode(np.fromfile(source+file, dtype=np.uint8), -1)
        #classes, confs, boxes = nnProcess(img, model)
        try :
            img = cv2.imdecode(np.fromfile(source+file, dtype=np.uint8), -1)
            #print(img)
            #img = cv2.resize(img,(700,700))   ### resize
            classes, confs, boxes = nnProcess(img, model)
            #print(classes)
            # print(len(boxes))
            if len(boxes) == 0:
                # 儲存原始除檔照片
                # saveClassify(img, './public_training_data/YOLOV4_pre/fail/' + file)
                saveClassify(img, savepath01 + file)
                fail += 1
                print('  物件偵測失敗：{}'.format(file))
                # cv2.imshow('img', img)
            # elif len(boxes) >= 2:
                # print('  物件偵測超過2個')
                # box_img = drawBox(img, classes, confs, boxes)
                # print(classes[0][0])
                # if classes[0][0] == 0:
                    # cut = cut_img(img, classes, confs, boxes)
                    # saveClassify(cut, 'E:\\workspace\\project_\\smoke_0407_img\\yolo_40\\' + file)
                    # print( "下載成功" )
                    # success += 1
                    # uptwo += 1
            else:
                # 框選後圖檔
                frame = drawBox(img, classes, confs, boxes)
                # 裁剪後圖檔
                if classes[0][0]== 0:
                    # cut = cut_img(img, classes, confs, boxes)
                    saveClassify(frame, savepath01 + file)
                    #print( "SAVE" )
                    #cut = cut_img(img, classes, confs, boxes)
                    # 儲存裁剪後圖檔
                    #saveClassify(cut, 'E:\\workspace\\project_\\smoke_data_test\\1_28_6_55_cut\\' + file)
                    print('下載成功')
                    success += 1
                    print('  物件偵測成功：{}'.format(file))
                    crypt_num +=len(boxes)
                    
                    #cv2.imshow('img', frame)
                    #cv2.imshow('cut', cut)
                # if classes[0][0]== 1:
                    # cut = cut_img(img, classes, confs, boxes)
                    # saveClassify(cut, 'E:\\workspace\\project_\\smoke_0407_img\\yolo_40\\' + file)
                    # print('下載成功')
                    # print('  物件偵測成功：{}'.format(file))
                    # success += 1
                # print('=' * 60)
                # cv2.waitKey()
            number += 1
        except:
            pass
    print('※ 程式執行完畢')
    print('※ 總計：成功 {} 張、失敗 {} 張'.format(success, fail))
    print('※ 總計：crypt 個數: {}'.format(crypt_num))
    #print('※ 物件超過兩個物件組 {} 張'.format(uptwo))
        
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
