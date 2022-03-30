import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module
from torch.optim import Adam
import pandas as pd
import os
from os import listdir
from tqdm.notebook import tqdm
from PIL import Image


PATH_train="D:\\workspace\\stanford_dogs_dataset\\train\\"
TRAIN =Path(PATH_train)
#Batch：每批丟入多少張圖片
batch_size = 4
#Learning Rate：學習率
LR = 0.003
transforms = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()]) #224,224

train_data = datasets.ImageFolder(TRAIN, transform=transforms)
print(train_data.class_to_idx)
#切分訓練集、驗證集
train_size = int(0.9 * len(train_data))
valid_size = len(train_data) - train_size
train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])
#Dataloader可以用Batch的方式訓練
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,shuffle=True)
#print(type(train_loader))

EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 8
#LR = 0.003              # learning rate
DOWNLOAD_MNIST = False

#model = models.densenet161(num_classes=2)
#model =  models.resnet18()
#model = models.vgg16(num_classes=2)#.resnet50()vgg16
#fc_features = model.fc.in_features
#修改类别为9
#model.fc = nn.Linear(2048, 2)
#cls_fc = nn.Linear(4096, 31)
 


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0) #output_shape=(16,220,220) #(224-5+1)/1 #(weigh-kernel+1)/stride 無條件進位
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,110,110) #(220/2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) #output_shape=(32,106,106)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,53,53)
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0) #output_shape=(16,51,51)
        self.relu3 = nn.ReLU() # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,25,25)
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0) #output_shape=(8,23,23)
        self.relu4 = nn.ReLU() # activation
        # Max pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) #output_shape=(8,11,11)
        # Fully connected 1 ,#input_shape=(8*12*12)
        self.fc1 = nn.Linear(8 * 11 * 11, 512) 
        self.relu5 = nn.ReLU() # activation
        #self.fc2 = nn.Linear(512, 8) 
        self.output = nn.Linear(512, 8) #nn.Softmax(dim=1)
        
    
    def forward(self, x):
        out = self.cnn1(x) # Convolution 1
        out = self.relu1(out)
        out = self.maxpool1(out)# Max pool 1
        out = self.cnn2(out) # Convolution 2
        out = self.relu2(out) 
        out = self.maxpool2(out) # Max pool 2
        out = self.cnn3(out) # Convolution 3
        out = self.relu3(out)
        out = self.maxpool3(out) # Max pool 3
        out = self.cnn4(out) # Convolution 4
        out = self.relu4(out)
        out = self.maxpool4(out) # Max pool 4
        out = out.view(out.size(0), -1) # last CNN faltten con. Linear NN
        out = self.fc1(out) # Linear function (readout)
        #out = self.fc2(out)
        out = self.output(out)

        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
#搭建網路的起手式,nn.module是所有網路的基類.
#我們開始定義一系列網路如下：  #train data ＝ (1,28,28)      
        self.conv1 = nn.Sequential(
            nn.Conv2d(           
            #convolution2D
                in_channels=3,  
                #input channel(EX:RGB)
                out_channels=16, 
                #output feature maps
                kernel_size=5,   
                #filter大小
                stride=1,        
                #每次convolution移動多少
                padding=2,       
                #在圖片旁邊補0                       
            ),
            nn.ReLU(), #activation function #(16,224,224) #(16,224,224)
            nn.MaxPool2d(kernel_size = 2), #(16,112,112) #(16,112,112)
        )
        #以上為一層conv + ReLu + maxpool
        
        #快速寫法：
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),  #(32,96,96)
            nn.ReLU(), 
            nn.MaxPool2d(2)   #(32,48,48)
        )
        
        self.out = nn.Linear(32*7*7, 8) #10=0~9
       
    def forward(self,x):
       x = self.conv1(x)
       x = self.conv2(x)
       x = x.view(x.size(0), -1)
       output = self.out(x)
       return output
       

#cnn = model
#cnn =CNN_Model()
cnn =CNN()
#print(cnn)  # net architecture

#optimizer = Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
#loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

def train(model,n_epochs,train_loader,valid_loader,optimizer,criterion):#,scheduler
    train_acc_his,valid_acc_his=[],[]
    train_losses_his,valid_losses_his=[],[]
    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss,valid_loss = 0.0,0.0
        train_losses,valid_losses=[],[]
        train_correct,val_correct,train_total,val_total=0,0,0,0
        train_pred,train_target=torch.zeros(8,1),torch.zeros(8,1)
        val_pred,val_target=torch.zeros(8,1),torch.zeros(8,1)
        count=0
        count2=0
        print('running epoch: {}'.format(epoch))
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in tqdm(train_loader):
            # move tensors to GPU if CUDA is available
            #if train_on_gpu: 
                #data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            #data, target = data.cpu(), target.cpu()
            output = model(data)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # calculate the batch loss
            loss = criterion(output, target)
            #calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            train_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())#.cpu()
            train_total += data.size(0)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            #lr_scheduler.step()
            # update training loss
            train_losses.append(loss.item()*data.size(0))
            
            if count==0:
                train_pred=pred
                train_target=target.data.view_as(pred)
                count=count+1
            else:
                train_pred=torch.cat((train_pred,pred), 0)
                train_target=torch.cat((train_target,target.data.view_as(pred)), 0)
        train_pred=train_pred.view(-1).cpu().numpy().tolist()#.cpu()
        train_target=train_target.view(-1).cpu().numpy().tolist()#.cpu()
######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in tqdm(valid_loader):
            # move tensors to GPU if CUDA is available
            #if train_on_gpu:
                #data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            #data, target = data.cuda(), target.cuda()
            output = model(data)
            # calculate the batch loss
            loss =criterion(output, target)
            #calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            val_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())#.cpu()
            val_total += data.size(0)
            valid_losses.append(loss.item()*data.size(0))
            if count2==0:
                val_pred=pred
                val_target=target.data.view_as(pred)
                count2=count+1
            else:
                val_pred=torch.cat((val_pred,pred), 0)
                val_target=torch.cat((val_target,target.data.view_as(pred)), 0)
        val_pred=val_pred.view(-1).cpu().numpy().tolist()#.cpu()
        val_target=val_target.view(-1).cpu().numpy().tolist()#.cpu()
        
        # calculate average losses
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        
        # calculate average accuracy
        train_acc=train_correct/train_total
        valid_acc=val_correct/val_total
        train_acc_his.append(train_acc)
        valid_acc_his.append(valid_acc)
        train_losses_his.append(train_loss)
        valid_losses_his.append(valid_loss)
# print training/validation statistics 
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            train_loss, valid_loss))
        print('\tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            train_acc, valid_acc))
    return train_acc_his,valid_acc_his,train_losses_his,valid_losses_his,model


#model1=CNN_Model()
model1 = cnn
n_epochs = 40
optimizer1 = torch.optim.Adam(model1.parameters() )#,lr=LR
#########
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=2)
#scheduler_state = lr_scheduler.state_dict()

##########


#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=2)#.ReduceLROnPlateau(optimizer1, 'max',verbose=1,patience=3)
criterion = CrossEntropyLoss()
#lr_scheduler.load_state_dict(scheduler_state)
train_on_gpu = True
train_acc_his,valid_acc_his,train_losses_his,valid_losses_his,model1=train(model1,n_epochs,train_loader,valid_loader,optimizer1,criterion)#,lr_scheduler
#scheduler.step(val_loss)




plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(train_losses_his, 'bo', label = 'training loss')
plt.plot(valid_losses_his, 'r', label = 'validation loss')
plt.title("Simple CNN Loss")
plt.legend(loc='upper left')
plt.subplot(222)
plt.plot(train_acc_his, 'bo', label = 'trainingaccuracy')
plt.plot(valid_acc_his, 'r', label = 'validation accuracy')
plt.title("Simple CNN Accuracy")
plt.legend(loc='upper left')
plt.show()
torch.save(model1, "./fall_stand/back")
#model1 = torch.load('..../Dogcat_resnet18')








'''
optimizer = Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
# training and testing
step =0
for epoch in range(EPOCH):
    for b_x, b_y in tqdm(train_loader):   # gives batch data, normalize x when iterate train_loader enumerate

        #output = cnn(b_x)[0]               # cnn output
        output = cnn(b_x)
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        step +=1

        if step % 50 == 0:
            test_output, last_layer = cnn()
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
'''

