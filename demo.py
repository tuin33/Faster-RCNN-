import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
import torch.optim as optim
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

# 定义自己的数据集
class MyDataset(Dataset):
    def __init__(self, txt_file, transform=None, target_size=(100, 100)):
        self.data = []
        self.transform = transform
        self.target_size = target_size
        self._classes = ('__background__',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line=line.split()[0]
                image_path = os.path.join('VOC2007/JPEGImages/',line+'.jpg')
                self.data.append((image_path, int(line)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class CNN_block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN_block1, self).__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.relu=nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out
    
class CNN_block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN_block2, self).__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv3=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.relu=nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        return out

# 定义Conv layers提取特征图 总共13层
class feature_map_CNN(nn.Module):
    def __init__(self):
        super(feature_map_CNN, self).__init__()
        self.block1=CNN_block1(3,64)
        self.block2=CNN_block1(64,128)
        self.block3=CNN_block2(128,256)
        self.block4=CNN_block2(256,512)
        self.block5=CNN_block2(512,512)
        self.pool=nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
      
    def forward(self, x):
        out=self.block1(x)
        out=self.pool(out)
        out=self.block2(out)
        out=self.pool(out)
        out=self.block3(out)
        out=self.pool(out)
        out=self.block4(out)
        out=self.pool(out)
        out=self.block5(out)
        return out

# 定义RPN(Region Proposal Networks)
# class rpn(nn.Module):
#     def __init__(self):
#         super(RPN_Net, self).__init__()
#         self.conv1=nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
#         # 对 Anchor box 内图像信息做二分类工作
#         self.conv_rpn_cls=nn.Conv2d(512, 18, kernel_size=1, padding=0, stride=1)
#         # 得到 Anchor box 四个坐标信息（偏移量）
#         self.conv_rpn_bbox=nn.Conv2d(512, 36, kernel_size=1, padding=0, stride=1)

#     # 为特征图60*40上的每个像素生成9个Anchor box，并且对生成的Anchor box进行过滤和标记
    
#     #   bottom: 'rpn_cls_score'   #仅提供特征图的height和width的参数大小
#     #   bottom: 'gt_boxes'        #ground truth box
#     #   bottom: 'im_info'         #包含图片大小和缩放比例，可供过滤anchor box
#     #   bottom: 'data'  
#     #   top: 'rpn_labels'  
#     #   top: 'rpn_bbox_targets'  
#     #   top: 'rpn_bbox_inside_weights'  
#     #   top: 'rpn_bbox_outside_weights' 
#     def anchor_target_layer(feat_stride=16,scales=[8, 16, 32]):


target_size=(600,1000) #宽600 长1000    
train_path='VOC2007/ImageSets/Main/mytrain.txt'
transform = transforms.Compose([
    transforms.Resize(target_size),  # 缩放图像到目标大小
    transforms.ToTensor()  # 将图像转换为张量
])

# 创建自定义数据集
# data_train = MyDataset(txt_file=train_path, transform=transform)
# data_loader = DataLoader(data_train, batch_size=16, shuffle=True)

batch_size=16
imdb, roidb, ratio_list, ratio_index = combined_roidb('VOC2007')
dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                        imdb.num_classes, datasets_name='VOC2007',training=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)

# 定义网络
model = feature_map_CNN()

# # 定义损失函数
# criterion = nn.CrossEntropyLoss()

# # 定义优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # 训练网络
# num_epochs = 10
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for images, labels in data_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         # plot
#         img = torchvision.utils.make_grid(images).numpy()
#         plt.imshow(np.transpose(img, (1, 2, 0)))
#         plt.pause(0.5)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader)}")
