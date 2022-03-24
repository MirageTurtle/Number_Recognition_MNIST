import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# torchvision is for downloading dataset
import torchvision
from torchvision import datasets, transforms
# cv2 is for displaying image
import cv2

# build Neural Networks


class LeNet(nn.Module):
    """
    卷积层 torch.nn.Conv2d
    激活层 torch.nn.ReLU
    池化层 torch.nn.MaxPool2d
    全连接层 torch.nn.Linear
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84),
                                 nn.BatchNorm1d(84),
                                 nn.ReLU(),
                                 nn.Linear(84, 10))  # 10 is the size of 0~9

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# download and load dataset
"""
root: dataset path
train: True for train, False for test
transform: the operation for data after import dataset
download: auto download
"""
train_dataset = datasets.MNIST(root="./num/",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)
test_dataset = datasets.MNIST(root="./num/",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)

"""
dataset: dataset
batch_size: size of dataset
"""
# The data is sorted randomly and packed during the data loading process
batch_size = 64
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

# # preview data
# images, labels = next(iter(train_loader))
# img = torchvision.utils.make_grid(images)
# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
# print(labels)
# cv2.imshow("win", img)
# key_pressed = cv2.waitKey(0)

# train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device: {}".format(device))
LR = 0.001  # ?what's LR
net = LeNet().to(device)
# 损失函数使用交叉商
criterion = nn.CrossEntropyLoss()
# 优化函数使用 Adam 自适应优化算法
optimizer = optim.Adam(net.parameters(), lr=LR)

epoch = 1

for epoch in range(epoch):
    # sum_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()  # 将梯度归零
        outputs = net(inputs)  # 将数据传入网络进行前向运算
        loss = criterion(outputs, labels)  # 得到损失函数
        loss.backward()
        optimizer.step()
        # print("loss: {}".format(loss))
        # sum_loss += loss.item()
        # if (i + 1) % 100 == 0:
        #     print("[%d, %d] loss:%.3f" %
        #             (epoch + 1, i + 1, sum_loss / 100))
        #     sum_loss = 0

path = "model_with_state_dict.pt"
# save model
torch.save(net.state_dict(), path)
# # load
# net = LeNet().to(device)
# net.load_state_dict(torch.load(path))
# net.eval()

# save as TorchScript
model_scripted = torch.jit.script_if_tracing(net)  # export to TorchScript
# !this command has bug
model_scripted.save("model_scripted.pt")  # save

net.eval()  # 将模型变换为测试模式
correct = 0
total = 0
for data_test in test_loader:
    images, labels = data_test
    images, labels = Variable(images), Variable(labels)
    output_test = net(images)
    _, predicted = torch.max(output_test, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("correct: {}".format(correct))
print("Test acc: {}".format(correct.item() / len(test_dataset)))
