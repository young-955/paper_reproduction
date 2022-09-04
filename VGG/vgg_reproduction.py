import torch
import torch.nn as nn

def conv_bn_relu(input_num, output_num, kernel_size = 3, padding = 1):
    return nn.Sequential(
        nn.Conv2d(input_num, output_num, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(output_num),
        nn.ReLU(inplace=True)
    )

class VGG(nn.Module):
    def __init__(self, input_size = (224, 224), output_size = 1000) -> None:
        super(VGG).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, output_size)
        self.conv1_bn_relu = conv_bn_relu(3, 64)
        self.conv2_bn_relu = conv_bn_relu(64, 128)
        self.conv3_bn_relu = conv_bn_relu(128, 256)
        self.conv4_bn_relu = conv_bn_relu(256, 256)
        self.conv5_bn_relu = conv_bn_relu(256, 512)
        self.conv6_bn_relu = conv_bn_relu(512, 512)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()

    def forward_A(self, x):
        x = self.conv1_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv3_bn_relu(x)
        x = self.relu(x)
        x = self.conv4_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv5_bn_relu(x)
        x = self.relu(x)
        x = self.conv6_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv6_bn_relu(x)
        x = self.relu(x)
        x = self.conv6_bn_relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        


