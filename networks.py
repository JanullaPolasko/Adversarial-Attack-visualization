#this file contains modules archutectures and used mapping
import torch
from torch import nn
import torchvision


#ALL  USED COMBINATION HERE
def get_dataset_mapping():
    #dataset_name, type,  model_class, num_classes, pretreined model
    dataset_mapping = (
        ('CIFAR10', 'CONV', CIFAR_10, 10, False),
        ('SVHN', 'CONV', CIFAR_10, 10, False),
        ('MNIST','CONV' , MNIST_conv, 10, False),
        ('MNIST', 'FC' , MNIST_FC, 10, False),
        ('FMNIST', 'CONV' ,MNIST_conv, 10, False),
        ('CIFAR10', 'RESNET' ,CIFAR10_ResNet, 10, True),
        ('MNIST', 'RESNET' ,MNIST_ResNet, 10, True),
        ('MNIST', 'VIT' ,MNIST_ViT, 10, True),
        ('CIFAR10', 'VIT' ,CIFAR10_ViT, 10, True ),
    )
    return dataset_mapping


class MNIST_FC(nn.Module):
    """
    """
    def __init__(self, loss):
        super(MNIST_FC, self).__init__()
        self.loss = loss
        self.activation = torch.Tensor()
        self.classifier = nn.Sequential(nn.Linear(784, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 10),
                                        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MNIST_conv(nn.Module):
    def __init__(self, loss):
        super(MNIST_conv, self).__init__()
        self.loss = loss
        self.activation = torch.Tensor()
        self.classifier = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5),  # 2 or 32 channels
                                        nn.MaxPool2d(2),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 16, kernel_size=5),  # 4 or 64 channels
                                        nn.MaxPool2d(2),
                                        nn.Flatten(),
                                        nn.ReLU(),
                                        nn.Linear(256, 10),  # 64 or 1024 input neurons
                                        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class CIFAR_10(nn.Module):
    """
    """
    def __init__(self, loss):
        super(CIFAR_10, self).__init__()
        self.loss = loss
        self.activation = torch.Tensor()
        self.classifier = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.2),

                                        nn.Conv2d(32, 64, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.3),

                                        nn.Conv2d(64, 128, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.4),

                                        nn.Flatten(),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),

                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 10)
                                        )

    def forward(self, x):
        x = self.classifier(x)
        return x



class CIFAR10_ResNet(nn.Module):
    def __init__(self):
        super(CIFAR10_ResNet, self).__init__()

        pretrained_model = torchvision.models.resnet18(pretrained=True)
        self.loss = nn.CrossEntropyLoss()
        self.up = nn.Upsample((160, 160))
        self.pad = nn.ZeroPad2d(32)
        self.pretrained = pretrained_model

        num_ftrs = pretrained_model.fc.in_features
        self.pretrained.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        x = self.pretrained(x)
        return x

class MNIST_ResNet(nn.Module):
    def __init__(self):
        super(MNIST_ResNet, self).__init__()
        pretrained_model = torchvision.models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept 1-channel input (grayscale)
        pretrained_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.loss = nn.CrossEntropyLoss()
        self.up = nn.Upsample((160, 160))
        self.pad = nn.ZeroPad2d(32)
        self.pretrained = pretrained_model

        num_ftrs = pretrained_model.fc.in_features
        self.pretrained.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        x = self.pretrained(x)
        return x


class CIFAR10_ViT(nn.Module):
    def __init__(self):
        super(CIFAR10_ViT, self).__init__()
        
        # Load pretrained ViT-B/16
        pretrained_model = torchvision.models.vit_b_16(pretrained=True)
        
        # Define loss, upsampling, and padding
        self.loss = nn.CrossEntropyLoss()
        self.up = nn.Upsample(size=(160, 160), mode='bilinear')  # From 32x32 to 160x160
        self.pad = nn.ZeroPad2d(32)  # 160 + 2*32 = 224
        
        # Modify classifier head
        num_ftrs = pretrained_model.heads[0].in_features
        pretrained_model.heads[0] = nn.Linear(num_ftrs, 10)
        
        self.pretrained = pretrained_model

    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        x = self.pretrained(x)
        return x

class MNIST_ViT(nn.Module):
    def __init__(self):
        super(MNIST_ViT, self).__init__()
        
        # Load pretrained ViT-B/16
        pretrained_model = torchvision.models.vit_b_16(pretrained=True)
        
        # Modify first convolutional layer for 1-channel input
        pretrained_model.conv_proj = nn.Conv2d(
            in_channels=1,
            out_channels=pretrained_model.hidden_dim,
            kernel_size=pretrained_model.patch_size,
            stride=pretrained_model.patch_size,
        )
        
        # Define loss, upsampling, and padding
        self.loss = nn.CrossEntropyLoss()
        self.up = nn.Upsample(size=(160, 160), mode='bilinear')  # From 28x28 to 160x160
        self.pad = nn.ZeroPad2d(32)  # 160 + 2*32 = 224
        
        # Modify classifier head
        num_ftrs = pretrained_model.heads[0].in_features
        pretrained_model.heads[0] = nn.Linear(num_ftrs, 10)
        
        self.pretrained = pretrained_model

    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        x = self.pretrained(x)
        return x