# this file contains functions for loading datasets
import torch
from datapath import my_path
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def load_data(dataset, batch_size_train=64, batch_size_test=64):
    """
    This function loads a dataset (MNIST, FMNIST, CIFAR10, or SVHN) and prepares it for use in training and testing by applying necessary transformations and creating DataLoaders.

    Parameters:
    - dataset (str): The name of the dataset to load. Valid options are 'MNIST', 'FMNIST', 'CIFAR10', and 'SVHN'.
    - batch_size_train (int, optional): The batch size for the training set. Default is 64.
    - batch_size_test (int, optional): The batch size for the test set. Default is 64.

    Returns:
    - trainloader (torch.utils.data.DataLoader): The DataLoader for the training set.
    - testloader (torch.utils.data.DataLoader): The DataLoader for the test set.
    - input_shape (tuple): The shape of the input images.
    - classes (list): The list of class labels.
    """
    
    dataset_mapping = {
        'MNIST': (torchvision.datasets.MNIST, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], (1, 28, 28), transforms.Normalize((0.13066041469573975,), (0.30150410532951355,))),
        'FMNIST': (torchvision.datasets.FashionMNIST, ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat',
                                                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'], (1, 28, 28), transforms.Normalize(0.2860407531261444, 0.320453405380249)),
        'CIFAR10': (torchvision.datasets.CIFAR10, ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'), (3, 32, 32), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2006))),
        'SVHN': (torchvision.datasets.SVHN, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], (3, 32, 32),  transforms.Normalize((0.43768206238746643, 0.4437699019908905, 0.4728040099143982),(0.12008649855852127, 0.12313701957464218, 0.10520392656326294)))
    }
    
    if dataset not in dataset_mapping:
        raise ValueError(f"Unknown dataset {dataset}")

    dataset_class, classes, input_shape, transform_default = dataset_mapping[dataset]
    
    transform = transform=transforms.Compose([transforms.ToTensor(),transform_default] )
    
    if dataset == 'SVHN':
        trainset = dataset_class(root=my_path()+'/data', split='train', download=True, transform=transform)
        testset = dataset_class(root=my_path()+'/data', split='test', download=True, transform=transform)
    else:
        trainset = dataset_class(root=my_path()+'/data', train=True, download=True, transform=transform)
        testset = dataset_class(root=my_path()+'/data', train=False, download=True, transform=transform)


    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, testloader, input_shape, classes



def calculate_mean_std(dataset, split='train', batch_size=64):
    '''
    This function calculates the mean and standard deviation for the specified dataset to normalize the images.

    Parameters:
    - dataset (str): The name of the dataset to calculate mean and std for. Valid options are 'MNIST', 'FMNIST', 'CIFAR10', and 'SVHN'.
    - split (str, optional): The split of the dataset to use ('train' or 'test'). Default is 'train'.
    - batch_size (int, optional): The batch size for loading the dataset. Default is 64.

    Returns:
    - mean (torch.Tensor): The calculated mean values for the dataset.
    - std (torch.Tensor): The calculated standard deviation values for the dataset.
    '''

    dataset_mapping = {
        'MNIST': (torchvision.datasets.MNIST) ,
        'FMNIST': (torchvision.datasets.FashionMNIST),
        'CIFAR10': (torchvision.datasets.CIFAR10),
        'SVHN': (torchvision.datasets.SVHN)
        }
    dataset_class = dataset_mapping[dataset]
        
    """ Function for mean and std  for dataset class without saving"""
    if dataset_class == torchvision.datasets.SVHN:
        dataset = dataset_class(root=my_path()+'/data', split=split, download=True, transform=transforms.ToTensor())
    else:
        dataset = dataset_class(root=my_path()+'/data', train=(split == 'train'), download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in loader:
            batch_samples = images.size(0)  
            images = images.view(batch_samples, images.size(1), -1)  
            
            mean += images.mean(dim=2).sum(dim=0)
            std += images.std(dim=2).sum(dim=0)  
            total_images += batch_samples

    mean /= total_images  
    std /= total_images  

    return mean, std

def calculate_min_max(dataset, split='train', batch_size=64):
    '''
    This function calculates the minimum and maximum pixel values for the specified dataset.

    Parameters:
    - dataset (str): The name of the dataset to calculate min and max for. Valid options are 'MNIST', 'FMNIST', 'CIFAR10', and 'SVHN'.
    - split (str, optional): The split of the dataset to use ('train' or 'test'). Default is 'train'.
    - batch_size (int, optional): The batch size for loading the dataset. Default is 64.

    Returns:
    - min_pixel (float): The minimum pixel value in the dataset.
    - max_pixel (float): The maximum pixel value in the dataset.
    '''
    dataset_mapping = {
        'MNIST': torchvision.datasets.MNIST,
        'FMNIST': torchvision.datasets.FashionMNIST,
        'CIFAR10': torchvision.datasets.CIFAR10,
        'SVHN': torchvision.datasets.SVHN
    }
    
    if dataset not in dataset_mapping:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataset_class = dataset_mapping[dataset]

    if dataset_class == torchvision.datasets.SVHN:
        dataset = dataset_class(root=my_path()+'/data', split=split, download=True, transform=transforms.ToTensor())
    else:
        dataset = dataset_class(root=my_path()+'/data', train=(split == 'train'), download=True, transform=transforms.ToTensor())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    min_pixel = float('inf')
    max_pixel = float('-inf')
    for images, _ in loader:
        min_pixel = min(min_pixel, images.min().item())
        max_pixel = max(max_pixel, images.max().item())

    return min_pixel, max_pixel

def unnormalize_image(image, dataset, split='train', batch_size=64):
    '''
    This function unnormalizes an image by reversing the normalization applied during preprocessing (i.e., converting back to original pixel values using mean and std).

    Parameters:
    - image (torch.Tensor): The image tensor to unnormalize.
    - dataset (str): The dataset for which to reverse the normalization. Valid options are 'MNIST', 'FMNIST', 'CIFAR10', and 'SVHN'.
    - split (str, optional): The split of the dataset to use ('train' or 'test'). Default is 'train'.
    - batch_size (int, optional): The batch size for loading the dataset. Default is 64.

    Returns:
    - unnorm_image (torch.Tensor): The unnormalized image tensor.
    '''

    dataset_mapping = {
        'MNIST': ((0.13066041469573975,), (0.30150410532951355,)),
        'FMNIST': (0.2860407531261444, 0.320453405380249),
        'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2006)),
        'SVHN': ((0.43768206238746643, 0.4437699019908905, 0.4728040099143982),(0.12008649855852127, 0.12313701957464218, 0.10520392656326294))
        }
    try:
        mean, std = dataset_mapping[dataset]
    except KeyError:
        mean, std = calculate_mean_std(dataset, split=split, batch_size=batch_size)
    
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=image.dtype, device=image.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=image.dtype, device=image.device)
    
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    unnorm_image = image * std + mean
    return unnorm_image







