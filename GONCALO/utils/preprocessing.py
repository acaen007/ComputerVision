import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# define a named function to replace the lambda 
def repeat_channels(x):
    return x.repeat(3, 1, 1)

# define transforms
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(repeat_channels), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(repeat_channels),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_fashionmnist(batch_size=64, root="./data"):
    """
    Loads the FashionMNIST dataset with the transforms exactly as used in the notebook.
    Returns train and test DataLoaders.
    """
    print("Loading FashionMNIST dataset...")

    trainset = torchvision.datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform_train
    )

    testset = torchvision.datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform_test
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return trainloader, testloader, class_names
