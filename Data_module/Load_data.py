from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# //TODO: Custom Transforms
transform = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.ToTensor(),  # //* 0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5],  # //* 0-1 to [-1,1] , formula (x-mean)/std
                         [0.5, 0.5, 0.5])
])


# //! custom Dataloader function
def load_data(train_path, test_path):
    train_loader = DataLoader(
        datasets.ImageFolder(train_path, transform=transform),
        batch_size=64, shuffle=True)

    test_loader = DataLoader(
        datasets.ImageFolder(test_path, transform=transform),
        batch_size=32, shuffle=True)

    # //TODO: load all classes(43)
    classes = datasets.ImageFolder(train_path, transform=transform).classes

    return train_loader, test_loader, classes


if __name__ == "__main__":
    # //? Path for training and testing directory
    train_path = "D:/PROGRAM/PROJECT-3rd SEM/code/GTSRB/Train"
    test_path = "D:/PROGRAM/PROJECT-3rd SEM/code/GTSRB/Test"

    train, test, classes = load_data(train_path, test_path)

    print(type(train), len(classes), classes[0], type(classes[0]))
