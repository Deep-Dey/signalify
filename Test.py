from Data_module import CNN_model
from Data_module import Class_labels
from torchvision.transforms import transforms
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
from PIL import Image


def test_model(file_path):
    transform = transforms.Compose([
        transforms.Resize((30, 30)),
        transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
        transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                             [0.5, 0.5, 0.5])
    ])

    checkpoint = torch.load("model/best_model_state.model")
    model = CNN_model.ConvNet(num_classes=43)
    model.load_state_dict(checkpoint)
    model.eval()

    image = Image.open(file_path).convert('RGB')
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    output = model(input)
    index = output.data.numpy().argmax()
    return Class_labels.label(index)


if __name__ == "__main__":
    file_path = "D:/PROGRAM/Projects/PROJECT-Minor/code/GTSRB/Meta/16.png"
    print(test_model(file_path))
