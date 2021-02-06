import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes=43):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dout1 = nn.Dropout(p=0.5)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=256)
        self.dout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    # ! Feed forwad function

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool1(output)
        output = self.dout1(output)

        output = self.conv3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.relu4(output)

        output = self.pool2(output)
        output = self.dout2(output)

        # * Above output will be in matrix form, with shape (64,32,15,15)
        output = output.view(-1, 64*7*7)
        output = self.fc1(output)
        output = self.dout3(output)
        output = self.fc2(output)

        return output


if __name__ == "__name__":
    model = ConvNet(num_classes=43)
