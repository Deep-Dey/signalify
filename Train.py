from Data_module import CNN_model
from Data_module import Load_data
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import glob


def train_model(train_path, test_path):
    # TODO: Load Dataset using DataLoader
    train_loader, test_loader, classes = Load_data.load_data(
        train_path, test_path)

    # TODO: checking for device (graphics card available or not)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_model.ConvNet(num_classes=43).to(device)

    # TODO: Define Optmizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    # TODO: calculating the size of training and testing images for accuracy and Loss
    train_count = len(glob.glob(train_path+"/**/*.png"))
    test_count = len(glob.glob(test_path+"/**/*.png"))

    # ! Model training and saving best model
    num_epochs = 10
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        # ! Evaluation and training on training dataset
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # TODO: zero the parameter gradients
            optimizer.zero_grad()
            # TODO: forward + backward + optimize
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.data*images.size(0)
            _, prediction = torch.max(outputs.data, dim=1)
            train_accuracy += int(torch.sum(prediction == labels.data))

        # ? Calculating accuracy and loss of Training dataset for an epoch
        train_accuracy = train_accuracy/train_count
        train_loss = train_loss/train_count

        # ! Evaluation on testing dataset
        model.eval()
        test_accuracy = 0.0
        for i, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            outputs = model(images)
            _, prediction = torch.max(outputs.data, dim=1)
            test_accuracy += int(torch.sum(prediction == labels.data))

        # ? Calculating accuracy of Test dataset for an epoch
        test_accuracy = test_accuracy/test_count

        print(f"Epoch: {epoch:2} Train Loss: {train_loss.item():2.8f} \
        Train Accuracy: {train_accuracy:2.8f} Test Accuracy: {test_accuracy:2.8f}")

        # TODO: Save the best model
        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), "model/best_model_state.model")
            best_accuracy = test_accuracy


if __name__ == "__main__":
    # ? Path for training and testing directory
    train_path = "D:/PROGRAM/Projects/PROJECT-Minor/code/GTSRB/Train"
    test_path = "D:/PROGRAM/Projects/PROJECT-Minor/code/GTSRB/Test"
    train_model(train_path, test_path)
