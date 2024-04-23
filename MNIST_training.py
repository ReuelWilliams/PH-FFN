# Description: This file contains the code for training the MNIST dataset using PyTorch
import torch.nn as nn
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for gpu optimization if available

class MNISTModel(nn.Module):

    ## simple MLP with a a variable number of layers
    def __init__(self, input_dim, output_dim, hidden_dims : torch.tensor, activation = torch.relu, output_activation = torch.softmax):
        super(MNISTModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.activation = activation
        self.output_activation = output_activation
        self.accuracy = 0

    def get_model_info(self) -> str:
        return f"Input Dimension: {self.input_dim}\nOutput Dimension: {self.output_dim}\nHidden Dimensions: {self.hidden_dims}\nActivation Function: {self.activation.__name__}\nModel Accuracy: {self.accuracy}"

    def load_MINST_model(self):
        mnist = datasets.MNIST(root='data', train=True, download=True) # train data only
        self.traindata = mnist.data.float().view(-1, 784)/255
        self.trainlabels = mnist.targets
        mnist = datasets.MNIST(root='data', train=False, download=True)
        self.testdata = mnist.data.float().view(-1, 784)/255
        self.testlabels = mnist.targets

    def get_one_class_data(self, class_num: int):
        class_data = self.traindata[self.trainlabels == class_num]
        return class_data

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return self.output_activation(x, dim = 1)
    



    def train(self, epochs, batch_size, learning_rate, momentum, weight_decay, criterion=nn.CrossEntropyLoss()):
        dataset = TensorDataset(self.traindata, self.trainlabels)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.to(device)  # Move model to the appropriate device
    
        for epoch in range(epochs):
            for batch, labels in train_loader:
                batch, labels = batch.to(device), labels.to(device)  # Move data to the device           
                optimizer.zero_grad()
                output = self(batch)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            print('Epoch: ', epoch, 'Loss: ', loss.item())


    def test(self):
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, self.testdata.shape[0], 100):
                batch = self.testdata[i:i+100]
                labels = self.testlabels[i:i+100]
                outputs = self(batch)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.accuracy = 100 * correct / total
        print('Accuracy of the network on the test images: %d %%' % self.accuracy)

    def pull_hidden_activations(self, input):
        """Returns the activations of the hidden layers"""
        activations = []
        for layer in self.layers[:-1]:
            input = self.activation(layer(input))
            activations.append(input)
        return activations


        