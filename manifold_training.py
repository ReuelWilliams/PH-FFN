import torch.nn as nn
import torch 
import copy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from manifold import Manifold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

class ManifoldTraining(nn.Module):
    def __init__(self, input_dim : int , hidden_dims : torch.tensor, output_dim: int , activation = torch.relu, output_activation = nn.Softmax(dim = 1)):
        super(ManifoldTraining, self).__init__()
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
        self.semi_metric_distances = []
        self.epochs = []
        self.weight_vectors = []
        
        ## complete structure including last layers
        self.activation_lengths = copy.deepcopy(hidden_dims)
        self.activation_lengths.append(output_dim)
        self.activation_lengths.insert(0,input_dim)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return self.output_activation(x)
        
    def get_model_info(self) -> str:
        return f"Input Dimension: {self.input_dim}\nOutput Dimension: {self.output_dim}\nHidden Dimensions: {self.hidden_dims}\nActivation Function: {self.activation.__name__}\nSampling Info:{self.sampling_name}\nAccuracy: {self.accuracy}"

    
    def load_manifold_data(self, training_manifold:Manifold):
        self.sampling_name = training_manifold.get_name()
        self.traindata, self.labels = training_manifold.get_training_data() 
        self.testdata, self.test_labels = training_manifold.get_test_data()

    def get_one_class_data(self, class_num: int, shuffle = True):
        class_data = self.traindata[self.labels == class_num]
        if shuffle:
            class_data = class_data[torch.randperm(class_data.shape[0])]
        return class_data
       
    
    def train(self, epochs, batch_size, learning_rate, momentum, weight_decay, criterion=nn.CrossEntropyLoss(), check_ph = False):
        dataset = TensorDataset(self.traindata, self.labels)
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
            if check_ph:
                self.semi_metric_distances.append(self.get_semi_metric_distance_matrix())
                self.weight_vectors.append(self.get_weight_vectors())
            self.epochs.append(loss.item())
            print('Epoch: ', epoch, 'Loss: ', loss.item())


    def test(self):
        correct = 0
        total = 0
        self.to(device)  # Make sure the model is on the right device

        with torch.no_grad():
            for i in range(0, self.testdata.shape[0], 100):
                # Ensure the batch of test images and labels are on the same device as the model
                batch = self.testdata[i:i+100].to(device)
                labels = self.test_labels[i:i+100].to(device)
            
                outputs = self(batch)  # Forward pass
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        self.accuracy : float = 100 * correct / total
        print('Accuracy of the network on the manifold data: %d %%' % (100 * correct / total))

    def pull_hidden_activations(self, input: torch.tensor):
        """Returns the activations of the hidden layers"""
        activations = []
        for layer in self.layers[:-1]:
            input = self.activation(layer(input))
            activations.append(input)
        return activations

    # Accessing weights for each layer
    def get_weight_matrices(self):   
        weight_matrices = []
        for i, weights in enumerate(self.parameters()):
            if i % 2 == 0:
                # Use .data to get the tensor
                weight_matrices.append(weights.data)
        return weight_matrices

    def get_weight_vectors(self):
        weight_matrices = self.get_weight_matrices()
        weight_vectors = []
        for matrix in weight_matrices:
            for row in range(matrix.shape[0]):
                weight_vectors.append(matrix[row].tolist())
        max_length = max(len(vector) for vector in weight_vectors)
        padded_weight_vectors = [vector + [0] * (max_length - len(vector)) for vector in weight_vectors]
        return padded_weight_vectors

    @staticmethod
    def total_ordering_to_layer_ordering(num, activation_lengths): # in format (neuron, layer)
        counter = 0
        while num >= 0:
            num -= activation_lengths[counter]
            counter += 1
        counter -= 1 # to account for last counter shift that wasn't used
        num += activation_lengths[counter] # to make j possible (get back neuron indexing)    
        return (num, counter)


    def get_semi_metric_distance_matrix(self):
        abs_weights = [torch.abs(x) for x in self.get_weight_matrices()] # pull the weight matrices and make absolute

        # this beginning loop will ensure we save time on computations: 
        # num_tensors is the total number of layers (including input layer)
        num_tensors = len( abs_weights) + 1
        layer_distances = {}

        # Loop through each tensor pair and perform matrix multiplication
        for i in range(num_tensors): # i and j index layers
            for j in range(i + 1, num_tensors):
                layer_distances[str(i) + str(j)] =  abs_weights[j - 1]
                k = j - 2
                while k >= i:
                    layer_distances[str(i) + str(j)] = torch.matmul(layer_distances[str(i) + str(j)],  abs_weights[k])
                    k -= 1
        #print(layer_distances)

        total_num_neurons : int = np.sum(self.activation_lengths)
        distance_matrix = torch.zeros(total_num_neurons, total_num_neurons)
        for i in range(total_num_neurons):
            for j in range(i, total_num_neurons):

                # distance between a neuron and itself is 0
                if i == j: 
                    distance_matrix[i][j] = 0
                    continue 

                # pulling layer ordering of neurons i and j
                i_layer_ordering = self.total_ordering_to_layer_ordering(num=i, activation_lengths = self.activation_lengths)
                j_layer_ordering = self.total_ordering_to_layer_ordering(num=j, activation_lengths = self.activation_lengths)

                # if they are in the same layer then the distance is zero
                if i_layer_ordering[1] == j_layer_ordering[1]:
                    distance_matrix[i][j] = np.inf
                    continue
                # if they are in different layers
                connection_strength =  layer_distances[str(i_layer_ordering[1]) +  str(j_layer_ordering[1])][j_layer_ordering[0]][i_layer_ordering[0]]
                if connection_strength == 0 :
                    distance_matrix[i][j] = np.inf
                    continue
                distance_matrix[i][j] =1/ layer_distances[str(i_layer_ordering[1]) +  str(j_layer_ordering[1])][j_layer_ordering[0]][i_layer_ordering[0]]
        return distance_matrix + distance_matrix.t()
    
    def get_distances(self):
        return self.distances
    def get_losses(self):
        return self.epochs


#get_distance_matrix(nn_manifold)
#print(get_weight_matrices(nn_manifold))
#get_distance_matrix(nn_manifold)