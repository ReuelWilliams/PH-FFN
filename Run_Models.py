import MNIST_training as mt
import manifold_training as mft
import manifold as m
import torch

""" The purpose of this class is to train a model and save
it to a file. This way, we can load the model from the file and run experiments on one 
instance of a trained MLP """

# HYPERPARAMETERS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
MNIST_DIM = 784
HIDDEN_ACTIVATIONS_LENGTHS = [10,10,10,10,10]
NUMBER_OF_MNIST_CLASSES = 10
NUMBER_OF_MANIFOLD_CLASSES = 2
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

## FOR MNIST MODELS
# creating an instance of the MNISTModel class, loading the MNIST dataset and training the model

#nn = mt.MNISTModel(input_dim=MNIST_DIM, output_dim=NUMBER_OF_MNIST_CLASSES, hidden_dims=HIDDEN_ACTIVATIONS_LENGTHS).to(DEVICE)
#nn.load_MINST_model()

# creating an instance of the ManifoldTraining class, loading the manifold data and training the model

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# FOR MANIFOLD MODELS
input_dim = 15
nn = mft.ManifoldTraining(input_dim=input_dim, output_dim=NUMBER_OF_MANIFOLD_CLASSES, hidden_dims=HIDDEN_ACTIVATIONS_LENGTHS).to(DEVICE)
manifold = m.Manifold(num_points=1000, dim=input_dim, sampling_function=m.sample_from_n_sphere, label_function=m.label_left_right)
nn.load_manifold_data(training_manifold=manifold)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------

nn.train(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# testing the model
nn.test()

# saving the model to a file
torch.save(nn, "first_sm_complete_model.pth")
