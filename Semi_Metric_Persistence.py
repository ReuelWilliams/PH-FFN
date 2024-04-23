import manifold_training as mft
import manifold as m
import torch
import copy
import Persistence_Data_Analysis as pda
import gudhi as gd




def run_experiment(num_experiments: int = 0, max_persistence_dim: int = 4,
                   max_edge_length: int = 10, max_persistence: int = 10, write_to_file: bool = True,
                   name_experiment_set: str = "", model=None, manifold : mft.Manifold=None, num_epochs : int = 100, batch_size : int = 64,
                   learning_rate : float = 0.1, weight_decay : float = 0.0005, momentum : int = 0.9) -> None:
    
    
    for experiment in range(num_experiments):
        betti_nums = [] # to store the betti numbers for each epoch
        folder_name: str = f"{name_experiment_set}/Experiment_{experiment}"
        pda.create_folder(folder_name)  
        model_prime = None # model_prime is the model that will be trained and used to compute the persistence homology. We must reset it for each experiment
        # model is the default model and is not expected to be trained. We make a deep copy of the model to avoid changing the original model
        model_prime = copy.deepcopy(model)
    
        # Train the model with check_ph = True to get the distances between nodes in the network graph for each epoch in training
        model_prime.load_manifold_data(training_manifold=manifold)
        model_prime.train(epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,check_ph=True)
        distances = model_prime.semi_metric_distances
        weight_vectors = model_prime.weight_vectors # weight_vectors is a list of the weight vectors for each epoch

        # now for each epoch we can pull the distance matrix and compute the persistence
        for epoch in range(num_epochs):
            #ph_results = gd.RipsComplex(distance_matrix=distances[epoch], max_edge_length=max_edge_length).create_simplex_tree(max_dimension=max_persistence_dim).persistence()
            ph_results = gd.RipsComplex(points=weight_vectors[epoch], max_edge_length=max_edge_length).create_simplex_tree(max_dimension=max_persistence_dim).persistence()
            epc_betti_nums = pda.pull_valid_features(max_persistence=max_persistence, ph_results=ph_results, max_persistence_dim=max_persistence_dim)
            betti_nums.append(epc_betti_nums)
            if write_to_file:
                pda.write_to_file(f"{folder_name}/ph_of_epoch_{epoch}", str(ph_results))
                pda.write_to_file(f"{folder_name}/epoch_losses", str(model_prime.epochs))
                pda.plot_persistence_diagram(ph_results, title=f"Persistence Diagram for Epoch {epoch}", filename=f"{folder_name}/ph_of_epoch_{epoch}")
                #pda.plot_betti_numbers_(betti_nums, title="Betti Numbers for Epoch {epoch}", filename=f"{folder_name}/betti_nums_of_epoch_{epoch}") - fix this
                pda.write_to_file(f"{folder_name}/betti_nums_epoch_{epoch}", str(betti_nums))
                



# HYPERPARAMETERS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
HIDDEN_ACTIVATIONS_LENGTHS = [100,100,10,10]
NUMBER_OF_MANIFOLD_CLASSES = 2
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


# CREATING MANIFOLD TO TRAIN ON
input_dim = 15
nn = mft.ManifoldTraining(input_dim=input_dim, output_dim=NUMBER_OF_MANIFOLD_CLASSES, hidden_dims=HIDDEN_ACTIVATIONS_LENGTHS).to(DEVICE)
manifold = m.Manifold(num_points=1000, dim=input_dim, sampling_function=m.sample_from_n_sphere, label_function=m.label_left_right)

run_experiment(num_experiments=5, max_persistence_dim=4, max_edge_length=10, max_persistence=10, write_to_file=True, name_experiment_set="Test_Experiment",
                model=nn, manifold=manifold, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

