#Author: Erik Connerty
#Date: 3/23/2024

#Pairwise Reservoir Approximation

import numpy as np
from sklearn.linear_model import LinearRegression,ElasticNet,Ridge,Lasso
from tqdm.auto import tqdm
from ESN import SimpleESN

def train_and_evaluate_with_states(esn, input_states, target_series):
    """
    Trains an ESN with precomputed input and target states, then computes a modified MSE.
    De-means predictions and target before computing MSE to focus on pattern similarity.
    """
    input_states = np.array(input_states)

    # Train the readout layer with ElasticNet
    regressor = Ridge(alpha=2)
    regressor.fit(input_states, target_series)
    esn.W_out = regressor.coef_

    # Predict using the biased target states
    predictions = esn.predict(input_states).flatten()

    # Compute MSE on de-meaned series
    mse = np.mean((predictions - target_series) ** 2)

    # Get signal to noise ratio
    mse = mse + 1e-20  # Avoid division by zero
    snr = np.mean(predictions ** 2) / mse
    return snr



def compute_adjacency_matrix_for_epoch(epoch_data,sampling_time=0.01,num_reservoir=10,leaky_rate=0.8):
    """
    Computes the adjacency matrix for a single epoch, optimizing by calculating
    reservoir states only once for each 'i' in the outer loop, and reusing these states
    for all 'j' comparisons.
    """
    n_series = epoch_data.shape[0]  # Number of time series
    mse_results = np.zeros((n_series, n_series))

    sparsity = .3
    if num_reservoir <= 10:
        sparsity = 0.0

    # Initialize the ESN instance
    esn = SimpleESN(n_reservoir=num_reservoir, spectral_radius=1.0, sparsity=sparsity,leaky_rate=leaky_rate)
    esn.initialize_weights()  # Initialize weights once at the start

    for i in range(n_series):
        esn.state = np.zeros(esn.n_reservoir)  # Reset state only here, at the start of each 'i' loop

        # Collect states for the 'i' series
        input_states_i = []
        for input_val in epoch_data[i, :, np.newaxis]:
            esn.update_state(input_val)
            input_states_i.append(esn.state.copy())

        for j in range(n_series):
            target_series = epoch_data[j, :]

            #Calculate the time derivative of the target series
            target_series = np.diff(target_series)/sampling_time
            
            mse_results[j, i] = train_and_evaluate_with_states(esn, input_states_i[:-1], target_series)

    return mse_results


#Pairwise Reservoir Approximation
def PRA(epoch_dat=None,sampling_frequency=250,num_reservoir=10,leaky_rate=0.8):
    """
    Compute the Pairwise Reservoir Approximation (PRA) for a given dataset.
    Args:
        epoch_dat: 3D numpy array of shape [epochs, time_points, series]
        sampling_frequency: Sampling frequency of the data
        num_reservoir: Number of reservoir nodes
        leaky_rate: Leaky rate of the reservoir nodes
    Returns:
        avg_adjacency_matrix: Average adjacency matrix across all epochs
    """
    #Calculate the sampling time
    sampling_time = 1/sampling_frequency
    # Main process
    n_epochs = epoch_dat.shape[0]
    n_series = epoch_dat.shape[1]  # Assuming var_dat is now [epochs, time_points, series]
    # Initialize a list to hold all adjacency matrices
    all_adjacency_matrices = []
    for k in tqdm(range(n_epochs)):
        epoch_data = epoch_dat[k, :, :]  # Extract data for the k-th epoch

        adj_matrix = compute_adjacency_matrix_for_epoch(epoch_data,sampling_time=sampling_time,num_reservoir=num_reservoir)
        all_adjacency_matrices.append(adj_matrix)

    # Average the adjacency matrices across all epochs
    avg_adjacency_matrix = np.mean(np.array(all_adjacency_matrices), axis=0)

    # Set diagonal to zero
    np.fill_diagonal(avg_adjacency_matrix, 0)
    #Min max normalize the matrix
    avg_adjacency_matrix = (avg_adjacency_matrix - np.min(avg_adjacency_matrix)) / (np.max(avg_adjacency_matrix) - np.min(avg_adjacency_matrix))

    return avg_adjacency_matrix

