import numpy as np
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression,ElasticNet,Ridge,Lasso
from ESN import SimpleESN
import matplotlib.pyplot as plt
import seaborn as sns

def normc(matrix):
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / column_norms[np.newaxis, :]
    return normalized_matrix

def normr(matrix):
    row_norms = np.linalg.norm(matrix, axis=1)
    normalized_matrix = matrix / row_norms[:, np.newaxis]
    return normalized_matrix

def MVRA(var_dat=None,sampling_frequency=250,num_reservoir=2,leaky_rate=0.5):
    # Initialize some example data (Replace these with your actual data)
    # Reading xarray Data from NetCDF file

    Coupling_strengths = []

    sampling_time = 1 / sampling_frequency


    for k in tqdm(range(var_dat.shape[0])):
        
        x = var_dat[k, :, :]
        y = []
        # Inferring Interactions
        for j in range(x.shape[0]):
            y.append(np.diff(x[j, :]) / sampling_time)
            
        y = np.array(y).T

        # Initialize a list to store reservoir states for each time series
        all_states = []
        
        for j in range(x.shape[0]):  # Iterate over each time series
            esn = SimpleESN(n_reservoir=num_reservoir, spectral_radius=1.0, sparsity=0.0,leaky_rate=leaky_rate)
            esn.initialize_weights()

            single_series_data = x[j, :-1]

            # Collect states for the current time series
            states = []
            for input_val in single_series_data:
                esn.update_state([input_val]) 
                states.append(esn.state.copy())

            all_states.append(np.array(states))

        # Reshape and concatenate states
        # Each item in all_states is of shape [time_points, n_reservoir], so stack along the second dimension
        phix = np.hstack(all_states)  # This forms a matrix of shape [time_points, n_reservoir*num_series]
        
        # Add a column of ones to the regressor matrix
        phix = np.insert(phix, 0, 1, axis=1)

        regressor = Ridge(alpha=8.0)
        #W = inverse @ y
        W = regressor.fit(phix, y).coef_.T
        y_pred = phix @ W


        L = []
        # Initialize G with zeros (or any other value you prefer)
        for i in range(x.shape[0]):  # Assuming x is a 2D NumPy array
            g = W[:, i]  # Copying to avoid modifying the original array
            # Remove the first element from g
            g = np.delete(g, 0)
            # Reshape g
            g = np.reshape(g, (num_reservoir, len(g) // num_reservoir))
            gh_i = np.sqrt(np.sum(g ** 2, axis=0))
            #Change the ith element to a zero
            gh_i[i] = 0
                
            L.append(gh_i)

        # Convert lists to NumPy arrays for further calculations
        L = np.array(L)
        Coupling_strengths.append(L)

    Coupling_strengths = np.array(Coupling_strengths).mean(0)
    Coupling_strengths = normc(Coupling_strengths)
    #Min max normalize the array
    Coupling_strengths = (Coupling_strengths - Coupling_strengths.min()) / (Coupling_strengths.max() - Coupling_strengths.min())

    return Coupling_strengths