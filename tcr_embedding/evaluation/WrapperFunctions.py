import numpy as np


def get_model_prediction_function(model, do_adata=False, metadata=None):
    """
    Wrapper function for our embedding models
    :param model: trained pytorch model
    :param do_adata: return the adata object (else numpy array)
    :param metadata: add these .obs from the original to the created anndata object
    :return: function for calculating the latent space of cells in an anndata object
    """
    def prediction_function(data):
        """
        Calculates the latent space for a dataset
        :param data: anndata object containing the cell data
        :return: numpy array (num_cells, hidden_dim) latent embedding for each cell
        """
        metadata_tmp = metadata if metadata is not None else []
        latent_space = model.get_latent(data, metadata=metadata_tmp, return_mean=True)
        if do_adata:
            return latent_space
        latent_space = latent_space.X
        return latent_space
    return prediction_function


def get_random_prediction_function(hidden_dim):
    """
    Wrapper for calculating a random latent space without input output dependencies as worst case baseline
    :param hidden_dim: dimensionality of the latent space
    :return: function which calculates the latent space
    """
    def prediction_function(data):
        """
        Calculates a random latent space independent of input data
        :param data: anndata object containing the cell data
        :return: numpy array (num_cells, hidden_dim) representing the latent space
        """
        n = len(data.obs)
        np.random.seed(29031995)
        latent_embedding = np.zeros(shape=(n, hidden_dim))
        for i in range(n):
            latent_embedding[i, :] = np.random.random(hidden_dim)
        return latent_embedding
    return prediction_function
