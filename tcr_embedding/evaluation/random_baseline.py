import numpy as np


def random_embedding(hidden_dim):
    """
    Gets a completely random latent space embedding
    :param hidden_dim: hidden dimensionality of the latent space
    :return: random numpy array (hidden_dim)
    """
    return np.random.random(hidden_dim)


def random_embedding_function(hidden_dim):
    """
    Creates a function for calculating a random latent space without input output dependencies
    :param hidden_dim: dimensionality of the latent space
    :return: function which calculates the latent space
    """
    def func(data):
        """
        Calculates a random latent space independent of input data
        :param data: anndata object containing the cell data
        :return: numpy array (num_cells, hidden_dim) representing the latent space
        """
        n = len(data.obs)
        latent_embedding = np.zeros(shape=(n, hidden_dim))
        for i in range(n):
            latent_embedding[i, :] = random_embedding(hidden_dim)
        return latent_embedding
    return func
