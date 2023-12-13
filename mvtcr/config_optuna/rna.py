def suggest_params(trial):
    dropout = trial.suggest_float('dropout', 0, 0.3, step=0.05)  # used twice
    activation = trial.suggest_categorical('activation', ['linear', 'leakyrelu'])  # used for conditional sampling
    rna_hidden = trial.suggest_int('rna_hidden', 500, 2000, step=250)  # hdim should be less than rna_hidden
    hdim = trial.suggest_int('hdim', 100, min(rna_hidden, 800), step=100)  # shared_hidden should be less than hdim
    shared_hidden = trial.suggest_int('shared_hidden', 100, min(hdim * 2, 500),
                                      step=100)  # zdim should be less than shared_hidden
    num_layers = trial.suggest_int('num_layers', 1, 3, step=1) if activation == 'leakyrelu' else 1
    rna_num_layers = trial.suggest_int('rna_num_layers', 1, 3, step=1)
    loss_weights_kl = trial.suggest_float('loss_weights_kl', 1e-10, 1e-4, log=True)

    params = {
        'batch_size': 512,
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'loss_weights': [1.0, 0.0, loss_weights_kl],

        'joint': {
            'activation': activation,
            'batch_norm': True,
            'dropout': dropout,
            'hdim': hdim,
            'losses': ['MSE', 'CE'],
            'num_layers': num_layers,
            'shared_hidden': [shared_hidden] * num_layers,
            'zdim': trial.suggest_int('zdim', 5, min(shared_hidden, 50), step=5),
            'c_embedding_dim': 20,
        },
        'rna': {
            'activation': 'leakyrelu',
            'batch_norm': True,
            'dropout': dropout,
            'gene_hidden': [rna_hidden] * rna_num_layers,
            'num_layers': rna_num_layers,
            'output_activation': 'linear'
        },
        'seq_model_hyperparams': None,
    }
    return params
