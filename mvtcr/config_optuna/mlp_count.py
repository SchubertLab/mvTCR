# Defines the hyper parameter of the count predictor

def suggest_params(trial):
    n_units = trial.suggest_int('rna_hidden', 100, 1000, step=100)
    num_layers = trial.suggest_int('num_layers', 1, 5, step=1)

    params = {
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-1, log=True),

        'mlp': {
            'activation': 'leakyrelu',
            'dropout': trial.suggest_float('dropout', 0, 0.3, step=0.05),
            'batch_norm': True,
            'hidden_layers': [n_units] * num_layers
        },
    }
    return params
