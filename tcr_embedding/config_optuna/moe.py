def suggest_params(trial):
    dropout = trial.suggest_float('dropout', 0, 0.3, step=0.05)  # used twice
    activation = trial.suggest_categorical('activation', ['linear', 'leakyrelu'])  # used for conditional sampling
    rna_hidden = trial.suggest_int('rna_hidden', 500, 2000, step=250)  # hdim should be less than rna_hidden
    hdim = trial.suggest_int('hdim', 100, min(rna_hidden, 800), step=100)  # shared_hidden should be less than hdim
    shared_hidden = trial.suggest_int('shared_hidden', 100, min(hdim * 2, 500),
                                      step=100)  # zdim should be less than shared_hidden
    num_layers = trial.suggest_int('num_layers', 1, 3, step=1) if activation == 'leakyrelu' else 1
    rna_num_layers = trial.suggest_int('rna_num_layers', 1, 3, step=1)
    tfmr_encoding_layers = trial.suggest_int('tfmr_encoding_layers', 1, 4, step=1)  # used twice
    loss_weights_kl = trial.suggest_float('loss_weights_kl', 1e-10, 1e-4, log=True)
    loss_weights_seq = trial.suggest_float('loss_weights_tcr', 1e-5, 1, log=True)

    params = {
        'batch_size': 512,
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'loss_weights': [1.0, loss_weights_seq, loss_weights_kl],

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
        'tcr': {
            'embedding_size': trial.suggest_categorical('tfmr_embedding_size', [16, 32, 64]),
            'num_heads': trial.suggest_categorical('tfmr_num_heads', [2, 4, 8]),
            'forward_expansion': 4,
            'encoding_layers': tfmr_encoding_layers,
            'decoding_layers': tfmr_encoding_layers,  # encoding_layers is used here too
            'dropout': trial.suggest_float('tfmr_dropout', 0, 0.3, step=0.05),
        },
    }
    return params
'''
Haniffa pub
def suggest_params(trial):
    params = {'batch_size': 512,
            'learning_rate': 0.00046187266176987815,
            'loss_weights': [1.0, 0.0018852035668153622, 1.2164799921249682e-08],
            'joint': {'activation': 'leakyrelu',
            'batch_norm': True,
            'dropout': 0.0,
            'hdim': 200,
            'losses': ['MSE', 'CE'],
            'num_layers': 2,
            'shared_hidden': [300, 300],
            'zdim': 35,
            'c_embedding_dim': 20,
            'num_conditional_labels': 94,
            'cond_dim': 20,
            'cond_input': True},
            'rna': {'activation': 'leakyrelu',
            'batch_norm': True,
            'dropout': 0.0,
            'gene_hidden': [1500, 1500],
            'num_layers': 2,
            'output_activation': 'linear',
            'xdim': 5000},
            'tcr': {'embedding_size': 64,
            'num_heads': 4,
            'forward_expansion': 4,
            'encoding_layers': 1,
            'decoding_layers': 1,
            'dropout': 0.3,
            'max_tcr_length': 27,
            'num_seq_labels': 24}}
    return params
'''