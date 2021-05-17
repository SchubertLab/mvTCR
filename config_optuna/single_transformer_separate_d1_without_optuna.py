def suggest_params(trial):
	dropout = trial.suggest_float('dropout', 0, 0.3, step=0.05)  # used twice
	activation = trial.suggest_categorical('activation', ['linear', 'leakyrelu'])  # used for conditional sampling
	hdim = trial.suggest_int('hdim', 100, 800, step=100)  # shared_hidden should be less than hdim
	shared_hidden = trial.suggest_int('shared_hidden', 100, min(hdim*2, 500), step=100)  # zdim should be less than shared_hidden
	num_layers = trial.suggest_int('num_layers', 1, 3, step=1) if activation == 'leakyrelu' else 1
	tfmr_encoding_layers = trial.suggest_int('tfmr_encoding_layers', 1, 4, step=1)  # used twice
	loss_weights_kl = trial.suggest_float('loss_weights_kl', 1e-6, 1e0, log=True)

	params = {'activation': activation,
			  'batch_norm': True,
			  'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
			  'dropout': dropout,
			  'hdim': hdim,
			  'loss_weights': [0.0, 1.0, loss_weights_kl],
			  'loss_weights_scRNA': 0.0,
			  'loss_weights_seq': 1.0,
			  'loss_weights_kl': loss_weights_kl,
			  'losses': ['MSE', 'CE'],
			  'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
			  'num_layers': num_layers,
			  'scRNA_model_arch': 'None',
			  'scRNA_model_hyperparams': {},
			  'seq_model_arch': 'Transformer',
			  'seq_model_hyperparams': {
				  'embedding_size': trial.suggest_categorical('tfmr_embedding_size', [16, 32, 64]),
				  'num_heads': trial.suggest_categorical('tfmr_num_heads', [1, 2, 4, 8]),
				  'forward_expansion': trial.suggest_categorical('tfmr_forward_expansion', [1, 2, 4]),
				  'encoding_layers': tfmr_encoding_layers,
				  'decoding_layers': tfmr_encoding_layers,  # encoding_layers is used here too
				  'dropout': trial.suggest_float('tfmr_dropout', 0, 0.3, step=0.05),
			  },
			  'shared_hidden': [shared_hidden] * num_layers,
			  'zdim': trial.suggest_int('zdim', 10, min(shared_hidden, 200), step=10)
			  }

	return params


# Keys need to match optuna param name in suggest_params
init_params = [{'activation': 'leakyrelu',
				'batch_size': 256,
				'dropout': 0.2,
				'hdim': 800,
				'loss_weights_seq': 1.0,
				'loss_weights_kl': 5e-5,
				'lr': 1e-3,
				'num_layers': 1,
				'shared_hidden': 200,
				'zdim': 100,

				'tfmr_embedding_size': 32,
				'tfmr_num_heads': 4,
				'tfmr_forward_expansion': 4,
				'tfmr_encoding_layers': 2,
				'tfmr_dropout': 0.1
				},

			   {'activation': 'leakyrelu',
				'batch_size': 512,
				'dropout': 0.25,
				'hdim': 300,
				'loss_weights_seq': 1.0,
				'loss_weights_kl': 5e-5,
				'lr': 0.00006,
				'num_layers': 1,
				'shared_hidden': 400,
				'zdim': 160,

				'tfmr_embedding_size': 32,
				'tfmr_num_heads': 4,
				'tfmr_forward_expansion': 4,
				'tfmr_encoding_layers': 2,
				'tfmr_dropout': 0.1
				},
			   {'activation': 'leakyrelu',
				'batch_size': 512,
				'dropout': 0.15,
				'hdim': 1000,
				'loss_weights_seq': 1.0,
				'loss_weights_kl': 5e-3,
				'lr': 0.00002,
				'num_layers': 1,
				'shared_hidden': 500,
				'zdim': 160,

				'tfmr_embedding_size': 32,
				'tfmr_num_heads': 4,
				'tfmr_forward_expansion': 4,
				'tfmr_encoding_layers': 2,
				'tfmr_dropout': 0.1
				}]
