def suggest_params(trial):
	dropout = trial.suggest_float('dropout', 0, 0.3, step=0.05)  # used twice
	activation = trial.suggest_categorical('activation', ['linear', 'leakyrelu'])  # used for conditional sampling
	rna_hidden = trial.suggest_int('rna_hidden', 500, 2000, step=250)  # hdim should be less than rna_hidden
	hdim = trial.suggest_int('hdim', 100, min(rna_hidden, 800), step=100)  # shared_hidden should be less than hdim
	shared_hidden = trial.suggest_int('shared_hidden', 100, min(hdim*2, 500), step=100)  # zdim should be less than shared_hidden
	num_layers = trial.suggest_int('num_layers', 1, 3, step=1) if activation == 'leakyrelu' else 1
	rna_num_layers = trial.suggest_int('rna_num_layers', 1, 3, step=1)
	loss_weights_kl = trial.suggest_float('loss_weights_kl', 1e-10, 1e-4, log=True)

	params = {'activation': activation,
			  'batch_norm': True,
			  'batch_size': 512,
			  'dropout': dropout,
			  'hdim': hdim,
			  'loss_weights': [10.0, 0.0, loss_weights_kl],
			  'loss_weights_scRNA': 10.0,
			  'loss_weights_seq': 0.0,
			  'loss_weights_kl': loss_weights_kl,
			  'losses': ['MSE', 'CE'],
			  'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
			  'num_layers': num_layers,
			  'scRNA_model_arch': 'MLP',
			  'scRNA_model_hyperparams': {
				  'activation': 'leakyrelu',
				  'batch_norm': True,
				  'dropout': dropout,
				  'gene_hidden': [rna_hidden] * rna_num_layers,
				  'num_layers': rna_num_layers,
				  'output_activation': 'linear'
			  },
			  'seq_model_arch': 'None',
			  'seq_model_hyperparams': {},
			  'shared_hidden': [shared_hidden] * num_layers,
			  'zdim': trial.suggest_int('zdim', 5, min(shared_hidden, 50), step=5)
			  }

	return params


# Keys need to match optuna param name in suggest_params
init_params = [{'activation': 'leakyrelu',
				'batch_size': 512,
				'dropout': 0.2,
				'hdim': 800,
				'loss_weights_seq': 0.0,
				'loss_weights_kl': 5e-8,
				'lr': 1e-3,
				'num_layers': 1,
				'shared_hidden': 200,
				'zdim': 50,

				'rna_hidden': 800,
				'rna_num_layers': 1,
				},

			   {'activation': 'leakyrelu',
				'batch_size': 512,
				'dropout': 0.25,
				'hdim': 300,
				'loss_weights_seq': 0.0,
				'loss_weights_kl': 5e-8,
				'lr': 0.00006,
				'num_layers': 1,
				'shared_hidden': 400,
				'zdim': 50,

				'rna_hidden': 1750,
				'rna_num_layers': 1,
				},
			   {'activation': 'leakyrelu',
				'batch_size': 256,
				'dropout': 0.2,
				'hdim': 400,
				'loss_weights_seq': 0.0,
				'loss_weights_kl': 5e-5,
				'lr': 0.001,
				'num_layers': 1,
				'shared_hidden': 400,
				'zdim': 50,

				'rna_hidden': 1250,
				'rna_num_layers': 1,
				}
			   ]
