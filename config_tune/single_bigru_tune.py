from ray import tune

params = {'activation': 'leakyrelu',
		  'batch_norm': True,
		  'batch_size': tune.choice([256, 512, 1024]),
		  'dropout': tune.quniform(0, 0.3, 0.05),  # TODO Maybe it interferes with batch_norm
		  'hdim': tune.qrandint(100, 1000, 100),
		  # tune.sample_from(lambda spec: tune.qloguniform(100, spec.config.scRNA_model_hyperparams.gene_hidden[0], 50)),
		  #           'loss_weights': [1.0, 0.0, tune.qloguniform(5e-6, 5e-3, 5e-6)],
		  'loss_weights_scRNA': 10.0,
		  'loss_weights_seq': tune.qloguniform(1e-1, 1e1, 1e-1),
		  'loss_weights_kl': tune.qloguniform(5e-6, 1e0, 5e-6),
		  'loss_scRNA': 'MSE',
		  'losses': ['MSE', 'CE'],
		  'lr': tune.qloguniform(1e-5, 1e-2, 1e-5),
		  'scRNA_model_arch': 'None',
		  'scRNA_model_hyperparams': {},
		  'seq_model_arch': 'BiGRU',
		  'seq_model_hyperparams': {
			  'embedding_dim': tune.qrandint(30, 100, 10),
			  'hidden_size': tune.qrandint(50, 500, 50),
			  'num_layers': tune.qrandint(1, 3, 1),
			  'dropout': tune.quniform(0.0, 0.3, 0.05),
			  'bidirectional': tune.choice([True, False]),
			  'teacher_forcing': tune.quniform(0.0, 1.0, 0.2)
		  },
		  #           'shared_hidden': [tune.qloguniform(100, 500, 100)],  Can't deal with sampling within list
		  'shared_hidden': tune.qrandint(100, 500, 100),
		  'zdim': tune.qrandint(20, 200, 20)
		  }

init_params = [{'activation': 'leakyrelu',
				'batch_norm': True,
				'batch_size': 256,
				'dropout': 0.2,
				'hdim': 800,
				#           'loss_weights': [1.0, 0.0, 5e-5],
				'loss_weights_scRNA': 10.0,
				'loss_weights_seq': 1.0,
				'loss_weights_kl': 5e-5,
				'loss_scRNA': 'MSE',
				'losses': ['MSE', 'CE'],
				'lr': 3e-4,
				'scRNA_model_arch': 'None',
				'scRNA_model_hyperparams': {},
				'seq_model_arch': 'BiGRU',
				'seq_model_hyperparams': {
					'embedding_dim': 64,
					'hidden_size': 256,
					'num_layers': 2,
					'dropout': 0.1,
					'bidirectional': True,
					'teacher_forcing': 1.0
				},
				'shared_hidden': 200,
				'zdim': 100
				}]
