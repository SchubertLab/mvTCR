from ray import tune


params = {'activation': 'leakyrelu',
		  'batch_norm': True,
		  'batch_size': tune.choice([256, 512, 1024]),
		  'dropout': tune.quniform(0, 0.3, 0.05),  # TODO Maybe it interferes with batch_norm
		  'hdim': tune.qrandint(100, 1000, 100),
		  # tune.sample_from(lambda spec: tune.qloguniform(100, spec.config.scRNA_model_hyperparams.gene_hidden[0], 50)),
		  #           'loss_weights': [1.0, 0.0, tune.qloguniform(5e-6, 5e-3, 5e-6)],
		  'loss_weights_scRNA': 10.0,
		  'loss_weights_seq': 0.0,
		  'loss_weights_kl': tune.qloguniform(5e-6, 1e0, 5e-6),
		  'losses': ['NB', 'CE'],
		  'lr': tune.qloguniform(1e-5, 1e-2, 1e-5),
		  'scRNA_model_arch': 'MLP',
		  'scRNA_model_hyperparams': {
			  'activation': 'leakyrelu',
			  'batch_norm': True,
			  'dropout': tune.quniform(0, 0.3, 0.05),
			  #               'gene_hidden': [tune.qloguniform(100, 1000, 100)],  Can't deal with sampling within list
			  'gene_hidden': tune.qrandint(500, 2000, 250),
			  'output_activation': 'relu'
		  },
		  'seq_model_arch': 'None',
		  'seq_model_hyperparams': {},
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
				'loss_weights_seq': 0.0,
				'loss_weights_kl': 5e-5,
				'losses': ['NB', 'CE'],
				'lr': 1e-3,
				'scRNA_model_arch': 'MLP',
				'scRNA_model_hyperparams': {
					'activation': 'leakyrelu',
					'batch_norm': True,
					'dropout': 0.2,
					'gene_hidden': 800,
					'output_activation': 'relu'
				},
				'seq_model_arch': 'None',
				'seq_model_hyperparams': {},
				'shared_hidden': 200,
				'zdim': 100
				}]
