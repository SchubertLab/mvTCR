from ray import tune


params = {
	'single_scRNA': {'activation': 'leakyrelu',
					 'batch_norm': True,
					 'batch_size': tune.choice([128, 256, 512, 1024]),
					 'dropout': tune.quniform(0, 0.3, 0.05),
					 'hdim': tune.quniform(100, 1000, 100),
					 # tune.sample_from(lambda spec: tune.qloguniform(100, spec.config.scRNA_model_hyperparams.gene_hidden[0], 50)),
					 #           'loss_weights': [1.0, 0.0, tune.qloguniform(5e-6, 5e-3, 5e-6)],
					 'loss_weights_scRNA': 1.0,
					 'loss_weights_seq': 0.0,
					 'loss_weights_kl': tune.qloguniform(5e-6, 5e-3, 5e-6),
					 'losses': ['MSE', 'CE'],
					 'lr': tune.loguniform(1e-5, 1e-2),
					 'scRNA_model_arch': 'MLP',
					 'scRNA_model_hyperparams': {
						 'activation': 'leakyrelu',
						 'batch_norm': True,
						 'dropout': tune.quniform(0, 0.3, 0.05),
						 #               'gene_hidden': [tune.qloguniform(100, 1000, 100)],  Can't deal with sampling within list
						 'gene_hidden': tune.qloguniform(100, 1000, 100),
						 'output_activation': 'relu'
					 },
					 'seq_model_arch': 'None',
					 'seq_model_hyperparams': {},
					 #           'shared_hidden': [tune.qloguniform(100, 500, 100)],  Can't deal with sampling within list
					 'shared_hidden': tune.qloguniform(100, 500, 100),
					 'zdim': tune.qloguniform(20, 100, 10)
					 }

}

init_params = {
	'single_scRNA': [{'activation': 'leakyrelu',
					  'batch_norm': True,
					  'batch_size': 256,
					  'dropout': 0.2,
					  'hdim': 800,
					  #           'loss_weights': [1.0, 0.0, 5e-5],
				      'loss_weights_scRNA': 1.0,
					  'loss_weights_seq': 0.0,
					  'loss_weights_kl': 5e-5,
					  'losses': ['MSE', 'CE'],
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
}
