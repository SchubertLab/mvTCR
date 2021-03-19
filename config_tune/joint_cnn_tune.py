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
		  'losses': ['MSE', 'CE'],
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
		  'seq_model_arch': 'CNN',
		  'seq_model_hyperparams': {
			  'encoder': {
				  'embedding_dim': tune.qrandint(20, 70, 10),
				  'kernel_1': tune.qrandint(3, 7),
				  'kernel_23': tune.qrandint(3, 5),
				  'stride_1': tune.qrandint(1, 3),
				  'stride_23': tune.qrandint(1, 3),
				  'num_features_1': tune.qrandint(20, 50, 10),
				  'num_features_2': tune.qrandint(50, 100, 25),
				  'num_features_3': tune.qrandint(50, 200, 50),
				  # 'kernel': [5, 3, 3],
				  # 'stride': [1, 3, 3],
				  # 'num_features': [32, 64, 128],
				  'num_layers': None,
				  'dropout': tune.quniform(0.0, 0.2, 0.05),
				  'batch_norm': tune.choice([True, False]),
				  'activation': 'leakyrelu',
			  },
			  'decoder': {
				  'kernel_1': tune.qrandint(3, 5),
				  'kernel_2': tune.qrandint(3, 5),
				  'stride_1': tune.qrandint(2, 3),
				  'stride_2': tune.qrandint(2, 3),
				  'num_features_1': tune.qrandint(40, 200, 40),
				  'num_features_2': tune.qrandint(20, 100, 20),
				  'num_features_3': tune.qrandint(20, 50, 10),
				  # 'kernel': [3, 3],  # omit last, as last kernel size is calculated
				  # 'stride': [2, 2],  # omit last, as last stride is calculated
				  # 'num_features': [64, 128, 64],  # first is input shape, omit last as it is number of classes
				  'initial_len': tune.randint(3, 10),
				  'num_layers': None,
				  'dropout': tune.quniform(0.0, 0.2, 0.05),
				  'batch_norm': tune.choice([True, False]),
				  'activation': 'relu'
			  }
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
				'seq_model_arch': 'CNN',
				'seq_model_hyperparams': {
					'encoder': {
						'embedding_dim': 64,
						'kernel_1': 5,
						'kernel_23': 3,
						'stride_1': 1,
						'stride_23': 3,
						'num_features_1': 32,
						'num_features_2': 64,
						'num_features_3': 128,
						# 'kernel': [7, 4, 2],
						# 'stride': [5, 3, 6],
						# 'num_features': [53, 64, 128],
						'num_layers': None,
						'dropout': 0.0,
						'batch_norm': False,
						'activation': 'leakyrelu',
					},
					'decoder': {
						'kernel_1': 3,
						'kernel_2': 3,
						'stride_1': 2,
						'stride_2': 2,
						'num_features_1': 64,
						'num_features_2': 128,
						'num_features_3': 64,
						# 'kernel': [7, 5, 3],  # omit last, as last kernel size is calculated
						# 'stride': [3, 1, 2],  # omit last, as last stride is calculated
						# 'num_features': [64, 87, 64, 12],  # first is input shape, omit last as it is number of classes
						'initial_len': 4,
						'num_layers': None,
						'dropout': 0.0,
						'batch_norm': False,
						'activation': 'relu'
					}
				},
				'shared_hidden': 200,
				'zdim': 100
				}]
