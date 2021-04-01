from ray import tune

params = {
	'seq_model_hyperparams': {
		'num_features_1': tune.qrandint(20, 50, 10),
		'num_features_2': tune.qrandint(50, 100, 25),
		'num_features_3': tune.qrandint(50, 200, 50),
		'dropout': tune.quniform(0.0, 0.2, 0.05),
		'batch_norm': tune.choice([True, False]),
		'embedding_dim': tune.qrandint(20, 80, 20),
		'encoder': {
			'kernel_1': tune.qrandint(3, 7),
			'kernel_23': tune.qrandint(3, 5),
			'stride_1': tune.qrandint(1, 3),
			'stride_23': tune.qrandint(1, 3),
			# 'kernel': [5, 3, 3],
			# 'stride': [1, 3, 3],
			# 'num_features': [32, 64, 128],
			'num_layers': None,
			'activation': 'leakyrelu',
		},
		'decoder': {
			'kernel_1': tune.qrandint(3, 5),
			'kernel_2': tune.qrandint(3, 5),
			'stride_1': tune.qrandint(2, 3),
			'stride_2': tune.qrandint(2, 3),
			# 'kernel': [3, 3],  # omit last, as last kernel size is calculated
			# 'stride': [2, 2],  # omit last, as last stride is calculated
			# 'num_features': [64, 128, 64],  # first is input shape, omit last as it is number of classes
			'initial_feature': tune.qrandint(50, 200, 50),
			'initial_len': tune.qrandint(3, 10, 1),
			'num_layers': None,
			'activation': 'relu'
		}
	},
	'use_vdj': True,
	'vdj_embedding_dim': tune.qrandint(20, 60, 20),
	'vdj_dec_layer_1': tune.qrandint(80, 200, 40),
	'vdj_dec_layer_2': tune.qrandint(20, 80, 20),
	'vdj_dec_layers': [128, 64],

	'use_embedding_matrix': tune.choice([True, False]),
	'dec_hdim_1': tune.qrandint(80, 200, 40),
	'dec_hdim_2': tune.qrandint(150, 300, 50),
	'dec_hdim': [128, 256],
	'enc_hdim_1': tune.qrandint(150, 300, 50),
	'enc_hdim_2': tune.qrandint(150, 300, 50),
	'enc_hdim': [256, 256],
	'zdim': tune.qrandint(20, 500, 20),

	# Loss and optimizer
	'lr': tune.qloguniform(1e-5, 1e0, 1e-5),
	'batch_size': tune.choice([2048, 4096, 8192]),
	'loss_weights_1': tune.qloguniform(1e-1, 1e1, 1e-1),
	'loss_weights_2': 1.0,
	'loss_weights_3': tune.qloguniform(1e-5, 1e0, 1e-5),
	'loss_weights': [1.0, 1.0, 1.0e-3]
}

init_params = [{
		'seq_model_hyperparams': {
			'num_features_1': 32,
			'num_features_2': 64,
			'num_features_3': 128,
			'num_features': [32, 64, 128],
			'dropout': 0.0,
			'batch_norm': False,
			'embedding_dim': 64,
			'encoder':
				{'kernel_1': 5,
				 'kernel_23': 3,
				 'stride_1': 1,
				 'stride_23': 3,
				 'kernel': [5, 3, 3],
				 'stride': [1, 3, 3],
				 'num_layers': 3,
				 'activation': 'leakyrelu'
				 },
			'decoder':
				{'kernel_1': 3,
				 'kernel_2': 3,
				 'stride_1': 2,
				 'stride_2': 2,
				 'kernel': [3, 3],  # omit last, as last kernel size is calculated
				 'stride': [2, 2],  # omit last, as last stride is calculated
				 'initial_feature': 64,
				 'initial_len': 4,
				 'num_layers': 3,
				 'activation': 'relu'
				 }
		},
		'use_vdj': True,
		'vdj_embedding_dim': 48,
		'vdj_dec_layer_1': 128,
		'vdj_dec_layer_2': 64,
		'vdj_dec_layers': [128, 64],

		'use_embedding_matrix': True,
		'dec_hdim_1': 128,
		'dec_hdim_2': 256,
		'dec_hdim': [128, 256],
		'enc_hdim_1': 256,
		'enc_hdim_2': 256,
		'enc_hdim': [256, 256],
		'zdim': 256,

		# Loss and optimizer
		'lr': 1.0e-3,
		'batch_size': 8192,
		'loss_weights_1': 1.0,
		'loss_weights_2': 1.0,
		'loss_weights_3': 1.0e-3,
		'loss_weights': [1.0, 1.0, 1.0e-3]
	},
	{
		'seq_model_hyperparams': {
			'num_features_1': 32,
			'num_features_2': 64,
			'num_features_3': 128,
			'num_features': [32, 64, 128],
			'dropout': 0.0,
			'batch_norm': False,
			'embedding_dim': 64,
			'encoder':
				{'kernel_1': 5,
				 'kernel_23': 3,
				 'stride_1': 1,
				 'stride_23': 3,
				 'kernel': [5, 3, 3],
				 'stride': [1, 3, 3],
				 'num_layers': 3,
				 'activation': 'leakyrelu'
				 },
			'decoder':
				{'kernel_1': 3,
				 'kernel_2': 3,
				 'stride_1': 2,
				 'stride_2': 2,
				 'kernel': [3, 3],  # omit last, as last kernel size is calculated
				 'stride': [2, 2],  # omit last, as last stride is calculated
				 'initial_feature': 64,
				 'initial_len': 4,
				 'num_layers': 3,
				 'activation': 'relu'
				 }
		},
		'use_vdj': True,
		'vdj_embedding_dim': 48,
		'vdj_dec_layer_1': 128,
		'vdj_dec_layer_2': 64,
		'vdj_dec_layers': [128, 64],

		'use_embedding_matrix': True,
		'dec_hdim_1': 128,
		'dec_hdim_2': 256,
		'dec_hdim': [128, 256],
		'enc_hdim_1': 256,
		'enc_hdim_2': 256,
		'enc_hdim': [256, 256],
		'zdim': 256,

		# Loss and optimizer
		'lr': 1.0e-4,
		'batch_size': 8192,
		'loss_weights_1': 1.0,
		'loss_weights_2': 1.0,
		'loss_weights_3': 1.0e-3,
		'loss_weights': [1.0, 1.0, 1.0e-3]
	}
]
