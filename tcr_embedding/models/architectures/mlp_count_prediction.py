from tcr_embedding.models.architectures.mlp import MLP


def build_mlp(params, n_in, n_out):
    print(params)
    mlp = MLP(n_inputs=n_in,
              n_outputs=n_out,
              hiddens=params['hidden_layers'],
              activation=params['activation'],
              output_activation='exponential',
              dropout=params['dropout'],
              batch_norm=params['batch_norm'],
              regularize_last_layer=False)
    return mlp
