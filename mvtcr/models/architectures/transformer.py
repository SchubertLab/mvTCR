import math
import torch
import torch.nn as nn


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TrigonometricPositionalEncoding(nn.Module):
    """
        This trigonometric positional embedding was taken from:
        Title: Sequence-to-Sequence Modeling with nn.Transformer and TorchText
        Date: 17th March 2021
        Availability: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, embedding_dim, dropout, max_len):
        super(TrigonometricPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, params, hdim, num_seq_labels):
        """
        :param params: hyperparameters as dict
        :param hdim: output feature dimension
        :param num_seq_labels: number of aa labels, input dim
        """
        super(TransformerEncoder, self).__init__()
        self.params = params

        self.num_seq_labels = num_seq_labels

        self.embedding = nn.Embedding(num_seq_labels, params['embedding_size'], padding_idx=0)
        self.positional_encoding = TrigonometricPositionalEncoding(params['embedding_size'],
                                                                   params['dropout'],
                                                                   params['max_tcr_length'])

        encoding_layers = nn.TransformerEncoderLayer(params['embedding_size'],
                                                     params['num_heads'],
                                                     params['embedding_size'] * params['forward_expansion'],
                                                     params['dropout'])
        self.transformer_encoder = nn.TransformerEncoder(encoding_layers, params['encoding_layers'])

        self.fc_reduction = nn.Linear(params['max_tcr_length'] * params['embedding_size'], hdim)

    def forward(self, x, tcr_len):
        x = self.embedding(x) * math.sqrt(self.num_seq_labels)
        x = x.transpose(0, 1)
        x = x + self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = x.flatten(1)
        x = self.fc_reduction(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, params, hdim, num_seq_labels):
        """

        :param params: hyperparameters as dict
        :param hdim: input feature dimension
        :param num_seq_labels: number of aa labels, output dim
        """
        super(TransformerDecoder, self).__init__()
        self.params = params
        self.hdim = hdim
        self.num_seq_labels = num_seq_labels

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fc_upsample = nn.Linear(hdim, self.params['max_tcr_length'] * params['embedding_size'])
        # the embedding size remains constant over all layers

        self.embedding = nn.Embedding(num_seq_labels, params['embedding_size'], padding_idx=0)
        self.positional_encoding = TrigonometricPositionalEncoding(params['embedding_size'],
                                                                   params['dropout'],
                                                                   self.params['max_tcr_length'])

        decoding_layers = nn.TransformerDecoderLayer(params['embedding_size'],
                                                     params['num_heads'],
                                                     params['embedding_size'] * params['forward_expansion'],
                                                     params['dropout'])
        self.transformer_decoder = nn.TransformerDecoder(decoding_layers, params['decoding_layers'])

        self.fc_out = nn.Linear(params['embedding_size'], num_seq_labels)

    def forward(self, hidden_state, target_sequence):
        """
        Forward pass of the Decoder module
        :param hidden_state: joint hidden state of the VAE
        :param target_sequence: Ground truth output
        :return:
        """
        hidden_state = self.fc_upsample(hidden_state)
        shape = (hidden_state.shape[0], self.params['max_tcr_length'], self.params['embedding_size'])
        hidden_state = torch.reshape(hidden_state, shape)

        hidden_state = hidden_state.transpose(0, 1)

        target_sequence = target_sequence[:, :-1]
        target_sequence = target_sequence.transpose(0, 1)

        target_sequence = self.embedding(target_sequence) * math.sqrt(self.num_seq_labels)
        target_sequence = target_sequence + self.positional_encoding(target_sequence)
        try:
            target_mask = nn.Transformer.generate_square_subsequent_mask(None, target_sequence.shape[0]).to(self.device)
        except:  # new version don't need the None
            target_mask = nn.Transformer.generate_square_subsequent_mask(target_sequence.shape[0]).to(self.device)
        x = self.transformer_decoder(target_sequence, hidden_state, tgt_mask=target_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return x
