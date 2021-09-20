import torch
import torch.nn as nn


class BiGRUEncoder(nn.Module):
	def __init__(self, params, hdim, num_seq_labels):
		"""

		:param params: hyperparameters as dict
		:param hdim: output feature dimension
		:param num_seq_labels: number of aa labels, input dim
		"""
		super(BiGRUEncoder, self).__init__()
		self.embedding_dim = params['embedding_dim']
		self.hidden_size = params['hidden_size']
		self.num_layers = params['num_layers']
		self.dropout = params['dropout']
		self.bidirectional = params['bidirectional']
		self.num_seq_labels = num_seq_labels
		self.hdim = hdim

		self.embedding = nn.Embedding(num_embeddings=num_seq_labels, embedding_dim=params['embedding_dim'], padding_idx=0)

		self.gru = nn.GRU(
			input_size=params['embedding_dim'],
			hidden_size=params['hidden_size'],
			num_layers=params['num_layers'],
			batch_first=True,
			dropout=params['dropout'],
			bidirectional=params['bidirectional']
		)

		input_size = params['hidden_size'] * params['num_layers'] * 2 if params['bidirectional'] else params['hidden_size'] * params['num_layers']

		self.output_layer = nn.Linear(input_size, hdim)

	def forward(self, tcr_seq, tcr_len):
		"""
		:param tcr_seq: torch.LongTensor, shape=[batch_size, max_seq_len]
		:param tcr_len: torch.LongTensor, shape=[batch_size, 1] holds the real sequence length (without padding)
		:return: torch.Tensor, shape=[batch_size, hdim]
		"""

		# initialize hidden state
		num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
		batch_size = tcr_seq.shape[0]
		h0 = torch.randn(num_layers, batch_size, self.hidden_size).to(tcr_seq.device)

		x = self.embedding(tcr_seq)  # [batch_size, max_seq_len, embedding_dim]
		x = torch.nn.utils.rnn.pack_padded_sequence(x, tcr_len, batch_first=True, enforce_sorted=False)
		tokens, hidden = self.gru(x, h0)  # hidden.shape=[num_layer(*2 if bidirectional), batch_size, hidden_size]
		# tokens, _ = torch.nn.utils.rnn.pad_packed_sequence(tokens, batch_first=True)  # output of each token not needed

		hidden = hidden.permute(1, 0, 2)  # shape=[batch_size, num_layers(*2 if biGRU), hidden_size]
		hidden = hidden.flatten(1)  # shape=[batch*num_layers, hidden_size]

		output = self.output_layer(hidden)

		return output


class BiGRUDecoder(nn.Module):
	def __init__(self, params, input_dim, num_seq_labels):
		"""

		:param params: hyperparameters as dict
		:param hdim: input feature dimension
		:param num_seq_labels: number of aa labels, output dim
		"""
		super(BiGRUDecoder, self).__init__()
		self.hidden_size = params['hidden_size']
		self.num_layers = params['num_layers']
		self.dropout = params['dropout']
		self.bidirectional = params['bidirectional']
		self.num_seq_labels = num_seq_labels
		self.input_dim = input_dim
		self.teacher_forcing = params['teacher_forcing']

		self.embedding = nn.Embedding(num_embeddings=num_seq_labels, embedding_dim=params['embedding_dim'], padding_idx=0)
		self.hidden_state_layer = nn.Linear(input_dim, params['hidden_size'] * params['num_layers'])

		self.gru = nn.GRU(
			input_size=params['embedding_dim'],
			hidden_size=params['hidden_size'],
			num_layers=params['num_layers'],
			batch_first=True,
			dropout=params['dropout'],
			bidirectional=False
		)

		self.output_layer = nn.Linear(params['hidden_size'], num_seq_labels)
		# self.softmax = nn.Softmax(dim=-1)  # not needed using nn.CrossEntropyLoss


	def forward(self, shared_hidden_state, gt_tcr_seq):
		"""
		Forward pass including autoregressively predicting the sequence using a for loop
		:param shared_hidden_state: shared hidden state after shared_decoder, torch.FloatTensor with shape [batch_size, input_dim]
		:param gt_tcr_seq: torch.LongTensor, shape=[batch_size, max_seq_len] ground truth tcr-seq, same as encoder input, only needed for shape and start symbol
		:return:
		"""
		decoder_raw_output = []
		batch_size, seq_len = gt_tcr_seq.shape

		hidden_state = self.hidden_state_layer(shared_hidden_state)  # shape=[batch_size, num_layers*hidden_size]
		hidden_state = hidden_state.reshape(batch_size, self.num_layers, self.hidden_size)  # shape=[batch_size, num_layers, hidden_size]
		hidden_state = hidden_state.permute(1, 0, 2)  # switch batch and layer dim, shape=[num_layers, batch_size, hidden_size]
		hidden_state = hidden_state.contiguous()  # permute returns a non-contiguous Tensor

		use_teacher_forcing = True if torch.rand(1) < self.teacher_forcing else False
		if use_teacher_forcing:
			input = gt_tcr_seq[:, :-1]  # all timesteps except last one
			embedding = self.embedding(input)

			output, hidden_state = self.gru(embedding, hidden_state)
			decoder_raw_output = self.output_layer(output)

		else:
			# Initialize first input and hidden state
			input = gt_tcr_seq[:, 0].unsqueeze(1)  # start symbol
			for step in range(seq_len-1):
				output, hidden_state = self.step(input, hidden_state)
				input = torch.argmax(output, dim=-1)  # only needs to be seq_len=1 as hidden_state contains the previous information

				decoder_raw_output.append(output)
			decoder_raw_output = torch.cat(decoder_raw_output, dim=1)

		return decoder_raw_output

	def step(self, input, hidden_state):
		"""
		A single forward step to predict the next token
		:param input: torch.Tensor with shape [batch_size, seq_len]
		:param hidden_state: torch.Tensor with shape [num_layers, batch_size, hidden_size]
		:return: output, hidden_state
		"""
		embedding = self.embedding(input)
		output, hidden_state = self.gru(embedding, hidden_state)
		output = self.output_layer(output)
		# output = self.softmax(output)  # not needed using nn.CrossEntropyLoss

		return output, hidden_state

	def predict(self, shared_hidden_state, gt_tcr_seq):
		"""
		Only feed in own prediction, don't use teacher forcing here
		:param shared_hidden_state:
		:param gt_tcr_seq:
		:return:
		"""
		decoder_raw_output = []
		batch_size, seq_len = gt_tcr_seq.shape

		hidden_state = self.hidden_state_layer(shared_hidden_state)  # shape=[batch_size, num_layers*hidden_size]
		hidden_state = hidden_state.reshape(batch_size, self.num_layers, self.hidden_size)  # shape=[batch_size, num_layers, hidden_size]
		hidden_state = hidden_state.permute(1, 0, 2)  # switch batch and layer dim, shape=[num_layers, batch_size, hidden_size]
		hidden_state = hidden_state.contiguous()  # permute returns a non-contiguous Tensor

		# Initialize first input and hidden state
		input = gt_tcr_seq[:, 0].unsqueeze(1)  # start symbol
		for step in range(seq_len-1):
			output, hidden_state = self.step(input, hidden_state)
			input = torch.argmax(output, dim=-1)  # only needs to be seq_len=1 as hidden_state contains the previous information

			decoder_raw_output.append(output)
		decoder_raw_output = torch.cat(decoder_raw_output, dim=1)

		return decoder_raw_output
