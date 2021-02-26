import numpy as np


def aa_encoding(adata, read_col, ohe_col=None, label_col=None, length_col=None, pad=False, aa_to_id=None, start_end_symbol=True):
	"""
	Encoding of protein or nucleotide sequence inplace, either one-hot-encoded or as index labels and/or one-hot-encoding
	:param adata: adata file
	:param read_col: str column containing sequence
	:param ohe_col: None or str, if str column to write one-hot-encoded sequence into
	:param label_col: None or str, if str then write labels as index to this column
	:param length_col: str column None or str, if str write sequence length into this column
	:param pad: bool or int value, if int value then the sequence will be pad to this value,
				if True then pad_len will be determined by taking the longest sequence length in adata
	:param aa_to_id: None or dict, None will create a dict in this code, dict should contain {aa: index}
	:param start_end_symbol: bool, add a start '<' and end '>' symbol to each sequence
	:return:
	"""
	if label_col is None and ohe_col is None:
		raise AssertionError('Specify at least one column to write: ohe_col or label_col')

	if start_end_symbol:
		adata.obs[read_col] = '<' + adata.obs[read_col].astype('str') + '>'
		if type(pad) is int:
			pad += 2

	if length_col:
		adata.obs[length_col] = adata.obs[read_col].str.len()

	# Padding if specified
	if type(pad) is int:
		sequence_col = adata.obs[read_col].str.ljust(pad, '_')
	elif pad:
		pad_len = adata.obs[read_col].str.len().max()
		sequence_col = adata.obs[read_col].str.ljust(pad_len, '_')
	else:
		sequence_col = adata.obs[read_col]

	# tokenize each character, i.e. create list of characters
	aa_tokens = sequence_col.apply(lambda x: list(x))

	# dict containing aa name as key and token-id as value
	if aa_to_id is None:
		unique_aa_tokens = sorted(set([x for sublist in aa_tokens for x in sublist]))
		aa_to_id = {aa: id_ for id_, aa in enumerate(unique_aa_tokens)}

	# convert aa to token_id (i.e. unique integer for each aa)
	token_ids = [[aa_to_id[token] for token in aa_token] for aa_token in aa_tokens]

	# convert token_ids to one-hot
	if ohe_col is not None:
		one_hot = [np.zeros((len(aa_token), len(aa_to_id))) for aa_token in aa_tokens]
		for x_seq, token_id_seq in zip(one_hot, token_ids):
			for x, token_id in zip(x_seq, token_id_seq):
				x[token_id] = 1.0
		adata.obs[ohe_col] = one_hot

	# If specified write label as index sequence
	if label_col is not None:
		token_ids = [np.array(token_id) for token_id in token_ids]
		adata.obs[label_col] = token_ids

	return aa_to_id
