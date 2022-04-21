import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit


def encode_tcr(adata, column_cdr3a, column_cdr3b, pad):
	"""
	Encodes the CDR3 alpha and CDR3 beta chain into numerical values
	:param adata: adata object
	:param column_cdr3a: column in adata.obs storing the cdr3alpha chain
	:param column_cdr3b: column in adata.obs storing the cdr3beta chain
	:param pad: int, amount of position to pad the sequence to
	:return: stores the numeric embedding to adata.obsm['alpha_seq'] and adata.obsm['beta_seq']
	"""
	aa_to_id = {'_': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
				'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '+': 21,
				'<': 22, '>': 23}
	adata.uns['aa_to_id'] = aa_to_id
	aa_encoding(adata, read_col=column_cdr3b, label_col='beta_seq', length_col='beta_len', pad=pad, aa_to_id=aa_to_id,
				start_end_symbol=False)
	aa_encoding(adata, read_col=column_cdr3a, label_col='alpha_seq', length_col='alpha_len', pad=pad, aa_to_id=aa_to_id,
				start_end_symbol=False)


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
		if type(pad) is not bool:
			pad += 2

	if length_col:
		adata.obs[length_col] = adata.obs[read_col].str.len()

	# Padding if specified
	if type(pad) is not bool:
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
		# adata.obs[ohe_col] = one_hot
		adata.obsm[ohe_col] = np.stack(one_hot)

	# If specified write label as index sequence
	if label_col is not None:
		token_ids = [np.array(token_id) for token_id in token_ids]
		# adata.obs[label_col] = token_ids
		adata.obsm[label_col] = np.stack(token_ids)

	adata.uns['aa_to_id'] = aa_to_id


def stratified_group_shuffle_split(df, stratify_col, group_col, val_split, random_seed=42):
	"""
	https://stackoverflow.com/a/63706321
	Split the dataset into train and test. To create a val set, execute this code twice to first split test+val and test
	and then split the test and val.

	The splitting tries to improve splitting by two properties:
	1) Stratified splitting, so the label distribution is roughly the same in both sets, e.g. antigen specificity
	2) Certain groups are only in one set, e.g. the same clonotypes are only in one set, so the model cannot peak into similar sample during training.

	If there is only one group to a label, the group is defined as training, else as test sample, the model never saw this label before.

	The outcome is not always ideal, i.e. the label distribution may not , as the labels within a group is heterogeneous (e.g. 2 cells from the same clonotype have different antigen labels)
	Also see here for the challenges: https://github.com/scikit-learn/scikit-learn/issues/12076

	:param df: pd.DataFrame containing the data to split
	:param stratify_col: str key for the column containing the classes to be stratified over all sets
	:param group_col: str key for the column containing the groups to be kept in the same set
	"""
	groups = df.groupby(stratify_col)
	all_train = []
	all_test = []
	for group_id, group in tqdm(groups):
		# if a group is already taken in test or train it must stay there
		group = group[~group[group_col].isin(all_train + all_test)]
		# if group is empty
		if group.shape[0] == 0:
			continue

		if len(group) > 1:
			train_inds, test_inds = next(
				GroupShuffleSplit(test_size=val_split, n_splits=1, random_state=random_seed).split(group, groups=group[
					group_col]))
			all_train += group.iloc[train_inds][group_col].tolist()
			all_test += group.iloc[test_inds][group_col].tolist()
		# if there is only one clonotype for this particular label
		else:
			all_train += group[group_col].tolist()

	train = df[df[group_col].isin(all_train)]
	test = df[df[group_col].isin(all_test)]

	return train, test


def group_shuffle_split(adata_tmp, group_col, val_split, random_seed=42):
	groups = adata_tmp.obs[group_col]
	splitter = GroupShuffleSplit(test_size=val_split, n_splits=5, random_state=random_seed)

	best_value = 1
	train, val = None, None
	for train_tmp, val_tmp in splitter.split(adata_tmp, groups=groups):
		split_value = abs(len(val_tmp) / len(adata_tmp) - val_split)
		if split_value < best_value:
			train = train_tmp
			val = val_tmp
			best_value = split_value

	train = adata_tmp[train]
	val = adata_tmp[val]
	return train, val
