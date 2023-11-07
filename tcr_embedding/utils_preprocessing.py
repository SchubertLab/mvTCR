import logging
import numpy as np
import scirpy as ir
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit


class Preprocessing():

	@staticmethod
	def check_if_valid_adata(adata):
		valid_adata = True
		#expression matrix data checks
		if adata.X.min() < 0:
			logging.warning('Invalid entries in expression matrix. (Negative values in adata.X)')
			valid_adata = False
		#check if data is normalized
		if False in adata.var.highly_variable.unique():
			if np.std(adata.X.sum(axis=1)) >= 1:
				logging.warning(f'Looks like your data is not normalized with counts per target_sum.\nStd of cells total sum of genes: {np.std(adata.X.sum(axis=1))}. In case of other normalizations, this warning might be false.')
		else:
			logging.warning('Only highly-variable genes found in adata. Make sure they are properly normalized before proceeding!')
		#log1p
		if adata.X.max() > np.log1p(10000):
			logging.warning('Looks like your data is not log1p transformed! Either use log1p or other variance stabilizing functions.')
			#valid_adata = False
		#highly var genes
		if adata.shape[1] > 5000:
			logging.warning('The data contains more than 5000 genes. Please make sure you only keep the highly-varibale ones.')
			#valid_adata = False
		elif adata.shape[1] < 500:
			logging.warning('The data contains less than 500 genes. Please make sure you have a sufficient amount of genes')
			#valid_adata = False
		#scirpy
		if 'airr' not in adata.obsm:
			logging.warning('No AIRR annotation found in adata.obsm. Please use scirpy annotation convention.')
			valid_adata = False
		#if 'True' in adata.obs['multi_chain'].unique():
		#	logging.warning('There are entries with multiple chains. Make sure to remove them.')
		junction_aa = ir.get.airr(adata, "junction_aa").copy()
		junction_aa_nans = np.array([
			junction_aa.VJ_1_junction_aa.isna().sum(),
			junction_aa.VJ_2_junction_aa.isna().sum(),
			junction_aa.VDJ_1_junction_aa.isna().sum(),
			junction_aa.VDJ_2_junction_aa.isna().sum(),
		])
		if junction_aa_nans[:1].sum() != 0:
			logging.warning(f'Not all cells have sufficient AIRR information. You need at least one sequence for alpha and beta chain respectively.\n {junction_aa_nans[0]} VJ nans, {junction_aa_nans[2]} VDJ nans.')
			valid_adata = False
		#logging.warn(f'Found {adata.shape[0] - junction_aa_nans[1]} VJ and {adata.shape[0] - junction_aa_nans[3]} VDJ cells with more than one sequence for a chain.')

		#return valid_adata
		return valid_adata

	@staticmethod
	def encode_clonotypes(adata, key_added='clonotype'):
		"""
		Encode the clonotypes with scirpy
		:param adata: adata object
		"""
		ir.tl.chain_qc(adata)
		ir.pp.ir_dist(adata)
		ir.tl.define_clonotypes(adata, key_added=key_added, receptor_arms='all', dual_ir='primary_only')

	@staticmethod
	def encode_tcr(adata, airr_name='junction_aa', alpha_label_key='alpha_seq', alpha_length_key='alpha_len', beta_label_key='beta_seq', beta_length_key='beta_len', aa_encoding_dict=None, pad=None, start_end_symbol=False):
		"""
		Encodes the CDR3 alpha and CDR3 beta chain into numerical values
		:param adata: adata object
		:param airr_name: str, name in awkward array where airr amino acid sequences are stored
		:param alpha_label_key: str, name in adata.obsm where the encoded sequences are stored
		:param alpha_length_key: str, name in adata.obsm where the length of the unpadded sequences are stored
		:param beta_label_key: str, name in adata.obsm
		:param beta_length_key: str, name in adata.obsm
		:param pad: int, amount of position to pad the sequence to
		:return: stores the numeric embedding to adata.obsm['alpha_seq'] and adata.obsm['beta_seq']
		"""
		junction_aa = ir.get.airr(adata, airr_name)
		if not pad:
			#TODO this correct? alpha VJ; beta VDJ
			len_alpha= junction_aa.VJ_1_junction_aa.str.len().max()
			len_beta = junction_aa.VDJ_1_junction_aa.str.len().max()
			pad = max(len_alpha, len_beta)

		if not aa_encoding_dict:
			aa_to_id = {'_': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
						'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '+': 21,
						'<': 22, '>': 23}
			
		if alpha_length_key:
			adata.obsm[alpha_length_key] = junction_aa.VJ_1_junction_aa.str.len().values
		if beta_length_key:
			adata.obsm[beta_length_key] = junction_aa.VDJ_1_junction_aa.str.len().values

	
		if start_end_symbol:
			adata.obsm['start_end_symbol_seqs'] = '<' + junction_aa[['VJ_1_junction_aa', 'VDJ_1_junction_aa']].astype('str') + '>'
			if type(pad) is not bool:
				pad += 2
	

		adata.uns['aa_to_id'] = aa_to_id

		Preprocessing._aa_encoding(adata, junction_aa.VJ_1_junction_aa, ohe_col=None, label_col=beta_label_key, pad=pad, aa_to_id=None)
		Preprocessing._aa_encoding(adata, junction_aa.VDJ_1_junction_aa, ohe_col=None, label_col=alpha_label_key, pad=pad, aa_to_id=None)

	@staticmethod
	def _aa_encoding(adata, seq_list, ohe_col=None, label_col=None, pad=False, aa_to_id=None):
		"""
		Encoding of protein or nucleotide sequence inplace, either one-hot-encoded or as index labels and/or one-hot-encoding
		:param adata: adata file
		:param seq_list: pd.series (or list or np array?) with amino acid sequences of TCA or TCB
		:param ohe_col: None or str, if str column to write one-hot-encoded sequence into
		:param label_col: None or str, if str then write labels as index to this column
		:param pad: bool or int value, if int value then the sequence will be pad to this value,
					if True then pad_len will be determined by taking the longest sequence length in adata
		:param aa_to_id: None or dict, None will create a dict in this code, dict should contain {aa: index}
		:return:
		"""
		if label_col is None and ohe_col is None:
			raise AssertionError('Specify at least one column to write: ohe_col or label_col')
		
		# Padding if specified
		if type(pad) is not bool:
			seq_list = seq_list.str.ljust(pad, '_')
		elif pad:
			pad_len = seq_list.str.len().max()
			seq_list = seq_list.str.ljust(pad_len, '_')

		# tokenize each character, i.e. create list of characters
		aa_tokens = seq_list.apply(lambda x: list(x))

		# dict containing aa name as key and token-id as value
		if aa_to_id is None:
			try:
				aa_to_id = adata.uns['aa_to_id']
			except:
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


	@staticmethod
	def encode_conditional_var(adata, column_id):
		"""
		One-hot encode additional features in adata.obs. Column data will be transformed and the categories saved in adata.uns
		:param adata: adata object
		:param column_id: identifier for the specific column in the adata.obs for ohe
		"""
		enc = OneHotEncoder(sparse=False)
		enc.fit(adata.obs[column_id].to_numpy().reshape(-1, 1))
		adata.obsm[column_id] = enc.transform(adata.obs[column_id].to_numpy().reshape(-1, 1))
		adata.uns[column_id + "_enc"] = enc.categories_

	@staticmethod
	def group_shuffle_split(adata, group_col, val_split, random_seed=42):
		'''
		Grou-shuffle-split
		:param adata_tmp: adata object
		:param group_col: str key for the column containing the groups to be kept in the same set
		:param val_split: float defining size of val split
		'''
		groups = adata.obs[group_col]
		splitter = GroupShuffleSplit(test_size=val_split, n_splits=5, random_state=random_seed)

		best_value = 1
		train, val = None, None
		for train_tmp, val_tmp in splitter.split(adata, groups=groups):
			split_value = abs(len(val_tmp) / len(adata) - val_split)
			if split_value < best_value:
				train = train_tmp
				val = val_tmp
				best_value = split_value

		train = adata[train]
		val = adata[val]
		return train, val

	@staticmethod
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
		:param val_split: float defining size of val split
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
	
	@staticmethod
	def preprocessing_pipeline(adata, clonotype_key_added, column_cdr3a, column_cdr3b, cond_vars, val_split, group_col, random_seed=42):
		
		if Preprocessing.check_if_valid_adata(adata):
			Preprocessing.encode_clonotypes(adata, key_added=clonotype_key_added)
			Preprocessing.encode_tcr(adata, column_cdr3a, column_cdr3b)

			for var in cond_vars:
				Preprocessing.encode_conditional_var(adata, var)
			
			train, val = Preprocessing.group_shuffle_split(adata, group_col, val_split, random_seed)
			adata.obs['set'] = 'train'
			adata.obs.loc[val.obs.index, 'set'] = 'val'


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
