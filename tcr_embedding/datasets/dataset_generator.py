def create_datasets(self, adata, val_split, metadata=[], train_masks=None,
                    label_key=None):
    """
    # todo move to dataset generator
    Create torch Dataset, see above for the input
    :param val_split:
    :param metadata:
    :param train_masks: None or list of train_masks: if None new train_masks are created, else the train_masks are used, useful for continuing training
    :return: train_dataset, val_dataset, train_masks (for continuing training)
    """
    dataset_names_train = []
    dataset_names_val = []
    scRNA_datas_train = []
    scRNA_datas_val = []
    seq_datas_train = []
    seq_datas_val = []
    seq_len_train = []
    seq_len_val = []
    adatas_train = {}
    adatas_val = {}
    index_train = []
    index_val = []
    metadata_train = []
    metadata_val = []
    conditional_train = []
    conditional_val = []

    if train_masks is None:
        masks_exist = False
        train_masks = {}
    else:
        masks_exist = True

    # Iterates through datasets with corresponding dataset name, scRNA layer and TCR column key
    # Splits everything into train and val
    for i, (adata, layer, seq_key) in enumerate(zip(adata, layers, seq_keys)):
        if masks_exist:
            train_mask = train_masks[name]
        else:
            if type(val_split) == str:
                train_mask = (adata.obs[val_split] == 'train').values
            else:
                # Create train mask for each dataset separately
                num_samples = adata.X.shape[0] if layer is None else len(adata.layers[layer].shape[0])
                train_mask = np.zeros(num_samples, dtype=np.bool)
                train_size = int(num_samples * (1 - val_split))
                if val_split != 0:
                    train_mask[:train_size] = 1
                else:
                    train_mask[:] = 1
                np.random.shuffle(train_mask)
            train_masks[name] = train_mask

        # Save dataset splits
        scRNA_datas_train.append(adata.X[train_mask] if layer is None else adata.layers[layer][train_mask])
        scRNA_datas_val.append(adata.X[~train_mask] if layer is None else adata.layers[layer][~train_mask])

        seq_datas_train.append(adata.obsm[seq_key][train_mask])
        seq_datas_val.append(adata.obsm[seq_key][~train_mask])

        seq_len_train += adata.obs['seq_len'][train_mask].to_list()
        seq_len_val += adata.obs['seq_len'][~train_mask].to_list()

        adatas_train[name] = adata[train_mask]
        adatas_val[name] = adata[~train_mask]

        dataset_names_train += [name] * adata[train_mask].shape[0]
        dataset_names_val += [name] * adata[~train_mask].shape[0]

        index_train += adata[train_mask].obs.index.to_list()
        index_val += adata[~train_mask].obs.index.to_list()

        metadata_train.append(adata.obs[metadata][train_mask].values)
        metadata_val.append(adata.obs[metadata][~train_mask].values)

        if self.conditional is not None:
            conditional_train.append(adata.obsm[self.conditional][train_mask])
            conditional_val.append(adata.obsm[self.conditional][~train_mask])
        else:
            conditional_train = None
            conditional_val = None

    train_dataset = TCRDataset(scRNA_datas_train, seq_datas_train, seq_len_train, adatas_train, dataset_names_train,
                               index_train, metadata_train, labels=None, conditional=conditional_train)
    val_dataset = TCRDataset(scRNA_datas_val, seq_datas_val, seq_len_val, adatas_val, dataset_names_val, index_val,
                             metadata_val, labels=None, conditional=conditional_val)

    return train_dataset, val_dataset, train_masks


print('Create Dataloader')
# Initialize dataloader
train_datasets, val_datasets, self.train_masks = self.create_datasets(self.adata, self.seq_keys,
                                                                      val_split, metadata,
                                                                      label_key=self.label_key)

if balanced_sampling is None:
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, worker_init_fn=self.seed_worker)
else:
    sampling_weights = self.calculate_sampling_weights(self.adatas, self.train_masks, self.names,
                                                       class_column=balanced_sampling)
    sampler = WeightedRandomSampler(weights=sampling_weights, num_samples=len(sampling_weights),
                                    replacement=True)
    # shuffle is mutually exclusive to sampler, but sampler is anyway shuffled
    if comet is not None:
        comet.log_parameters({'sampling_weight_min': sampling_weights.min(),
                              'sampling_weight_max': sampling_weights.max()})
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=False,
                                  sampler=sampler, worker_init_fn=self.seed_worker)
val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
print('Dataloader created')