# todo major reconstruction

adata = adata[adata.obs['set'] != 'test']  # This needs to be inside the function, ray can't deal with it outside
adata.obs['binding_name'] = adata.obs['binding_name'].astype(str)
raw_adata = adata
if args.donor != 'all':
	adata = adata[adata.obs['donor'] == 'donor_'+args.donor]
if args.without_non_binder:
	# Filter out no_data and rare classes, but only from training set
	adata = adata[~((adata.obs['set'] == 'train') & ~(adata.obs['binding_name'].isin(tcr.constants.HIGH_COUNT_ANTIGENS)))]
	experiment.log_parameter('without_non_binder', args.without_non_binder)
experiment.log_parameter('donors', adata.obs['donor'].unique().astype(str))

if os.path.exists(os.path.join(save_path, f'{name}_best_rec_model.pt')):
	print('kNN for best reconstruction loss model')
	# Evaluate Model (best model based on reconstruction loss)
	model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
	# Evaluation excluding no_data aka non-binder
	test_embedding_func = get_model_prediction_function(model, batch_size=params['batch_size'])
	try:
		summary = run_imputation_evaluation(adata, test_embedding_func, query_source='val', use_non_binder=False, use_reduced_binders=True)
	except:
		tune.report(weighted_f1=0.0)
		return
	metrics = summary['knn']
	for antigen, metric in metrics.items():
		if antigen != 'accuracy':
			experiment.log_metrics(metric, prefix='best_recon_'+antigen, step=int(model.epoch*epoch2step), epoch=model.epoch)
		else:
			experiment.log_metric('best_recon_accuracy', metric, step=int(model.epoch*epoch2step), epoch=model.epoch)
	tune.report(weighted_f1=metrics['weighted avg']['f1-score'])

	# Include no_data aka non-binder
	test_embedding_func = get_model_prediction_function(model, batch_size=params['batch_size'])
	try:
		summary = run_imputation_evaluation(raw_adata, test_embedding_func, query_source='val', use_non_binder=True, use_reduced_binders=True)
	except:
		tune.report(weighted_f1=0.0)
		return

	metrics = summary['knn']
	for antigen, metric in metrics.items():
		if antigen != 'accuracy':
			experiment.log_metrics(metric, prefix='with_no_data_best_recon_'+antigen, step=int(model.epoch*epoch2step), epoch=model.epoch)
		else:
			experiment.log_metric('with_no_data_best_recon_accuracy', metric, step=int(model.epoch*epoch2step), epoch=model.epoch)

	# For visualization purpose, we set all rare specificities to no_data
	adata.obs['binding_label'][~adata.obs['binding_name'].isin(tcr.constants.HIGH_COUNT_ANTIGENS)] = -1
	adata.obs['binding_name'][~adata.obs['binding_name'].isin(tcr.constants.HIGH_COUNT_ANTIGENS)] = 'no_data'
	# For visualization purpose, else the scanpy plot script thinks the rare specificities are still there and the colors get skewed
	adata.obs['binding_name'] = adata.obs['binding_name'].astype(str)

	print('UMAP for best f1 score model')
	model.load(os.path.join(save_path, f'{name}_best_knn_model.pt'))
	val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=512, metadata=['binding_name', 'clonotype', 'donor'])
	fig_donor, fig_clonotype, fig_antigen = tcr.utils.plot_umap(val_latent, title=name+'_val_best_f1')
	experiment.log_figure(figure_name=name+'_val_best_f1_donor', figure=fig_donor, step=model.epoch)
	experiment.log_figure(figure_name=name+'_val_best_f1_clonotype', figure=fig_clonotype, step=model.epoch)
	experiment.log_figure(figure_name=name+'_val_best_f1_antigen', figure=fig_antigen, step=model.epoch)

	print('UMAP for best reconstruction loss model')
	model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
	val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=512, metadata=['binding_name', 'clonotype', 'donor'])
	fig_donor, fig_clonotype, fig_antigen = tcr.utils.plot_umap(val_latent, title=name + '_val_best_recon')
	experiment.log_figure(figure_name=name + '_val_best_recon_donor', figure=fig_donor, step=model.epoch)
	experiment.log_figure(figure_name=name + '_val_best_recon_clonotype', figure=fig_clonotype, step=model.epoch)
	experiment.log_figure(figure_name=name + '_val_best_recon_antigen', figure=fig_antigen, step=model.epoch)

	# Filter out non-binders and plot UMAP
	adata = adata[adata.obs['binding_name'] != 'no_data']

	print('UMAP for best f1 score model')
	model.load(os.path.join(save_path, f'{name}_best_knn_model.pt'))
	val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=512, metadata=['binding_name', 'clonotype', 'donor'])
	fig_donor, fig_clonotype, fig_antigen = tcr.utils.plot_umap(val_latent, title=name+'_val_best_f1_without_no_data')
	experiment.log_figure(figure_name=name+'_val_best_f1_donor_without_no_data', figure=fig_donor, step=model.epoch)
	experiment.log_figure(figure_name=name+'_val_best_f1_clonotype_without_no_data', figure=fig_clonotype, step=model.epoch)
	experiment.log_figure(figure_name=name+'_val_best_f1_antigen_without_no_data', figure=fig_antigen, step=model.epoch)

	print('UMAP for best reconstruction loss model')
	model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))
	val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=512, metadata=['binding_name', 'clonotype', 'donor'])
	fig_donor, fig_clonotype, fig_antigen = tcr.utils.plot_umap(val_latent, title=name + '_val_best_recon_without_no_data')
	experiment.log_figure(figure_name=name + '_val_best_recon_donor_without_no_data', figure=fig_donor, step=model.epoch)
	experiment.log_figure(figure_name=name + '_val_best_recon_clonotype_without_no_data', figure=fig_clonotype, step=model.epoch)
	experiment.log_figure(figure_name=name + '_val_best_recon_antigen_without_no_data', figure=fig_antigen, step=model.epoch)

else:
	print('There is no best reconstruction model saved')
	tune.report(weighted_f1=0.0)
