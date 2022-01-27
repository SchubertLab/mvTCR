import scanpy as sc
import tcr_embedding.evaluation.Metrics as Metrics


def evaluate_pertubation(data_val, prediction, per_column, pertubation, indicator='pre', column_cluster=None):
    """
    Evaluate pertubation prediction by Pearson squared for all genes and top100, overall and per classes
    :param data_val: adata object containing the ground truth validation data
    :param prediction: adata object containing the predicted latent space
    :param per_column: column from which the classes are used
    :param pertubation: column of pertubation
    :param indicator: value in pertubation indicating pre state
    :param column_cluster: column indicating the clusters for which the top 100 DEG are determined
    :return: dictionary containing a summary of the performance
    """
    ground_truth = data_val[data_val.obs[pertubation] != indicator].copy()

    # evaluate over all genes and cell types
    summary = {'all_genes': Metrics.get_square_pearson(ground_truth, prediction)}

    # evaluate by column
    if per_column is not None:
        groups = data_val.obs[per_column].unique()
        summary[f'per_{per_column}'] = evaluate_per_column(ground_truth, prediction, per_column, groups)

    # evaluate over top 100 genes
    groups = [group for group, amount in data_val.obs[column_cluster].value_counts().iteritems() if amount > 1]
    if column_cluster is None or len(data_val.obs[column_cluster].unique()) <= 1 or len(groups) <= 1:
        return summary
    sc.tl.rank_genes_groups(data_val, column_cluster, groups=groups, n_genes=100, method='wilcoxon')
    top_100_genes = data_val.uns['rank_genes_groups']['names']
    top_100_genes = [gene for group in top_100_genes for gene in group]
    ground_truth = ground_truth[:, top_100_genes]
    prediction = prediction[:, top_100_genes]

    # evaluate over top 100 genes per cell type
    summary['top_100_genes'] = Metrics.get_square_pearson(ground_truth, prediction)

    # evaluate by column
    if per_column is not None:
        groups = data_val.obs[per_column].unique()
        summary[f'per_{per_column}_top_100'] = evaluate_per_column(ground_truth, prediction, per_column, groups)
    return summary


def evaluate_per_column(ground_truth, prediction, per_column, groups):
    """
    Evaluate scores for subcategories of the cells
    :param ground_truth: adata object containing the real transcriptomic data
    :param prediction: adata object containing the predicted transcriptomic data
    :param per_column: column for grouping
    :param groups: possible entries in this column
    :return: dict {group_name: performance_summary}
    """
    sub_summary = {}
    for entry in groups:
        if entry not in ground_truth.obs[per_column].unique() or entry not in prediction.obs[per_column].unique():
            continue
        sub_prediction = prediction[prediction.obs[per_column] == entry]
        sub_ground_truth = ground_truth[ground_truth.obs[per_column] == entry]
        sub_summary[entry] = Metrics.get_square_pearson(sub_ground_truth, sub_prediction)
    return sub_summary
