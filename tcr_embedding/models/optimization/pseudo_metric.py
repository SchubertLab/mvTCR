import operator
from tcr_embedding.evaluation.WrapperFunctions import get_model_prediction_function
from tcr_embedding.evaluation.kNN import run_knn_within_set_evaluation


def report_pseudo_metric(adata, model, optimization_mode_params, epoch, comet):
    """
    Calculate a pseudo metric based on kNN of multiple meta information
    :param epoch:
    :return:
    """
    prediction_label = optimization_mode_params['prediction_labels']

    labels = prediction_label.copy()
    if isinstance(prediction_label, dict):
        labels = list(prediction_label.keys())

    test_embedding_func = get_model_prediction_function(model, do_adata=True, metadata=labels)
    summary = run_knn_within_set_evaluation(adata, test_embedding_func, labels, subset='val')

    if isinstance(prediction_label, list):
        summary['pseudo_metric'] = sum(summary.values())
    else:
        summary['pseudo_metric'] = sum([summary[f'weighted_f1_{label}'] * prediction_label[label] for label in prediction_label])

    if comet is not None:
        comet.log_metrics(summary, epoch=epoch)

    return summary['pseudo_metric'], operator.gt
