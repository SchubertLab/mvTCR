import operator
from tcr_embedding.evaluation.WrapperFunctions import get_model_prediction_function
from tcr_embedding.evaluation.kNN import run_knn_within_set_evaluation


def report_pseudo_metric(adata, model, optimization_mode_params, batch_size, epoch, comet):
    """
    Calculate a pseudo metric based on kNN of multiple meta information
    :param epoch:
    :return:
    """
    test_embedding_func = get_model_prediction_function(model, batch_size=batch_size, do_adata=True,
                                                        metadata=optimization_mode_params['prediction_labels'])
    try:
        summary = run_knn_within_set_evaluation(adata, test_embedding_func,
                                                optimization_mode_params['prediction_labels'], subset='val')
        summary['pseudo_metric'] = sum(summary.values())
    except Exception as e:
        print(e)
        print('Error in kNN')
        return

    if comet is not None:
        comet.log_metrics(summary, epoch=epoch)

    return summary['pseudo_metric'], operator.gt
