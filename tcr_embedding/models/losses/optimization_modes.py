# <- Initialisation for the different optimization modes ->
def init_optimization_mode_params(optimization_mode, mode_params):
    """
    Initialise the mode params for the different modes
    :param optimization_mode: str, name of the optimization modes
    :param mode_params: dict or None, providing mode specific parameters
    :return:
    """
    modes = {
        'Prediction': default_prediction,
        'scGen': default_scgen,
        'Reconstruction': default_reconstruction,
        'PseudoMetric': default_pseudo_metric,
    }
    if optimization_mode is None:
        get_setting = default_reconstruction
    else:
        get_setting = modes[optimization_mode]
    default_settings = get_setting()
    if mode_params is None:
        mode_params = {}
    for name, value in default_settings.items():
        if name not in mode_params:
            mode_params[name] = value
    return mode_params


def default_prediction():
    default_settings = {
        'prediction_column': 'binding_name'
    }
    # todo filter for other datasets
    return default_settings


def default_scgen():
    default_settings = {
    }
    raise NotImplementedError
    # return default_settings


def default_reconstruction():
    default_settings = {
    }
    return default_settings


def default_pseudo_metric():
    default_settings = {
        'weight_cell_type': 1.,
        'weight_clonotype': 1.,  # todo
    }
    return default_settings
