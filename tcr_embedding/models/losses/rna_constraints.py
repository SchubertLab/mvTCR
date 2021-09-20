"""
When training the joint model, the latent space is heavily biased towards the TCR space.
Therefore, we experiment with various ways of constraining the joint space to be more similar to the gene expression.
"""
import tcr_embedding.models.losses.kld as kld
import tcr_embedding.models.losses.mmd as mmd
import tcr_embedding.models.losses.mse_latent as mse


def init_constraints_params(constraint_mode, mode_params):
    """
    Initialise the mode params for the different modes
    :param constraint_mode: str, name of the constraint modes
    :param mode_params: dict or None, providing mode specific parameters
    :return:
    """
    modes = {
        'RNA_KLD': default_rna_kld,
        'RNA_MMD': default_rna_mmd,
        'RNA_MEAN': default_rna_mean,
    }
    if constraint_mode is None:
        get_setting = default_empty
    else:
        get_setting = modes[constraint_mode]
    default_settings = get_setting()
    if mode_params is None:
        mode_params = {}
    for name, value in default_settings.items():
        if name not in mode_params:
            mode_params[name] = value
    return mode_params


def default_rna_kld():
    default_settings = {
        'loss': kld.KLD(),
        'do_annealing': False,
    }
    return default_settings


def default_rna_mmd():
    default_settings = {
        'loss': mmd.MMD(),
        'do_annealing': False,
    }
    return default_settings


def default_rna_mean():
    default_settings = {
        'loss': mse.MseLatent(),
        'do_annealing': False,
    }
    return default_settings


def default_empty():
    return {}
