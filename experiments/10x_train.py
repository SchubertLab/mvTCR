import scanpy as sc
import yaml
import tcr_embedding as tcr
import tcr_embedding.evaluation.WrapperFunctions as Wrapper
import tcr_embedding.evaluation.Imputation as Imputation

PATH_BASE = '../'
PATH_SAVE = '../saved_models/'


def load_data():
    path_data = PATH_BASE + 'data/10x_CD8TC/v5_train_val_test.h5ad'
    data_full = sc.read_h5ad(path_data)
    data_full = data_full[data_full.obs['set'] != 'test']
    return data_full


def load_config(file_name):
    path_configs = PATH_BASE + 'config/' + file_name + '.yaml'
    with open(path_configs) as config_file:
        config = yaml.load(config_file)
    return config


def create_model(datas):
    model = tcr.models.joint_model.JointModel(
        adatas=[datas],  # adatas containing gene expression and TCR-seq
        names=['10x'],
        aa_to_id=data_tc.uns['aa_to_id'],  # dict {aa_char: id}
        seq_model_arch=params['seq_model_arch'],  # seq model architecture
        seq_model_hyperparams=params['seq_model_hyperparams'],  # dict of seq model hyperparameters
        scRNA_model_arch=params['scRNA_model_arch'],
        scRNA_model_hyperparams=params['scRNA_model_hyperparams'],
        zdim=params['zdim'],  # zdim
        hdim=params['hdim'],  # hidden dimension of scRNA and seq encoders
        activation=params['activation'],  # activation function of autoencoder hidden layers
        dropout=params['dropout'],
        batch_norm=params['batch_norm'],
        shared_hidden=params['shared_hidden'],  # hidden layers of shared encoder / decoder
        gene_layers=[],  # [] or list of str for layer keys of each dataset
        seq_keys=[]  # [] or list of str for seq keys of each dataset
    )
    return model


def train_model(model):
    model.train(
        experiment_name='test',
        n_iters=None,
        n_epochs=1000,
        batch_size=params['batch_size'],
        lr=params['lr'],
        losses=params['losses'],  # list of losses for each modality: losses[0] := scRNA, losses[1] := TCR
        loss_weights=params['loss_weights'],  # [] or list of floats storing weighting of loss in order [scRNA, TCR, KLD]
        val_split='set',  # float or str, if float: split is determined automatically, if str: used as key for train-val column
        metadata=[],
        validate_every=1,
        save_every=25,
        num_workers=0,
        early_stop=20,
        save_path=PATH_SAVE,
        device=None,
        comet=None
    )


def evaluate_model(model, data):
    model.load(PATH_SAVE + '/test_best_model.pt')
    embedding_function = Wrapper.get_model_prediction_function(model, batch_size=512)
    eval_score = Imputation.run_imputation_evaluation(data, embedding_function, query_source='val')
    return eval_score


data_tc = load_data()
params = load_config('transformer')
model_test = create_model(data_tc)
train_model(model_test)
score = evaluate_model(model_test, data_tc)
print(score)
