import os

configs = {
    'version': '1', # version id for easy identification
    'epochs': 350,
    'model_path': os.getcwd() + '/tmp_models/',
    'experiment': 'DBLP_author_clf',
    'embed_dimen': [128, 128, 128, 128],
    'data_path': os.getcwd() + '/data/',
    'A_n': 4057,
    'P_n': 14376,
    'V_n': 20,
    'T_n': 8920,
    'rank': 4,
    'gpu': True,
    'lr': 0.001,
    'batch_sz': 200,
    'label_file': "author_label_remaster.txt"
}