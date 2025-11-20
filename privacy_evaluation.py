import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import pandas as pd
from syntheval.syntheval import SynthEval
import json
import warnings
warnings.filterwarnings("ignore")

def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                # syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]
                syn_df[i] = syn_target

    return syn_df

def calculate_dcr(real_num_train, real_num_test, real_cat_train, real_cat_test, real_target_train, real_target_test, syn_num_train, syn_cat_train, syn_target_train):
    
    real_cat_train = np.hstack([real_cat_train, real_target_train.reshape(-1, 1)])
    real_cat_test = np.hstack([real_cat_test, real_target_test.reshape(-1, 1)])
    syn_cat_train = np.hstack([syn_cat_train, syn_target_train.reshape(-1, 1)])
    
    encoder = OneHotEncoder()
    combined_data = np.vstack([real_cat_train, real_cat_test])
    encoder.fit(combined_data)
    
    real_cat_train_ohe = encoder.transform(real_cat_train).toarray()
    real_cat_test_ohe = encoder.transform(real_cat_test).toarray()
    syn_cat_train_ohe = encoder.transform(syn_cat_train).toarray()
    
    num_ranges = []
    for i in range(len(real_num_train[0])):
        num_ranges.append(real_num_train[:, i].max() - real_num_train[:, i].min())
    num_ranges = np.array(num_ranges)
    
    real_num_train = real_num_train / num_ranges
    syn_num_train = syn_num_train / num_ranges
    real_num_test = real_num_test / num_ranges
    
    real_num_train = np.nan_to_num(real_num_train)
    real_num_test = np.nan_to_num(real_num_test)
    syn_num_train = np.nan_to_num(syn_num_train)
    
    real_data_np = np.concatenate([real_num_train, real_cat_train_ohe], axis=1)
    syn_data_np = np.concatenate([syn_num_train, syn_cat_train_ohe], axis=1)
    test_data_np = np.concatenate([real_num_test, real_cat_test_ohe], axis=1)
    real_data_np = np.random.permutation(real_data_np)[:test_data_np.shape[0]]

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    real_data_th = torch.tensor(real_data_np).to(device)
    syn_data_th = torch.tensor(syn_data_np).to(device)  
    test_data_th = torch.tensor(test_data_np).to(device)

    dcrs_real = []
    dcrs_test = []
    batch_size = 100

    for i in range((syn_data_th.shape[0] // batch_size) + 1):
        if i != (syn_data_th.shape[0] // batch_size):
            batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
        else:
            batch_syn_data_th = syn_data_th[i*batch_size:]
            
        dcr_real = (batch_syn_data_th[:, None] - real_data_th).abs().sum(dim = 2).min(dim = 1).values
        dcr_test = (batch_syn_data_th[:, None] - test_data_th).abs().sum(dim = 2).min(dim = 1).values
        dcrs_real.append(dcr_real)
        dcrs_test.append(dcr_test)
        
    dcrs_real = torch.cat(dcrs_real)
    dcrs_test = torch.cat(dcrs_test)
    
    
    score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]
    
    return score
    
def syntheval_main(real_num_train, real_num_test, real_cat_train, real_cat_test, 
                real_target_train, real_target_test, 
                syn_num_train, syn_cat_train, syn_target_train, info):
    
    df_train = recover_data(real_num_train, real_cat_train, real_target_train, info)
    df_test = recover_data(real_num_test, real_cat_test, real_target_test, info)
    df_syn = recover_data(syn_num_train, syn_cat_train, syn_target_train, info)
    
    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    
    df_train.rename(columns = idx_name_mapping, inplace=True)
    df_test.rename(columns = idx_name_mapping, inplace=True)
    df_syn.rename(columns = idx_name_mapping, inplace=True)
    
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]
    col_name = info["column_names"]
    categorical_columns = np.array(col_name)[cat_col_idx].tolist()
    target_column = str(np.array(col_name)[target_col_idx[0]])
    categorical_columns.append(target_column)

    print("Running SynthEval...")
    
    S = SynthEval(df_train, holdout_dataframe=df_test, cat_cols=categorical_columns, verbose=True)
    
    score = S.evaluate(df_syn, target_column, "custom_eval")
    
if __name__ == "__main__":
    dataset_name = "CICIoT2023" # "CIC-IDS2018", "UNSW-NB15", or "CICIoT2023"
    baseline_method = "TabSyn" # "STaSy", "TabDDPM", or "TabSyn" 
    print(f"Evaluating dataset: {dataset_name} with baseline method: {baseline_method}")
    print("Loading data...")
    
    real_train_num = np.load(f"./real_data/{dataset_name}/train_data_num.npy").astype(np.float64)
    real_train_cat = np.load(f"./real_data/{dataset_name}/train_data_cat.npy").astype(np.int32)
    real_train_target = np.load(f"./real_data/{dataset_name}/train_data_target.npy").astype('int')

    real_test_num = np.load(f"./real_data/{dataset_name}/test_data_num.npy").astype(np.float64)
    real_test_cat = np.load(f"./real_data/{dataset_name}/test_data_cat.npy").astype(np.int32)
    real_test_target = np.load(f"./real_data/{dataset_name}/test_data_target.npy").astype('int')

    synth_data_num = np.load(f"./synth_data/{dataset_name}/{baseline_method}/Syn_num.npy").astype(np.float64)
    synth_data_cat = np.load(f"./synth_data/{dataset_name}/{baseline_method}/Syn_cat.npy").astype(np.int32)
    synth_data_target = np.load(f"./synth_data/{dataset_name}/{baseline_method}/Syn_target.npy").astype('int')

    refined_data_num = np.load(f"./refined_data/{dataset_name}/{baseline_method}/Syn_num.npy").astype(np.float64)
    refined_data_cat = np.load(f"./refined_data/{dataset_name}/{baseline_method}/Syn_cat.npy").astype(np.int32)
    refined_data_target = np.load(f"./refined_data/{dataset_name}/{baseline_method}/Syn_target.npy").astype('int')
    
    print("Calculating DCR score...")
    dcr_synth = calculate_dcr(real_train_num, real_test_num, real_train_cat, real_test_cat, real_train_target, real_test_target,
                              synth_data_num, synth_data_cat, synth_data_target)
    dcr_refined = calculate_dcr(real_train_num, real_test_num, real_train_cat, real_test_cat, real_train_target, real_test_target,
                              refined_data_num, refined_data_cat, refined_data_target)
    print(f"DCR Score of Synthesized Data: {dcr_synth:.4f}")
    print(f"DCR Score of Refined Data: {dcr_refined:.4f}")
    
    print("Run SynthEval...")
    info_path = f"./real_data/{dataset_name}/info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    dcr_synth = syntheval_main(real_train_num, real_test_num, real_train_cat, real_test_cat, real_train_target, real_test_target,
                              synth_data_num, synth_data_cat, synth_data_target, info)
    dcr_refined = syntheval_main(real_train_num, real_test_num, real_train_cat, real_test_cat, real_train_target, real_test_target,
                              refined_data_num, refined_data_cat, refined_data_target, info)
    
    