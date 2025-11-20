import numpy as np
from catboost import CatBoostClassifier
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, OrdinalEncoder
import rtdl_num_embeddings
import tabm
import torch
from torch import Tensor
import torch.nn.functional as F
import scipy
from typing import Optional
import math
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

def train_and_evaluate_catboost(X_train, y_train, X_test, y_test, task_type, devices):
    model = CatBoostClassifier(task_type=task_type, devices=devices, logging_level='Silent', allow_writing_files=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1

def train_and_evaluate_tabm(X_train, y_train, X_test, y_test, num_numerical_features, n_classes, device):
    train_idx, val_idx = train_test_split(np.arange(len(y_train)), train_size=0.8, shuffle=True, stratify=y_train)
    
    data_numpy = {
        'train': {'x': X_train[train_idx], 'y': y_train[train_idx]},
        'val': {'x': X_train[val_idx], 'y': y_train[val_idx]},
        'test': {'x': X_test, 'y': y_test}
    }
    
    n_num_features = num_numerical_features
    n_cat_features = X_train.shape[1] - n_num_features
    
    data_numpy_processed = {'train': {}, 'val' : {}, 'test': {}}
    for part in data_numpy:
        data_numpy_processed[part]['x_num'] = data_numpy[part]['x'][:, :n_num_features].astype(np.float32)
        if n_cat_features > 0:
            data_numpy_processed[part]['x_cat'] = data_numpy[part]['x'][:, n_num_features:].astype(np.int64)
        data_numpy_processed[part]['y'] = data_numpy[part]['y'].astype(np.int64)
    
    data_numpy = data_numpy_processed
    x_num_train_numpy = data_numpy['train']['x_num']
    
    preprocessing = QuantileTransformer(
        n_quantiles=max(min(len(x_num_train_numpy) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9
    ).fit(x_num_train_numpy)
    
    for part in data_numpy:
        data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])
        
    Y_train_numpy = data_numpy['train']['y'].copy()
    
    if n_cat_features > 0:
        x_cat_train_numpy = X_train[:, n_num_features:].astype(np.int64)
        x_cat_test_numpy = X_test[:, n_num_features:].astype(np.int64)
        x_cat_combined_numpy = np.vstack([x_cat_train_numpy, x_cat_test_numpy])
        cat_cardinalities = [
            int(np.max(x_cat_combined_numpy[:, i])) + 1 for i in range(n_cat_features)
        ]
    else:
        cat_cardinalities = []
        
    num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(torch.as_tensor(data_numpy['train']['x_num']), n_bins=48),
        d_embedding=16,
        activation=False,
        version='B'
    )
    
    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=n_classes,
        num_embeddings=num_embeddings,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    gradient_clipping_norm: Optional[float] = 1.0
    base_loss_fn = F.cross_entropy
    
    evaluation_mode = torch.inference_mode

    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()} for part in data_numpy
    }
    Y_train = torch.as_tensor(Y_train_numpy, device=device)
    
    share_training_batches = True

    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[part]['x_num'][idx],
                data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
            )
            .float()
        )
        
    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = y_pred.flatten(0, 1)
        if share_training_batches:
            y_true = y_true.repeat_interleave(model.backbone.k)
        else:
            y_true = y_true.flatten(0, 1)
        return base_loss_fn(y_pred, y_true)
    
    @evaluation_mode()
    def evaluate(part: str) -> float:
        model.eval()
        eval_batch_size = 8096
        y_pred: np.ndarray = (
            torch.cat(
                [apply_model(part, idx)
                 for idx in torch.arange(len(data[part]['y']), device=device).split(eval_batch_size)]
            )
            .cpu().numpy()
        )
        
        y_pred = scipy.special.softmax(y_pred, axis=-1).mean(1)
        y_true = data[part]['y'].cpu().numpy()
        score = sklearn.metrics.f1_score(y_true, y_pred.argmax(1), average='weighted')
        return float(score)
    
    n_epochs = 1000
    train_size = len(train_idx)
    batch_size = 256
    patience = 16
    
    epoch = -1
    metrics = {'val': -math.inf, 'test': -math.inf}
    
    best_checkpoint = {'model': deepcopy(model.state_dict()), 'epoch': -1, 'metrics': metrics}
    
    for epoch in range(n_epochs):
        batches = (
            torch.randperm(train_size, device=device).split(batch_size)
            if share_training_batches
            else (
                torch.rand((train_size, model.backbone.k), device=device).argsort(0).split(batch_size, dim=0)
            )
        )
        
        for batch_idx in batches:
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
            loss.backward()
            if gradient_clipping_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)
            optimizer.step()
        
        val_f1 = evaluate('val')
        metrics = {'val': val_f1, 'test': evaluate('test')}
        
        val_score_improved = metrics['val'] > best_checkpoint['metrics']['val']
        
        if val_score_improved:
            best_checkpoint = {'model': deepcopy(model.state_dict()), 'epoch': epoch, 'metrics': metrics}
            remaining_patience = patience
        else:
            remaining_patience -= 1
        
        if remaining_patience < 0:
            print(f"TabM: Early stopping at epoch {epoch}")
            break
        
    model.load_state_dict(best_checkpoint['model'])
    print(f"TabM: Best epoch {best_checkpoint['epoch']} | Val F1: {best_checkpoint['metrics']['val']:.4f} | Test F1: {best_checkpoint['metrics']['test']:.4f}")
    
    @evaluation_mode()
    def get_predictions(part: str) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        eval_batch_size = 8096
        y_pred_raw: np.ndarray = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        y_pred_prob_all_k = scipy.special.softmax(y_pred_raw, axis=-1)
        final_probabilities = y_pred_prob_all_k.mean(axis=1)
        predicted_classes = final_probabilities.argmax(axis=1)
        return predicted_classes, final_probabilities

    predicted_classes_test, _ = get_predictions('test')
    
    y_true_test = data_numpy['test']['y']
    f1 = f1_score(y_true_test, predicted_classes_test, average="weighted")
    
    return f1

if __name__ == "__main__":
    dataset_name = "UNSW-NB15" # "CIC-IDS2018", "UNSW-NB15", or "CICIoT2023"
    baseline_method = "STaSy" # "STaSy", "TabDDPM", or "TabSyn" 
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
    
    print("Preprocessing data...")

    all_targets = np.concatenate([
        real_train_target.reshape(-1, 1),
        real_test_target.reshape(-1, 1),
        synth_data_target.reshape(-1, 1),
        refined_data_target.reshape(-1, 1)
    ])

    oe = OrdinalEncoder()
    oe.fit(all_targets)

    real_train_target = oe.transform(real_train_target.reshape(-1, 1)).flatten()
    real_test_target = oe.transform(real_test_target.reshape(-1, 1)).flatten()
    synth_data_target = oe.transform(synth_data_target.reshape(-1, 1)).flatten()
    refined_data_target = oe.transform(refined_data_target.reshape(-1, 1)).flatten()
    num_classes = len(np.unique(all_targets))
    
    combined_cat = np.vstack([real_train_cat, real_test_cat])
    for col_idx in range(combined_cat.shape[1]):
        oe = OrdinalEncoder()
        oe.fit(combined_cat[:, col_idx].reshape(-1, 1))
        real_train_cat[:, col_idx] = oe.transform(real_train_cat[:, col_idx].reshape(-1, 1)).flatten()
        real_test_cat[:, col_idx] = oe.transform(real_test_cat[:, col_idx].reshape(-1, 1)).flatten()
        synth_data_cat[:, col_idx] = oe.transform(synth_data_cat[:, col_idx].reshape(-1, 1)).flatten()
        refined_data_cat[:, col_idx] = oe.transform(refined_data_cat[:, col_idx].reshape(-1, 1)).flatten()
    
    num_samples = len(synth_data_num)
    random_indices = np.random.permutation(len(real_train_num))[:num_samples]
    real_train_num = real_train_num[random_indices]
    real_train_cat = real_train_cat[random_indices]
    real_train_target = real_train_target[random_indices]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_type = "GPU" if device == "cuda" else None
    catboost_device = "0" if device == "cuda" else None
    
    nunique_per_col = np.array([np.unique(real_train_num[:, i]).shape[0] for i in range(real_train_num.shape[1])])
    valid_cols_mask = nunique_per_col > 1
    real_train_num = real_train_num[:, valid_cols_mask]
    real_test_num = real_test_num[:, valid_cols_mask]
    synth_data_num = synth_data_num[:, valid_cols_mask]
    refined_data_num = refined_data_num[:, valid_cols_mask]
    
    X_real_train = np.hstack([real_train_num, real_train_cat])
    X_real_test = np.hstack([real_test_num, real_test_cat])
    X_synth = np.hstack([synth_data_num, synth_data_cat])
    X_refined = np.hstack([refined_data_num, refined_data_cat])
    
    num_numerical_features = real_train_num.shape[1]
    
    print("Training and evaluating classifiers...")
    
    f1_catboost_real = train_and_evaluate_catboost(X_real_train, real_train_target, X_real_test, real_test_target, task_type, catboost_device)
    f1_catboost_synth = train_and_evaluate_catboost(X_synth, synth_data_target, X_real_test, real_test_target, task_type, catboost_device)
    f1_catboost_refined = train_and_evaluate_catboost(X_refined, refined_data_target, X_real_test, real_test_target, task_type, catboost_device)
    
    print(f"CatBoost F1 - Real: {f1_catboost_real:.4f}, Synth: {f1_catboost_synth:.4f}, Refined: {f1_catboost_refined:.4f}")
    reduction_rate = (1 - (f1_catboost_real - f1_catboost_refined) / (f1_catboost_real - f1_catboost_synth)) * 100
    print(f"CatBoost Refinement Reduction Rate: {reduction_rate:.2f}%")
    
    f1_tabm_real = train_and_evaluate_tabm(X_real_train, real_train_target, X_real_test, real_test_target, num_numerical_features, num_classes, device)
    f1_tabm_synth = train_and_evaluate_tabm(X_synth, synth_data_target, X_real_test, real_test_target, num_numerical_features, num_classes, device)
    f1_tabm_refined = train_and_evaluate_tabm(X_refined, refined_data_target, X_real_test, real_test_target, num_numerical_features, num_classes, device)
    print(f"TabM F1 - Real: {f1_tabm_real:.4f}, Synth: {f1_tabm_synth:.4f}, Refined: {f1_tabm_refined:.4f}")
    reduction_rate = (1 - (f1_tabm_real - f1_tabm_refined) / (f1_tabm_real - f1_tabm_synth)) * 100
    print(f"TabM Refinement Reduction Rate: {reduction_rate:.2f}%")