# Description: Script for keeping track of already calculated NNs
# Author: Anton D. Lautrup
# Date: 21-08-2023

# import gower

import numpy as np
import torch

from typing import Literal
from sklearn.neighbors import NearestNeighbors

#TODO: Save the NNs, so that they can be reused

def _create_matrix_with_ones(indices, num_rows):
    matrix = np.zeros((len(indices),num_rows), dtype=int)
    for i, index in enumerate(indices):
        matrix[i,index] = 1
    return matrix

### Custom Gower matrix
from sklearn.preprocessing import OrdinalEncoder
from scipy.spatial.distance import cdist

def _gower_matrix_sklearn(data_x, data_y=None, cat_features: list = None, weights=None, num_attribute_ranges=None, nums_metric: Literal['L1', 'EXP_L2'] = 'L1'):
    """Modified version of the python gower distance metric implementation
    url: https://pypi.org/project/gower/"""

    X = data_x
    if data_y is None: Y = data_x 
    else: Y = data_y 

    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if not isinstance(Y, np.ndarray): Y = np.asarray(Y)

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape 
    
    out_shape = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    ### Bit to infer, cat_features if nothing is supplied 
    if cat_features is None:
        if not isinstance(X, np.ndarray): 
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)    
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col]=True
    else:          
        cat_features = np.array(cat_features)

    ### Separate out weights
    if weights is None:
        weights = np.ones(X.shape[1])
            
    weights_cat = weights[cat_features]
    weights_num = weights[np.logical_not(cat_features)]

    ### Subsetting
    Z = np.concatenate((X,Y))
    
    x_index = range(0,x_n_rows)
    y_index = range(x_n_rows,x_n_rows+y_n_rows)
    
    Z_num = Z[:,np.logical_not(cat_features)]
    Z_cat = Z[:,cat_features]

    ### Make the denominator for the nummerical normalisation 
    if num_attribute_ranges is None:
        num_attribute_ranges = np.max(np.stack((np.array(np.ptp(Z_num,axis=0),dtype=np.float64),np.ones(len(weights_num)))),axis=0)

    X_num = Z_num[x_index,]
    Y_num = Z_num[y_index,]

    ### Do the nummerical step
    if not np.array_equal(cat_features,np.ones(X.shape[1])):
        if nums_metric == 'L1':
                nums_sum = cdist(X_num.astype(float), Y_num.astype(float), 'minkowski', p=1, w=(weights_num/num_attribute_ranges))

        elif nums_metric == 'EXP_L2':
                nums_sum = cdist(X_num.astype(float), Y_num.astype(float), 'minkowski', p=2, w=(weights_num/num_attribute_ranges**2))#/np.sqrt(len(weights_num))

        else: raise NotImplementedError("The keyword literal is not a valid!")
    else: nums_sum = out_shape
    
    ### Do the categorical step
    if not np.array_equal(cat_features,np.zeros(X.shape[1])):
        Z_cat_enc = OrdinalEncoder().fit_transform(Z_cat)

        X_cat = Z_cat_enc[x_index,]
        Y_cat = Z_cat_enc[y_index,]

        cat_sum = cdist(X_cat.astype(int),Y_cat.astype(int), 'hamming', w=weights_cat)*len(weights_cat)
    else: cat_sum = out_shape
    
    return (nums_sum+cat_sum)/weights.sum()

def _gower_matrix_torch(data_x, data_y=None, cat_features: list = None, weights=None,
                        num_attribute_ranges=None, nums_metric='L1', device='cuda'):
    """
    PyTorch implementation of Gower distance matrix
    """
    X = data_x
    Y = data_y if data_y is not None else data_x

    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if not isinstance(Y, np.ndarray): Y = np.asarray(Y)

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape

    # Infer categorical features if not provided
    if cat_features is None:
        cat_features = np.zeros(x_n_cols, dtype=bool)
        for col in range(x_n_cols):
            if not np.issubdtype(type(X[0, col]), np.number):
                cat_features[col] = True
    else:
        cat_features = np.array(cat_features)

    if weights is None:
        weights = np.ones(X.shape[1])
    weights_cat = weights[cat_features]
    weights_num = weights[~cat_features]

    # Concatenate for normalization
    Z = np.concatenate((X, Y), axis=0)
    x_index = range(0, x_n_rows)
    Z_num = Z[:, ~cat_features]
    Z_cat = Z[:, cat_features]

    # Numerical normalization
    if num_attribute_ranges is None:
        num_attribute_ranges = np.max(
            np.stack((np.ptp(Z_num, axis=0), np.ones(len(weights_num)))), axis=0)

    X_num = torch.tensor(Z_num[x_index, :], dtype=torch.float32, device=device)
    weights_num_tensor = torch.tensor(weights_num / num_attribute_ranges, dtype=torch.float32, device=device)

    if not np.all(~cat_features):
        Z_cat_enc = OrdinalEncoder().fit_transform(Z_cat)
        X_cat = torch.tensor(Z_cat_enc[x_index, :], dtype=torch.int32, device=device)
        Y_cat_all = torch.tensor(Z_cat_enc[x_n_rows:, :], dtype=torch.int32, device=device)
        weights_cat_tensor = torch.tensor(weights_cat, dtype=torch.float32, device=device)
    else:
        X_cat = Y_cat_all = weights_cat_tensor = None

    Y_num_all = torch.tensor(Z_num[x_n_rows:, :], dtype=torch.float32, device=device)

    result_list = []
    batch_size = 100

    for start in range(0, y_n_rows, batch_size):
        end = min(start + batch_size, y_n_rows)
        Y_num = Y_num_all[start:end, :]

        if not np.all(cat_features):
            if nums_metric == 'L1':
                diff = torch.abs(X_num[:, None, :] - Y_num[None, :, :])
                nums_sum = torch.sum(diff * weights_num_tensor, dim=2)
            elif nums_metric == 'EXP_L2':
                diff = (X_num[:, None, :] - Y_num[None, :, :]) ** 2
                nums_sum = torch.sum(diff * (weights_num_tensor ** 2), dim=2)
            else:
                raise NotImplementedError("The keyword literal is not a valid!")
        else:
            nums_sum = torch.zeros((x_n_rows, end - start), device=device)

        if not np.all(~cat_features):
            Y_cat = Y_cat_all[start:end, :]
            X_cat_exp = X_cat[:, None, :].expand(-1, Y_cat.shape[0], -1)
            Y_cat_exp = Y_cat[None, :, :].expand(X_cat.shape[0], -1, -1)
            neq = (X_cat_exp != Y_cat_exp).to(torch.float32)
            cat_sum = torch.sum(neq * weights_cat_tensor, dim=2)
        else:
            cat_sum = torch.zeros((x_n_rows, end - start), device=device)

        value = (nums_sum + cat_sum) / weights.sum()
        result_list.append(value)

    full_result = torch.cat(result_list, dim=1)
    return full_result.cpu().numpy() if device == 'cuda' else full_result

def _knn_distance(a, b, cat_cols, num, metric: Literal['gower', 'euclid', 'EXPERIMENTAL_gower'] = 'gower', weights=None, sampling_size=10000):
    def gower_knn(a, b, bool_cat_cols, gower_variant):
            """Function used for finding nearest neighbours"""
            d = []
            if np.array_equal(a,b):
                matrix = _gower_matrix_torch(a, cat_features=bool_cat_cols, weights=weights, nums_metric=gower_variant)+np.eye(len(a))
                for _ in range(num):
                    d.append(matrix.min(axis=1))
                    matrix += _create_matrix_with_ones(matrix.argmin(axis=1,keepdims=True),len(a))
            else:
                matrix = _gower_matrix_torch(a, b, cat_features=bool_cat_cols, weights=weights, nums_metric=gower_variant)
                for _ in range(num):
                    d.append(matrix.min(axis=1))
                    matrix += _create_matrix_with_ones(matrix.argmin(axis=1,keepdims=True),len(b))
            return d

    def eucledian_knn(a, b):
            """Function used for finding nearest neighbours"""
            d = []
            nn = NearestNeighbors(n_neighbors=num+1, metric_params={'w':weights}) #TODO: add num_att_range here as well
            if np.array_equal(a,b):
                nn.fit(a)
                dists, _ = nn.kneighbors(a)
                for i in range(num):
                    d.append(dists[:,1+i])
            else:
                nn.fit(b)
                dists, _ = nn.kneighbors(a)
                for i in range(num):
                    d.append(dists[:,i])
            return d

    if sampling_size is not None:
        b = b.sample(n=min(len(b), sampling_size), random_state=42)

    if metric=='gower' or metric=='EXPERIMENTAL_gower':
        bool_cat_cols = [col1 in cat_cols for col1 in a.columns]
        num_cols = [col2 for col2 in a.columns if col2 not in cat_cols]
        a[num_cols] = a[num_cols].astype("float")
        b[num_cols] = b[num_cols].astype("float")
        if metric=='gower': return gower_knn(a,b,bool_cat_cols, gower_variant = 'L1')
        else: return gower_knn(a,b,bool_cat_cols, gower_variant='EXP_L2')
    if metric=='euclid':
        return eucledian_knn(a,b)
    else: raise Exception("Unknown metric; options are 'gower' or 'euclid'")
