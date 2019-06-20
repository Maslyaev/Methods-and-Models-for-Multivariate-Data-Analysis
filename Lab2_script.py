#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:36:22 2019

@author: mike_ubuntu
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

pd.set_option('display.expand_frame_repr', False)

def center(A, axis_idx = 0):
    mean = A.mean(axis = axis_idx)
    std = A.std(axis = axis_idx)
    return (A - mean[0, :])/std

def sclearn_PCA(data):
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=data.shape[1])
    pca_results = pca.fit_transform(data)
    pca_comps = pca.components_
    pca_var = pca.explained_variance_
    #print(pca_values)
    return pca_var, pca_comps

def spectral_decomp(data, normed = True):
    matrix = tf.Variable(data)
    init = tf.variables_initializer([matrix], name="init")
    mm = tf.matmul(matrix, matrix, transpose_a = True)
    
    with tf.Session() as s:
        s.run(init)
        covar = (s.run(mm))
    covar_normed = covar / float(data.shape[0] - 1)

    pc_vals, pc_vecs = np.linalg.eig(covar_normed)
    print(type(pc_vecs))
    indexes = pc_vals.argsort()[::-1]
    pc_vals = pc_vals[indexes]
    pc_vecs = pc_vecs[:,indexes]
    print(type(pc_vecs))
    pc_vecs = np.linalg.inv(pc_vecs)
    if normed:
        pc_vals = pc_vals / np.sum(pc_vals)
    #S = np.diag(pc_vals)
    #A = np.dot(data, pc_vecs)
    return pc_vals, pc_vecs

def singular_value_decomp(data, normed = True):
    svd = TruncatedSVD(n_components=data.shape[1]-1)
    svd.fit(data)
    variance_svd = svd.explained_variance_   
    components_svd = svd.components_
    if normed:
        variance_svd = svd.explained_variance_ratio_   
    return variance_svd, components_svd

def domain_specific_approach(threshold, pc_var, pc_comps):
    pc_var_filtered = [x for x in pc_var if x/float(pc_var.sum()) > threshold]
    pc_comps_filtered = [pc_comps[i] for i in range(len(pc_var_filtered))]
    return pc_var_filtered, pc_comps_filtered   

def Kaiser_approach(pc_var, pc_comps):
    pc_var_filtered = [x for x in pc_var if x > np.mean(pc_var)]
    pc_comps_filtered = [pc_comps[i] for i in range(len(pc_var_filtered))]  
    return pc_var_filtered, pc_comps_filtered 

def broken_stick(pc_var, pc_comps):
    brokes = np.sort(np.concatenate((np.random.uniform(0, 1, pc_var.shape[0] - 1), np.array([0., 1.]))))
    lengths = np.flip(np.sort(np.array([brokes[i+1] - brokes[i] for i in range(brokes.shape[0]-1)])))
    pc_var_filtered = []
    for idx in range(len(lengths)):
        if pc_var[idx]/pc_var.sum() > lengths[idx]:
            pc_var_filtered.append(pc_var[idx])
        else:
            break
    pc_comps_filtered = [pc_comps[i] for i in range(len(pc_var_filtered))]  
    return pc_var_filtered, pc_comps_filtered
    #print(brokes, lengths)
    

mydateparser = lambda x: pd.datetime.strptime(x, "%Y%m%d%H%M")
data = pd.read_csv('FluxNet_Hourly.csv', header = 0, parse_dates=['TIMESTAMP_START', 'TIMESTAMP_END'], 
                   date_parser=mydateparser)
temp = data[['TA_F', 'PA_F', 'LW_IN_F' , 'VPD_F', 'SW_IN_F', 'CO2_F_MDS', 
             'WS_F', 'LE_F_MDS', 'H_F_MDS', 'RH', 'USTAR']]

temp = np.matrix(temp.values)
print('temps shape:', temp.shape)
temp = center(temp)


comp_variance_spectral, components_spectral = spectral_decomp(temp)
#comp_variance_spectral, components_spectral = Kaiser_approach(comp_variance_spectral, components_spectral)
print(comp_variance_spectral)

#plt.figure(figsize=(12,7))
#plt.plot(np.cumsum(comp_variance_spectral), linewidth=3.0)
#plt.show()

comp_variance_SVD, components_SVD = singular_value_decomp(temp)
#comp_variance_SVD, components_SVD = broken_stick(comp_variance_SVD, components_SVD)
print(comp_variance_SVD)
#plt.figure(figsize=(12,7))
#plt.plot(np.cumsum(comp_variance_SVD), linewidth=3.0)
#plt.show()

#print(components_SVD)
scores_spectral = pd.DataFrame(index=['TA_F', 'PA_F', 'LW_IN_F' , 'VPD_F', 'SW_IN_F', 'CO2_F_MDS', 
             'WS_F', 'LE_F_MDS', 'H_F_MDS', 'RH', 'USTAR'],
                          data = np.transpose(components_spectral),
                          columns=['PC{}'.format(i+1) for i in range(components_spectral.shape[1])])
print(scores_spectral.head(11))

fa = FactorAnalyzer(n_factors=12, rotation="varimax")
fa.fit(temp)
loadings_df = pd.DataFrame(index=['TA_F', 'PA_F', 'LW_IN_F' , 'VPD_F', 'SW_IN_F', 'CO2_F_MDS', 
             'WS_F', 'LE_F_MDS', 'H_F_MDS', 'RH', 'USTAR'],
                        data = fa.loadings_,
                        columns=['F{}'.format(i+1) for i in range(fa.loadings_.shape[1])])
print(loadings_df.head(11))
variances = np.array(fa.get_factor_variance()[:][0])
print(variances/sum(variances))
plt.figure(figsize=(12,7))
plt.plot(np.cumsum(variances/sum(variances)), linewidth=3.0)
plt.show()

#comp_variance, components = sclearn_PCA(temp.values)
#print(components)

#for i in range(len(comp_variance)):
#    print('Described variance: %1.6F' % (float(comp_variance[i]) / float(comp_variance.sum())))
#    print(comp_variance[i], '\n')
#print(components[0])
#print(components[:, 0])
#
#print(domain_specific_approach(0.1, comp_variance, components))
#print(broken_stick(comp_variance, components))
#PC_sk, comps_sk = sclearn_PCA(temp.values)

#print('Custom:')
#print(comps_custom)
#print('Sklearn:')
#print(comps_sk)

#print(covar_normed)