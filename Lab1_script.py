#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:53:16 2019

@author: mike_ubuntu
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, shapiro, probplot, kstest, ranksums, normaltest
from scipy.optimize import minimize
import math
from sklearn.neighbors import KernelDensity
from scipy.special import erfinv

def nd_inverse_cdf(y, location, scale):
    return scale * math.sqrt(2.) * erfinv(2. * y - 1) + location

def nd_density(x, *parameters):
    loc = parameters[0]; scale = parameters[1]
    return 1/(scale * math.sqrt(2.0 * math.pi)) * math.exp(-pow(x - loc, 2) / (2.0 * pow(scale, 2)))

def error(params, x_values, y_values):
    res = sum([pow(y - nd_density(x, *params), 2) for x, y in list(zip(x_values, y_values))])
    return res

def least_squares_method(data, init = (90, 1)):
    bins = 0.2
    bands = {}
    N = data.shape[0]
    for val in data:
        band = (math.floor(val / bins) + 0.5) * bins
        if band in bands:
            bands[band] += 1 / (bins * N)
        else:
            bands[band] = 1 / (bins * N)
    x_values = sorted(bands)
    y_values = [bands[x] for x in x_values]
    LSM_loc, LSM_scale = minimize(error, x0 = init, args = (x_values, y_values)).x
    print('LSM: location %3.4f , scale %2.5f' % (LSM_loc, LSM_scale))
    return LSM_loc, LSM_scale
    #y_est = [funct(x, loc, scale) for x in x_values]
    #plt.plot(x_values, y_values)
    #plt.plot(x_values, y_est)

def likelihood_function_2(temp):
    loc = temp[0]; scale = temp[1]
    #print(loc, scale) #- math.pi/2.0*math.log(2*math.pi) 
    N = pressure_1.shape[0]
    res = - N*math.log(scale) - 1/(2.0 * pow(scale, 2)) * np.sum(pow(pressure_1 - loc, 2))
    return 1 / res

def max_likelihood_analytical(data):
    loc = np.sum(data)/float(data.shape[0])
    variance = abs(np.sqrt(np.sum(pow(data - loc, 2))/float(data.shape[0])))
    return loc, variance

def moments_method(data):
    loc = np.sum(data)/float(data.shape[0])
    variance = abs(np.sqrt(np.sum(pow(data - loc, 2))/float(data.shape[0])))
    return loc, variance

def kernel_density_estimator(x, x_grid, bandwidth=0.2, kernel_type = 'gaussian', **kwargs):
    kde_skl = KernelDensity(bandwidth=bandwidth, kernel = kernel_type, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)
    
#def plot_kde(data, data_range = (-1, 1)):
#    x = np.linspace(data_range[0], data_rLSM_loc, LSM_scaleange[1], 1000)
#    pdf = kernel_density_estimator(data, x, bandwidth = 0.2)
#    #plt.plot(x, pdf)

def kholmogorov_test(data, loc, scale):
    print(kstest(data, 'norm', (loc, scale))) 
    return kstest(data, 'norm', (loc, scale))

def wilkoxon_test(data, loc, scale):
    distr = np.random.normal(loc, scale, data.shape[0])
    print(ranksums(data, distr))
    return ranksums(data, distr)

def box_muller_transform(loc, scale, number = 10000, plot = False):
    y_sim_1 = []; y_sim_2 = []
    for i in range(number):
        k_x1 = np.random.uniform(0, 1)
        k_x2 = np.random.uniform(0, 1)
        y_sim_1.append(loc + scale * math.cos(2*math.pi*k_x1) * math.sqrt(-2 * math.log(k_x2)))
        y_sim_2.append(loc + scale * math.sin(2*math.pi*k_x1) * math.sqrt(-2 * math.log(k_x2)))
    y_sim_1 = np.array(y_sim_1); y_sim_2 = np.array(y_sim_2)
    if plot:
        fig = plt.figure(figsize=(16,13))
        bins = int(5 * math.log(y_sim_1.shape[0])) 
        plt.hist(y_sim_1, bins = bins, normed = True)  
        plt.show()
    return y_sim_1, y_sim_2
        
        
def inverse_function_method(location, scale, number = 10000, plot = False):
    x_sim = np.random.uniform(0, 1, number)
    y_sim = np.array([nd_inverse_cdf(x, location, scale) for x in x_sim])
    if plot:
        fig = plt.figure(figsize=(16,13))
        bins = int(5 * math.log(y_sim.shape[0])) 
        plt.hist(y_sim, bins = bins, normed = True)  
        plt.show()
    return y_sim     

            
def geometric_method(location, scale, interval, number = 10000, plot = False):
    if location > interval[1] and location < interval[0]:
        raise Exception('location is out of the studied interval')
    y_sim = []
    for i in range(number):
        k_x1 = np.random.uniform(0, 1)
        k_x2 = np.random.uniform(0, 1)
        k_x1_star = interval[0] + (interval[1] - interval[0]) * k_x1
        k_x2_star = nd_density(location, location, scale) * k_x2
        while k_x2_star > nd_density(k_x1_star, location, scale):
            k_x1 = np.random.uniform(0, 1)
            k_x2 = np.random.uniform(0, 1)
            k_x1_star = interval[0] + (interval[1] - interval[0]) * k_x1
            k_x2_star = nd_density(location, location, scale) * k_x2
        y_sim.append(k_x1_star)
    y_sim = np.array(y_sim)
    if plot:
        fig = plt.figure(figsize=(16,13))
        bins = int(5 * math.log(y_sim.shape[0])) 
        plt.hist(y_sim, bins = bins, normed = True)
        plt.show()
    return y_sim

    
    
data = np.loadtxt('MI_Fedorovskoje.txt')
press = data[:, 0]
pressure = press[np.logical_not(press == -9999)]
#pressure = np.random.normal(1, 2, 10000)

pressure_1 = pressure#[400:5400]


shap_stat, shap_pval = shapiro(pressure_1)
print("Shapiro-Wilk test: statistics value: %3.5f , p-value: %10.3E" % (shap_stat, shap_pval))
dag_stat, dag_pval = normaltest(pressure_1)
print("D’Agostino-Pearson’s test: statistics value: %3.5f , p-value: %10.3E" % (dag_stat, dag_pval))

x = np.linspace(np.min(pressure_1) - 2, np.max(pressure_1) + 2, 1000)
pdfs = []
bdwth = 0.3
kernel_types = ['epanechnikov', 'gaussian', 'tophat']
for kernel_type in kernel_types:
    pdf = kernel_density_estimator(pressure_1, x, bandwidth=bdwth, kernel_type = kernel_type)
    pdfs.append(pdf)

bins = int(5 * math.log(pressure_1.shape[0])) 
fig = plt.figure(figsize=(16,13))   
plt.hist(pressure, bins = bins, normed = True)    
for pdf_idx in range(len(pdfs)):
    print("writing")
    plt.plot(x, pdfs[pdf_idx], label = kernel_types[pdf_idx], linewidth = 3)
plt.legend(fontsize = 'xx-large')
plt.show()
    
MLE_analyt_exp, MLE_analyt_scale = max_likelihood_analytical(pressure_1)
print("Maximum likelihood estimation (analytical): loc: %3.4f, variance: %2.4f" % (MLE_analyt_exp, pow(MLE_analyt_scale, 2)))
MLE_numer_exp, MLE_numer_scale = minimize(likelihood_function_2, max_likelihood_analytical(pressure_1), method='Nelder-Mead').x
print("Maximum likelihood estimation (numerical): loc: %3.4f, variance: %2.4f" % (MLE_numer_exp, pow(MLE_numer_scale, 2)))

moments_exp, moments_scale = moments_method(pressure_1)
print(moments_exp, moments_scale) 
print("Method of moments: loc: %3.4f, variance: %2.4f" % (moments_exp, pow(moments_scale, 2)))

LSM_loc, LSM_scale = least_squares_method(pressure, init = (90, 1))
#press_theor = np.random.normal(loc = MLE_numer_exp, scale = MLE_numer_scale, size = 1000)

fig = plt.figure(figsize=(16,13))
probplot(pressure_1, sparams=(MLE_analyt_exp, MLE_analyt_scale), plot=plt)
plt.show()
fig = plt.figure(figsize=(16,13))
probplot(pressure_1, sparams=(LSM_loc, LSM_scale), plot=plt)
plt.show()

kholmogorov_test(pressure_1, LSM_loc, LSM_scale)
wilkoxon_test(pressure_1, LSM_loc, LSM_scale)

y_sim_inverse = inverse_function_method(LSM_loc, LSM_scale, number = 100000)
y_sim_geometric = geometric_method(LSM_loc, LSM_scale, (np.min(pressure)-2, np.max(pressure)+2), number = 100000)
y_sim_box_muller = box_muller_transform(LSM_loc, LSM_scale, number = 100000)

bins = int(5 * math.log(pressure_1.shape[0])) 
#fig = plt.figure(figsize=(16,13))   
#plt.hist(pressure, bins = bins, normed = True)    
#plt.hist(y_sim_inverse, bins = bins, normed = True, alpha = 0.5, color = 'r')    
#plt.show()

#fig = plt.figure(figsize=(16,13))   
#plt.hist(pressure, bins = bins, normed = True)   
#plt.hist(y_sim_geometric, bins = bins, normed = True, alpha = 0.5, color = 'g') 
#plt.show()   

fig = plt.figure(figsize=(16,13))   
plt.hist(pressure, bins = bins, normed = True)   
plt.hist(y_sim_box_muller[0], bins = bins, normed = True, alpha = 0.5, color = 'y') 
plt.show()   #plt.show()
