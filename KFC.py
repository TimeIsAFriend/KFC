#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import spatial

np.seterr(all="ignore")

def get_neighbors(data,k):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data)    
    distances, indices = nbrs.kneighbors(data)
    return distances, indices.astype(int)

def cal_outlier_score(data,k):
    clf = LocalOutlierFactor(n_neighbors=k)
    clf.fit_predict(data)
    return -clf.negative_outlier_factor_

def get_medoid(coords):
    coords = np.array(coords)
    cost = distance.cdist(coords, coords, 'cityblock')  # Manhattan distance,  'euclidean','minkowski'
    return np.argmin(cost.sum(axis=0))

def cal_c_for_KFC(outlierScore, data, neighbor, k):
    idx_medoid_object, idx_median_score = np.zeros(data.shape[0]), np.zeros(data.shape[0])
    for i in range(len(data)):
        idx_neighbor = neighbor[i]
        # find index of medoid data 
        data_of_neighbors = data[idx_neighbor]
        idx_medoid = get_medoid(data_of_neighbors)
        idx_medoid_object[i]=idx_medoid
        
        # find index of median score
        score_of_neighbors = outlierScore[idx_neighbor]
        idx_median = np.argsort(score_of_neighbors)[len(score_of_neighbors) // 2]
        idx_median_score[i]=idx_median
        
    u,v= np.array([np.mean(outlierScore[neighbor[i][1:]]) for i in range(len(outlierScore))]),np.array([outlierScore[i] for i in range(len(outlierScore))])
    c_kfcs=1-spatial.distance.cosine(u,v)
    u,v=np.array(idx_medoid_object),np.array(idx_median_score)
    c_kfcr=1-spatial.distance.cosine(u,v)  
    
    return c_kfcs, c_kfcr


def FKC(k_candidates,data):
    distances, neighbors = get_neighbors(data,100)
    arr_c_kfcs,arr_c_kfcr=[],[]
    for k in k_candidates:
        neighbor=neighbors[:,:k+1]
        outlier_score=cal_outlier_score(data,k)
        c_kfcs, c_kfcr=cal_c_for_KFC(outlier_score, data, neighbor, k)
        arr_c_kfcs.append(c_kfcs)
        arr_c_kfcr.append(c_kfcr)
    idx_kfcs,idx_kfcr=np.argmax(arr_c_kfcs),np.argmax(arr_c_kfcr)
    return k_candidates[idx_kfcs],k_candidates[idx_kfcr]



#KFC example with LOF on Parkinson

file_name='Parkinson_withoutdupl_75'
data, label = load_data(file_name)


# find optimal k
k_min,k_max=3,99
k_candidates=range(k_min,k_max)
optimal_k_by_kfcs,optimal_k_by_kfcr=FKC(k_candidates,data)
print(optimal_k_by_kfcs,optimal_k_by_kfcr)



#plot roc, c_kfcs,c_kfcr
distances, neighbors = get_neighbors(data,100)
k_min,k_max=3,99
arr_c_kfcs,arr_c_kfcr,arr_roc=[],[],[]
for k in range(k_min,k_max):
    neighbor=neighbors[:,:k+1]
    outlier_score=cal_outlier_score(data,k)
    roc=roc_auc_score(label,outlier_score)
    arr_roc.append(roc)
    
    c_kfcs, c_kfcr=cal_c_for_KFC(outlier_score, data, neighbor, k)
    arr_c_kfcs.append(c_kfcs)
    arr_c_kfcr.append(c_kfcr)

    

x=range(k_min,k_max)
plt.plot(x,np.array(arr_roc), '--.',c='chartreuse',label='AUC') 
plt.plot(x,np.array(arr_c_kfcs), '-^',c='turquoise',label='KFCS') 
plt.plot(x,np.array(arr_c_kfcr), '-x',c='cornflowerblue',label='KFCR') 
plt.legend()
plt.title(file_name) 
plt.show()


# In[ ]:




