
# coding: utf-8

# In[1]:


from dypac import dypac
from niak_load_confounds import load_confounds
import niak_load_confounds
import numpy as np
import sklearn as sk
import scipy as sp
import os
import matplotlib.pyplot as plt
import time
from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiMapsMasker
from nilearn import plotting
from nilearn.decomposition import DictLearning, CanICA
from scipy.stats import pearsonr
from joblib import Memory


# In[2]:


path_base = '/home/lussier/Projects/data/embeddings'
path_func = 'func'
#path_anat = 'anat'
anat = datasets.load_mni152_template()#os.path.join(path_base,path_anat,'template_aal.nii.gz')


# In[3]:


func = []
func_file = open((os.path.join(path_base,path_func, 'functional.txt')),'r')
for line in func_file:
    func.append(os.path.join(path_base,path_func, (line.strip())))#.split(','))
func_file.close()
print (func)


# In[4]:


conf_list = []
conf_file = open((os.path.join(path_base,path_func, 'confounds.txt')),'r')
for line in conf_file:
    conf_list.append(os.path.join(path_base,path_func, (line.strip())))#.split(','))
conf_file.close()
print (conf_list)


# In[5]:


conf = []
for idx in conf_list:
    conf.append(load_confounds(idx))
conf


# In[60]:


model = dypac(n_clusters=100, n_states=100, verbose=1, n_batch=21, n_init=1, n_init_aggregation=1, n_replications=30, 
                      detrend=False, smoothing_fwhm=5, standardize=True, threshold_sim=0.2)
model.fit(func, confounds=conf)


# In[61]:


import pickle
pickle.dump(model, open("dypac_abide_sub1050clu100sta100bat21rep30thr02.pickle", "wb"))


# In[59]:


num_comp = 3
comp = model.masker_.inverse_transform(model.components_[num_comp,:].todense())
plotting.view_img(comp, bg_img=anat, threshold=0.1, vmax=1, title="Dwell time: {dt}".format(dt=model.dwell_time_[num_comp]))


# In[ ]:


#tseries = model.masker_.transform(func[0], conf[0])

tseries = []
for f, c in zip(func, conf):
    tseries.append(model.masker_.transform(f, c))
tseries[0]


# In[ ]:


#weights = model.transform_sparse(func[0], conf[0])

weights = []
for f, c in zip(func, conf):
    weights.append(model.transform_sparse(f, c))
weights[0]


# In[ ]:


plt.plot(weights[0][:, num_comp])


# In[ ]:


num_t = 40
img = model.masker_.inverse_transform(tseries[1][num_t, :])
plotting.view_img(img, bg_img=anat, cut_coords=[2, -22, 10], vmax=4)


# In[ ]:


img_r = model.masker_.inverse_transform(weights[1][num_t,:] * model.components_)
plotting.view_img(img_r, bg_img=anat, cut_coords=[2, -22, 10], vmax=4)


# In[ ]:


img_diff = image.new_img_like(img, img.get_fdata() - img_r.get_fdata())
plotting.view_img(img_diff, bg_img=anat, cut_coords=[2, -22, 10], vmax=4)


# In[ ]:


for tt in range(0,tseries[1].shape[0]):
    img = model.masker_.inverse_transform(tseries[1][tt, :])
    if (tt==0):
        diff = np.square(img.get_fdata()) # The data is demeaned, so this is a square diff to the mean
    else:
        diff = diff + np.square(img.get_fdata())
img_orig_std = image.new_img_like(img, diff / tseries[0].shape[0])
plotting.view_img(img_orig_std, bg_img=anat, cut_coords=[2, -22, 10])


# In[ ]:


for tt in range(0,tseries[1].shape[0]):
    img = model.masker_.inverse_transform(tseries[1][tt, :])
    img_r = model.masker_.inverse_transform(weights[1][tt,:] * model.components_)
    if (tt==0):
        diff = np.square(img.get_fdata() - img_r.get_fdata())
    else:
        diff = diff + np.square(img.get_fdata() - img_r.get_fdata())
img_std = image.new_img_like(img, diff / tseries[1].shape[0])
img_std.to_filename(os.path.join(path_base,'test2r2map02.nii.gz'))
plotting.view_img(img_std, bg_img=anat, cut_coords=[2, -22, 10], vmax=1)

