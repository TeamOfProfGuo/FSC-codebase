# encoding:utf-8

import torch
import numpy as np
import h5py

class SimpleHDF5Dataset:
    def __init__(self, file_handle = None):
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0 
        else:
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]   # h5py convert to np array [N(sample_size), D]
            self.all_labels = self.f['all_labels'][...]      # h5py convert to np array [N(sample_size]   (NB*B)
            self.total = self.f['count'][0]                  # total sample_size
           # print('here')
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

def init_loader(filename):
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)

    #labels = [ l for l  in fileset.all_labels if l != 0]
    feats = fileset.all_feats_dset    # [N, D]
    labels = fileset.all_labels
    while np.sum(feats[-1]) == 0:     # 去掉为0的feature和label  最后不够的补0吗
        feats  = np.delete(feats,-1,axis = 0)
        labels = np.delete(labels,-1,axis = 0)
        
    class_list = np.unique(np.array(labels)).tolist()
    inds = range(len(labels))

    cl_data_file = {}     # dict: cls_id = [list of feats]
    for cl in class_list:
        cl_data_file[cl] = []
    for ind in inds:
        cl_data_file[labels[ind]].append( feats[ind])

    return cl_data_file
    # 返回dict {cls_id: [list of feats]}