# encoding:utf-8

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = "softmax", cuda=True):
        super(BaselineFinetune, self).__init__( model_func,  n_way, n_support, cuda=cuda)
        self.cuda = cuda
        self.loss_type = loss_type

    def set_forward(self,x,is_feature = True):
        return self.set_forward_adaptation(x,is_feature); #Baseline always do adaptation
 
    def set_forward_adaptation(self,x,is_feature = True):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,is_feature)   #已经放到cuda上了

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )    # [NK, -1]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )        # [NM, -1]

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support )) # [0, 0, 0, 0, 0, 1, 1...]
        y_support = Variable(y_support.cuda() if self.cuda else y_support)

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':        
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda() if self.cuda else linear_clf

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda() if self.cuda else loss_function
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):          # finetune 100 个epoch
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                # 选择当前batch
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ])
                selected_id = selected_id.cuda() if self.cuda else selected_id
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)   #[NM, N]
        return scores


    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
        

