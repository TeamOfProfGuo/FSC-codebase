import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file


##=======================================================================================================================
import argparse
def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--dataset', default='CUB', help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model', default='Conv4',
                        help='model: Conv{4|6} / ResNet{10|18|34|50|101}')  # 50 and 101 are not used in the paper
    parser.add_argument('--method', default='baseline',
                        help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}')  # relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way', default=5, type=int,
                        help='class num to classify for training')  # baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way', default=5, type=int,
                        help='class num to classify for testing (validation) ')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ')  # still required for save_features.py and test.py to find the model path correctly

    if script == 'train':
        parser.add_argument('--num_classes', default=200, type=int,
                            help='total number of classes in softmax, only used in baseline')  # make it larger than the maximum label value in base class
        parser.add_argument('--save_freq', default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=-1, type=int,
                            help='Stopping epoch')  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume', action='store_true',
                            help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup', action='store_true',
                            help='continue from baseline, neglected if resume is true')  # never used in the paper
    elif script == 'save_features':
        parser.add_argument('--split', default='novel',
                            help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,
                            help='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split', default='novel',
                            help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,
                            help='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')
    else:
        raise ValueError('Unknown script')

    return parser.parse_args([])

##=======================================================================================================================


np.random.seed(10)
params = parse_args('train')
params.cuda = False

if params.dataset == 'cross':
    base_file = configs.data_dir['miniImagenet'] + 'all.json'
    val_file = configs.data_dir['CUB'] + 'val.json'
elif params.dataset == 'cross_char':
    base_file = configs.data_dir['omniglot'] + 'noLatin.json'
    val_file = configs.data_dir['emnist'] + 'val.json'
else:
    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'

if 'Conv' in params.model:
    if params.dataset in ['omniglot', 'cross_char']:
        image_size = 28
    else:
        image_size = 84
else:
    image_size = 224

if params.dataset in ['omniglot', 'cross_char']:
    assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
    params.model = 'Conv4S'

optimization = 'Adam'

if params.stop_epoch == -1:
    if params.method in ['baseline', 'baseline++']:
        if params.dataset in ['omniglot', 'cross_char']:
            params.stop_epoch = 5
        elif params.dataset in ['CUB']:
            params.stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
        elif params.dataset in ['miniImagenet', 'cross']:
            params.stop_epoch = 400
        else:
            params.stop_epoch = 400  # default
    else:  # meta-learning methods
        if params.n_shot == 1:
            params.stop_epoch = 600
        elif params.n_shot == 5:
            params.stop_epoch = 400
        else:
            params.stop_epoch = 600  # default

if params.method in ['baseline', 'baseline++']:
    base_datamgr = SimpleDataManager(image_size, batch_size=16)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    val_datamgr = SimpleDataManager(image_size, batch_size=64)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    if params.dataset == 'omniglot':
        assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
    if params.dataset == 'cross_char':
        assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

    if params.method == 'baseline':
        model = BaselineTrain(model_dict[params.model], params.num_classes, cuda = params.cuda)
    elif params.method == 'baseline++':
        model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')

elif params.method in ['protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
    n_query = max(1, int(
        16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

    if params.method == 'protonet':
        model = ProtoNet(model_dict[params.model], **train_few_shot_params)
    elif params.method == 'matchingnet':
        model = MatchingNet(model_dict[params.model], **train_few_shot_params)
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S':
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model](flatten=False)
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

        model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
    elif params.method in ['maml', 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **train_few_shot_params)
        if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            model.n_task = 32
            model.task_update_num = 1
            model.train_lr = 0.1
else:
    raise ValueError('Unknown method')

model = model.cuda()

params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
if params.train_aug:
    params.checkpoint_dir += '_aug'
if not params.method in ['baseline', 'baseline++']:
    params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

start_epoch = params.start_epoch
stop_epoch = params.stop_epoch
if params.method == 'maml' or params.method == 'maml_approx':
    stop_epoch = params.stop_epoch * model.n_task  # maml use multiple tasks in one update

if params.resume:
    resume_file = get_resume_file(params.checkpoint_dir)
    if resume_file is not None:
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch'] + 1
        model.load_state_dict(tmp['state'])
elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used in our paper
    baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
    configs.save_dir, params.dataset, params.model, 'baseline')
    if params.train_aug:
        baseline_checkpoint_dir += '_aug'
    warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
    tmp = torch.load(warmup_resume_file)
    if tmp is not None:
        state = tmp['state']
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.",
                                     "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
        model.feature.load_state_dict(state)
    else:
        raise ValueError('No warm_up file')




######===========================================   Start Training =====================================================
# model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
# def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
if optimization == 'Adam':
    optimizer = torch.optim.Adam(model.parameters())
else:
   raise ValueError('Unknown optimization, please define by yourself')

max_acc = 0

for epoch in range(start_epoch,stop_epoch):
    model.train()
    model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return
    model.eval()

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    acc = model.test_loop( val_loader)
    if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
        print("best model! save...")
        max_acc = acc
        outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
        torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
        outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
        torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

