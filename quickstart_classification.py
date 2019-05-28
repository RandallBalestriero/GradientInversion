import sys
sys.path.insert(0, "../")

import sknet
import matplotlib
matplotlib.use('Agg')
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pylab as pl
import time
import tensorflow as tf
from sknet.dataset import BatchIterator
from sknet import ops,layers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['mnist', 'fashionmnist',
                    'svhn', 'cifar10', 'cifar100'], default='mnist')
parser.add_argument('--model', type=str, choices=['cnn','smallresnet',
                    'largeresnet'], default='cnn')
parser.add_argument("--data_augmentation", type=int, default=0)
parser.add_argument("--parameter", help="parameter", type=float, default=0)

args = parser.parse_args()
DATASET = args.dataset
MODEL = args.model
DATA_AUGMENTATION = args.data_augmentation
PARAMETER = args.parameter

# Data Loading
# -------------
if DATASET == 'mnist':
    dataset = sknet.dataset.load_mnist()
elif DATASET == 'fashionmnist':
    dataset = sknet.dataset.load_fashonmnist()
elif DATASET == 'cifar10':
    dataset = sknet.dataset.load_cifar10()
elif DATASET == 'cifar100':
    dataset = sknet.dataset.load_cifar100()
elif DATASET == 'svhn':
    dataset = sknet.dataset.load_svhn()

if "valid_set" not in dataset.sets:
    dataset.split_set("train_set", "valid_set", 0.15)

standardize = sknet.dataset.Standardize().fit(dataset['images/train_set'])
dataset['images/train_set'] = \
                        standardize.transform(dataset['images/train_set'])
dataset['images/test_set'] = \
                        standardize.transform(dataset['images/test_set'])
dataset['images/valid_set'] = \
                        standardize.transform(dataset['images/valid_set'])


# create the unsupervised set
dataset.split_set('train_set','utrain_sets'
perm = np.random.permutation(dataset['images/train_set'].shape[0])
dataset['uimages/train_set'] = dataset['images/train_set'][perm[:-100]]
dataset['ulabels/train_set'] = dataset['labels/train_set'][perm[:-100]]
dataset['images/train_set'] = dataset['images/train_set'][perm[-100:]]
dataset['labels/train_set'] = dataset['labels/train_set'][perm[-100:]]


iterator = BatchIterator(32, {'train_set': 'random_see_all',
                         'test_set': 'continuous', 'valid_set': 'continuous'})

dataset.create_placeholders(iterator, device="/cpu:0")

# Create Network
# --------------

dnn = sknet.Network(name='simple_model')
images = tf.concat([dnn.images, dnn.uimages], 0)
if DATA_AUGMENTATION:
    dnn.append(ops.RandomAxisReverse(images, axis=[-1]))
    dnn.append(ops.RandomCrop(dnn[-1], (28, 28), seed=10))
else:
    dnn.append(images)

if MODEL == 'cnn':
    sknet.networks.ConvLarge(dnn, dataset.n_classes)
elif MODEL == 'smallresnet':
    sknet.networks.Resnet(dnn, dataset.n_classes, D=2, W=1)
elif MODEL == 'largeresnet':
    sknet.networks.Resnet(dnn, dataset.n_classes, D=4, W=2)

prediction = dnn[-1]

reconstruction = tf.gradients(prediction,images)[0]
loss_recons = sknet.losses.MSE(reconstruction,images)
loss_classif = sknet.losses.crossentropy_logits(p=dataset.labels,
                                                q=prediction[:32])
accu = sknet.losses.StreamingAccuracy(dataset.labels, prediction[:32])
uaccu = sknet.losses.StreamingAccuracy(dataset.ulabels, prediction[32:])


B         = dataset.N_BATCH('train_set')
lr        = sknet.schedules.PiecewiseConstant(0.05,
                                    {100*B:0.005,200*B:0.001,250*B:0.0005})
optimizer = sknet.optimizers.NesterovMomentum(loss,lr,
                                    params=dnn.variables(trainable=True))
minimizer = tf.group(optimizer.updates+dnn.updates)

# Workers
#---------

min1  = sknet.Worker(name='minimizer',context='train_set',op=[minimizer,loss],
        deterministic=False, period=[1,100])

accu1 = sknet.Worker(name='accu',context='test_set', op=accu,
        deterministic=True, transform_function=np.mean,verbose=1)

queue = sknet.Queue((min1,accu1))

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_queue(queue,repeat=350)


