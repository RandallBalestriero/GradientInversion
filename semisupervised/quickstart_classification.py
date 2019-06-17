import sys
sys.path.insert(0, "../../Sknet")
import numpy as np
import sknet
import tensorflow as tf
from sknet.dataset import BatchIterator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['mnist', 'fashionmnist',
                    'svhn', 'cifar10', 'cifar100'], default='mnist')
parser.add_argument('--model', type=str, choices=['cnn','smallresnet',
                    'largeresnet'], default='cnn')
parser.add_argument("--data_augmentation", type=int, default=0)
parser.add_argument("--parameter", type=float, default=0)
parser.add_argument("--samples", type=int, default=100)

args = parser.parse_args()
DATASET = args.dataset
MODEL = args.model
DATA_AUGMENTATION = args.data_augmentation
PARAMETER = args.parameter
SAMPLES = args.samples

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
perm = np.random.permutation(dataset['images/train_set'].shape[0])
dataset['uimages/train_set'] = dataset['images/train_set'][perm[:-SAMPLES]]
dataset['ulabels/train_set'] = dataset['labels/train_set'][perm[:-SAMPLES]]
dataset['images/train_set'] = dataset['images/train_set'][perm[-SAMPLES:]]
dataset['labels/train_set'] = dataset['labels/train_set'][perm[-SAMPLES:]]

dataset['uimages/test_set'] = dataset['uimages/train_set'][[0]]
dataset['ulabels/test_set'] = dataset['ulabels/train_set'][[0]]
dataset['uimages/valid_set'] = dataset['uimages/train_set'][[0]]
dataset['ulabels/valid_set'] = dataset['ulabels/train_set'][[0]]


iterator = BatchIterator(16, {'train_set': 'random_see_all',
                         'test_set': 'continuous', 'valid_set': 'continuous'})

dataset.create_placeholders(iterator, device="/cpu:0")

# Create Network
# --------------

dnn = sknet.Network(name='simple_model')
inputs = tf.concat([dataset.images, dataset.uimages], 0)
if DATA_AUGMENTATION:
    dnn.append(sknet.ops.RandomAxisReverse(inputs, axis=[-1]))
    image_shape = dnn[-1].shape.as_list()[-2:]
    crop_shape = (image_shape[-2], image_shape[-1])
    dnn.append(sknet.ops.RandomCrop(dnn[-1], crop_shape, seed=10))
else:
    dnn.append(inputs)

images = dnn[-1]

if MODEL == 'cnn':
    sknet.networks.ConvLarge(dnn, dataset.n_classes)
elif MODEL == 'smallresnet':
    sknet.networks.Resnet(dnn, dataset.n_classes, D=2, W=1)
elif MODEL == 'largeresnet':
    sknet.networks.Resnet(dnn, dataset.n_classes, D=4, W=2)

prediction = dnn[-1]

reconstruction = tf.gradients(dnn[-3], images)[0]
train_recons = sknet.losses.MSE(reconstruction, images)
train_classif = sknet.losses.crossentropy_logits(p=dataset.labels,
                                                 q=prediction[:16])
train_uaccu = sknet.losses.StreamingAccuracy(dataset.ulabels, prediction[16:])



test_recons = sknet.losses.StreamingMean(sknet.losses.MSE(
                                          reconstruction[:16], images[:16]))
test_accu = sknet.losses.StreamingAccuracy(dataset.labels, prediction[:16])

B = dataset.N('train_set')//16
lr = sknet.schedules.PiecewiseConstant(0.01, {75*B:0.005, 125*B:0.001})
optimizer = sknet.optimizers.Adam(train_classif+train_recons*PARAMETER,
                                  dnn.variables(trainable=True), lr)
minimizer = tf.group(optimizer.updates+dnn.updates)
reset_op = tf.group(dnn.reset_variables_op, optimizer.reset_variables_op)

# Workers
# ---------

train_w = sknet.Worker(name='minimizer', context='train_set', op=[minimizer,
                       train_recons, train_classif, train_uaccu],
                       deterministic=False, period=[1, 100, 100, 1],
                       verbose=[0, 2, 2, 1])

valid_w = sknet.Worker(name='accu', context='valid_set', op=[test_accu,
                       test_recons], deterministic=True, verbose=1)

test_w = sknet.Worker(name='accu', context='test_set', op=[test_accu,
                      test_recons], deterministic=True, verbose=1)

workplace = sknet.utils.Workplace(dnn, dataset=dataset)
path = '/mnt/drive1/rbalSpace/GradientInversion/semisupervised/semisupervised'
for i in range(10):
    filename = '_classif_{}_{}_{}_{}_{}_run{}'.format(DATA_AUGMENTATION,
                                                      DATASET, MODEL,
                                                      PARAMETER, SAMPLES, i)

    queue = sknet.Queue((train_w, valid_w, test_w), filename=path+filename)
    workplace.execute_queue(queue, repeat=150)
    workplace.session.run(reset_op)

