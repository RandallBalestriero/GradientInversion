import tensorflow as tf
import sys
import argparse

sys.path.insert(0, "../../Sknet")
import sknet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['mnist', 'fashionmnist',
                    'svhn', 'cifar10', 'cifar100'], default='mnist')
parser.add_argument('--model', type=str, choices=['cnn', 'smallresnet',
                    'largeresnet'], default='cnn')
parser.add_argument("--data_augmentation", type=int, default=0)
parser.add_argument("--parameter", help="parameter", type=float, default=0)

args = parser.parse_args()
DATASET = args.dataset
MODEL = args.model
DATA_AUGMENTATION = args.data_augmentation
PARAMETER = args.parameter

# Dataset
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
standardize.transform(dataset['images/train_set'], inplace=True)
standardize.transform(dataset['images/test_set'], inplace=True)
standardize.transform(dataset['images/valid_set'], inplace=True)

dataset.create_placeholders(32, {'train_set': 'random_see_all',
                                 'test_set': 'continuous',
                                 'valid_set': 'continuous'}, device="/cpu:0")

# Network
# --------------
dnn = sknet.Network(name='simple_model')
dnn.append(sknet.ops.RandomAxisReverse(dataset.images, axis=[-1]))
image_shape = dnn[-1].shape.as_list()[-2:]
crop_shape = (image_shape[0]-4, image_shape[1]-4)
dnn.append(sknet.ops.RandomCrop(dnn[-1], crop_shape, seed=10))

images = dnn[-1]

if MODEL == 'cnn':
    sknet.networks.ConvLarge(dnn, dataset.n_classes)
elif MODEL == 'smallresnet':
    sknet.networks.Resnet(dnn, dataset.n_classes, D=4, W=1)
elif MODEL == 'largeresnet':
    sknet.networks.Resnet(dnn, dataset.n_classes, D=8, W=2)

prediction = dnn[-1]

reconstruction = tf.gradients(dnn[-3], images)[0]
loss_recons = sknet.losses.squared_error(reconstruction, images)
loss_classif = sknet.losses.crossentropy_logits(p=dataset.labels,
                                                q=prediction)
test_accu = sknet.losses.streaming_mean(sknet.losses.accuracy(dataset.labels,
                                                              prediction))
test_recons = sknet.losses.streaming_mean(loss_recons)
test_classif = sknet.losses.streaming_mean(loss_classif)

B = dataset.N('train_set')//32
lr = sknet.schedules.PiecewiseConstant(0.001, {90*B: 0.0005, 160*B: 0.0001})
optimizer = sknet.optimizers.Adam(loss_classif+PARAMETER*loss_recons,
                                  dnn.variables(trainable=True), lr)
minimizer = tf.group(*optimizer.updates, *dnn.updates)
reset_op = tf.group(optimizer.reset_variables_op, dnn.reset_variables_op)

# Workers
# ---------
period = sknet.dataset.BatchPeriod(1000)
train = sknet.Worker(minimizer, loss_recons=loss_recons,
                     images=(dataset.images, period),
                     reconstructions=(reconstruction, period),
                     loss_classif=loss_classif, context='train_set',
                     deterministic=False)

valid = sknet.Worker(loss_recons=loss_recons, loss_classif=loss_classif,
                     context='valid_set', deterministic=True)

test = sknet.Worker(loss_recons=loss_recons, loss_classif=loss_classif,
                    images=(dataset.images, period),
                    reconstructions=(reconstruction, period),
                    context='test_set', deterministic=True)

path = '/mnt/drive1/rbalSpace/GradientInversion/supervised/'
workplace = sknet.utils.Workplace(dataset)

for i in range(10):
    filename = '{}_{}_{}_run{}'.format(DATASET, MODEL, PARAMETER, i)
    queue = sknet.Queue((train, valid, test), filename=path+filename)
    workplace.execute_queue(queue, deter_func=dnn.deter_dict, repeat=200)
    workplace.session.run(reset_op)
