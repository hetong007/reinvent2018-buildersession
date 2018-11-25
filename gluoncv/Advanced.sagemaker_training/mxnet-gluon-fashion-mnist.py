from __future__ import print_function

import argparse
import json
import logging
import os
import time
import numpy as np

import mxnet as mx
from mxnet import nd, autograd, gluon, image, init
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon import nn

from collections import namedtuple

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

class LeNet(nn.HybridBlock):
    def __init__(self, num_classes=10, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(
                # TODO fill in the LeNet architecture from previous notebook.
            )
            self.output = nn.Dense(num_classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

def train(current_host, hosts, num_cpus, num_gpus, model_dir, hyperparameters, **kwargs):
    batch_size = hyperparameters.batch_size
    epochs = hyperparameters.epochs
    learning_rate = hyperparameters.learning_rate
    momentum = hyperparameters.momentum
    log_interval = hyperparameters.log_interval
    learning_rate_update_interval = hyperparameters.learning_rate_update_interval
    learning_rate_update_factor = hyperparameters.learning_rate_update_factor
    wd = hyperparameters.wd
    num_classes = hyperparameters.num_classes

    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    net = LeNet(num_classes)
    net.initialize()

    with net.name_scope():
        net.output = nn.Dense(num_classes)

    net.output.initialize(init.Xavier(magnitude=2.24), ctx = ctx)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    batch_size *= max(1, len(ctx))

    preprocess_threads=num_cpus*4
    transformer = gdata.vision.transforms.Compose([
        gdata.vision.transforms.ToTensor()
    ])

    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)

    train_data = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=preprocess_threads)
    val_data = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=preprocess_threads)

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            optimizer_params={'learning_rate': learning_rate,
                                              'momentum': momentum,
                                              'wd': wd})

    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    time_training_start = time.time()

    best_accuracy = 0.0

    for epoch in range(epochs):
        time_epoc_start = time.time()
        time_batch_start = time.time()
        metric.reset()

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with autograd.record():
                for x, y in zip(data, label):
                    output = net(x)
                    L = loss(output, y)
                    Ls.append(L)
                    outputs.append(output)
                for L in Ls:
                    L.backward()

            trainer.step(batch[0].shape[0])
            metric.update(label, outputs)

            if i % log_interval == 0 and i > 0:
                name, acc = metric.get()
                print('Epoch [%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f' %
                                 (epoch, i, batch_size / (time.time() - time_batch_start), name, acc))

            time_batch_start = time.time()

        if epoch % learning_rate_update_interval == 0 and epoch > 0:
            trainer.set_learning_rate(trainer.learning_rate * learning_rate_update_factor)
            print('Epoch [%d] learning_rate: %f' %
                             (epoch, trainer.learning_rate))


        train_accuracy = evaluate_accuracy(ctx, net, train_data)
        validation_accuracy = evaluate_accuracy(ctx, net, val_data)

        if current_host == hosts[0]:
            if validation_accuracy > best_accuracy:
                print('Epoch [%d] save model' % (epoch))
                net.export('%s/model' % model_dir, 0)
                best_accuracy = validation_accuracy

        print('Epoch [%d] train accuracy: %f' % (epoch, train_accuracy))
        print('Epoch [%d] validation accuracy: %f' % (epoch, validation_accuracy))
        print('Epoch [%d] best validation accuracy: %f' % (epoch, best_accuracy))
        print('Epoch [%d] training time %d sec'% (epoch, (time.time() - time_epoc_start)))

    print('Total training time %d sec'% (time.time() - time_training_start))


def evaluate_accuracy(ctx, net, data):
    acc = mx.metric.Accuracy()
    for i, batch in enumerate(data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        acc.update(label, outputs)
    return acc.get()[1]


def model_fn(model_dir):
    net = gluon.nn.SymbolBlock.imports(model_dir + '/model-symbol.json', ['data'], model_dir + '/model-0000.params')

    return net


def transform_fn(net, data, input_content_type, output_content_type):

    x = mx.image.imdecode(data)
    image = mx.img.resize_short(x, image_size)
    cropped_image, (x, y, width, height) = mx.img.center_crop(image, (image_size, image_size))
    imagex = cropped_image.astype('float32')/255
    imagex = mx.image.color_normalize(imagex,
                                          mean=mx.nd.array([0.485, 0.456, 0.406]),
                                          std=mx.nd.array([0.229, 0.224, 0.225]))
    imagex = nd.transpose(imagex, (2, 0, 1))
    imagex = nd.expand_dims(imagex, 0)

    output = net(imagex)

    softmax_result = nd.softmax(output[0])

    response_payload = json.dumps(softmax_result.asnumpy().tolist(), ensure_ascii=False)

    return response_payload, 'application/json'



if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--learning_rate_update_interval', type=int, default=5)
    parser.add_argument('--learning_rate_update_factor', type=float, default=0.9)
    parser.add_argument('--num_classes', type=int, default=257)

    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=int, default=json.loads(os.environ['SM_HOSTS']))

    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--num_cpus', type=int, default=os.environ['SM_NUM_CPUS'])

    args, _ = parser.parse_known_args()

    print('epochs: %d' % (args.epochs))
    print('batch_size: %d' % (args.batch_size))
    print('momentum: %f' % (args.momentum))
    print('wd: %f' % (args.wd))
    print('log_interval: %d' % (args.log_interval))
    print('learning_rate_update_interval: %d' % (args.learning_rate_update_interval))
    print('learning_rate_update_factor: %f' % (args.learning_rate_update_factor))
    print('num_classes: %d' % (args.num_classes))
    print('model_dir: %s' % (args.model_dir))
    print('current_host: %s' % (args.current_host))
    print('hosts: %s' % (args.hosts))
    print('num_cpus: %d' % (args.num_cpus))
    print('num_gpus: %d' % (args.num_gpus))

    train(args.current_host, args.hosts, args.num_cpus, args.num_gpus,
          args.model_dir, args)
