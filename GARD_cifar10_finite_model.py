from jax.lib import xla_bridge
from jax import vmap, grad, jit
from jax import random
import jax.numpy as jnp  # jax numpy
import jax
from glob import glob
import numpy as np  # vanilla numpy
import os.path as osp
import argparse
from tqdm import tqdm
import time
from functools import partial
import tensorflow_datasets as tfds
import os
import sys
import inspect
from jax.experimental import optimizers
import torch
import torchvision
import torchvision.transforms as transforms
from art.experimental.estimators.classification.jax import JaxClassifier
from art.attacks.evasion import FastGradientMethod
from jax.experimental import stax
import pickle
from typing import Optional


par = argparse.ArgumentParser()

# admin
# architecture config
par.add_argument('--model', default='mlp', choices=['cnn', 'mlp'],
                 help='model type: cnn or mlp')
par.add_argument('--weights_file', default='./best_ckpt_ens_200.pkl', choices=['cnn', 'mlp'],
                 help='model type: cnn or mlp')

#par.add_argument('--get', default='ntk', choices=['ntk', 'nngp'], help='get ntk or nngp kernel')

par.add_argument('--act', default='relu',
                 choices=['relu', 'erf'], help='nonlinear activation function')

par.add_argument('--num_classes', type=int, default=10,
                 help='number of classes')

par.add_argument('--epochs', type=int, default=100,
                 help='epocs')
par.add_argument('--f', type=int, default=512,
                 help='model width: in filters for cnn or hidden units for mlp')

par.add_argument('--layers', type=int, default=2,
                 help='number of layers in mlp')

par.add_argument('--ws', type=float, default=1,
                 help='weight standard deviation at initialization')

par.add_argument('--diag', type=float, default=0,
                 help='diagonal regularization term')

par.add_argument('--lr', type=float, default=1,
                 help='lr')
par.add_argument('--batch_size', type=int, default=128,
                 help='batch_size')
par.add_argument('--ens', type=int, default=200,
                 help='ens number')

par.add_argument('--ts', type=int, default=14, help='upper time value')

par.add_argument('--num_ts', type=int, default=100,
                 help='number of time steps to evaluate')

par.add_argument('--seed', type=int, default=10, help='random seed')
par.add_argument('--data_dir', type=str, default="./data",
                 help='dir of the data')
args = par.parse_args("")
key = random.PRNGKey(args.seed)


def mlp_model(num_units, num_layers, num_output, nonlin='relu'):
    """Dense network of an arbitrary width and number of layers"""

    if nonlin == 'relu':
        act = stax.Relu
    else:
        act = stax.Erf

    blocks = []
    for _ in range(num_layers):
        blocks += [stax.serial(stax.Dense(num_units), act)]
    for _ in range(num_layers):
        blocks += [stax.serial(stax.Dense(num_units//2), act)]

    for _ in range(num_layers):
        blocks += [stax.serial(stax.Dense(num_units//4), act)]

    blocks += [stax.Dense(num_output)]

    return stax.serial(*blocks)


def create_network(key):

    init_fn, apply_fn = mlp_model(args.f, args.layers, args.num_classes, nonlin=args.act)
    _, params = init_fn(key, (-1, 3072))
    return params



def get_art_model(
    model_kwargs: dict=None, wrapper_kwargs: dict=None, weights_path: Optional[str] = None
) -> JaxClassifier:
    init_fn, apply_fn = mlp_model(
        args.f, args.layers, args.num_classes, nonlin=args.act)

    key = random.PRNGKey(args.seed)
    ensemble_key = random.split(key, args.ens)
    # loading
    # ens_params = vmap(create_network)(ensemble_key)
    ens_params = pickle.load(open(weights_path, "rb"))
    print(weights_path)
    print("Model Loaded Successifully...")
    def loss_func(model, x, y):
        print("___________")
        print(x.shape)
        # x = np.transpose(x, (0, 2, 3, 1))
        x = x.reshape(-1, 3072)
        y = np.array(y[:, None] == np.arange(args.num_classes), np.float32)

        init_fn, apply_fn = mlp_model(
            args.f, args.layers, args.num_classes, nonlin=args.act)

        preds = np.mean(vmap(apply_fn, in_axes=(0, None))
                       (model, x), axis=0)
        print("shape")
        print(preds.shape)
        loss = 0.5 * (np.mean(preds - y) ** 2)

        return loss

    # model is the ens params
    def predict_func(model, x):

        x = x.reshape(-1, 3072)

        init_fn, apply_fn = mlp_model(
            args.f, args.layers, args.num_classes, nonlin=args.act)

        pred = np.mean(vmap(apply_fn, in_axes=(0, None))
                       (model, x), axis=0)

        # pred = np.argmax(pred, axis=1)

        return pred

    @jit
    def update_func(model, x, y):
        x = np.transpose(x, (0, 2, 3, 1))
        x = x.reshape(-1, 3072)
        y = np.array(y[:, None] == np.arange(
        args.num_classes), np.float32)

        opt_init, opt_update, get_params = optimizers.sgd(args.lr)

        opt_state = vmap(opt_init)(model)

        loss = jit(lambda params, x, y: 0.5 *
                   np.mean((apply_fn(params, x) - y) ** 2))

        grad_loss = jit(lambda state, x, y: grad(loss)
                        (get_params(state), x, y))

        grad_ens = vmap(grad_loss, in_axes=(0, None, None))(opt_state, x, y)

        opt_state = vmap(opt_update, (None, 0, None))(0, grad_ens, opt_state)

        model = vmap(get_params)(opt_state)

        return model

    classifier = JaxClassifier(
        model=ens_params,
        predict_func=predict_func,
        loss_func=loss_func,
        update_func=update_func,
        input_shape=(32, 32, 3),
        nb_classes=10,
    )

    return classifier


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
#                                             shuffle=False, num_workers=2)

# classifier=get_art_model(weights_path=args.weights_file)
# for i, data in enumerate(tqdm(testloader)):
#     inputs, labels = data
#     inputs = inputs.numpy()
#     labels = labels.numpy()

#     inputs = np.array(inputs)
#     labels = np.array(labels)
#     print(inputs)
#     grads=classifier.loss_gradient(inputs,labels)
#     print(grads)
#     print(grads.shape)
#     out=classifier.predict(inputs)
#     print(np.mean(out==labels))
#     adv_crafter = FastGradientMethod(classifier, eps=10/255.0)
#     x_test_adv = adv_crafter.generate(x=inputs)
#     out=classifier.predict(x_test_adv)
#     print(np.mean(out==labels))
#     print("=========>")
    