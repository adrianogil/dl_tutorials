#!/usr/bin/env python

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping, ForceFloatX

from dl_tutorials.part_1_theano.logistic_regression import (
    LogisticRegressor
)
from dl_tutorials.utils.build_2d_datasets import build_2d_datasets

from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop


######
import os

import matplotlib.image as pmimage
import numpy

from fuel.datasets import IndexableDataset

from fuel.schemes import ShuffledScheme

import logging
from argparse import ArgumentParser
import copy
import theano
from theano import tensor
import blocks

def GetPedestrianData(path):

    pedestrianpath=path+'ped_examples'
    nonpedestrianpath=path+'non-ped_examples'

    pedestrian_fn_list = os.listdir(pedestrianpath)
    nonpedestrian_fn_list = os.listdir(nonpedestrianpath)

    ### Creating Forest Dataset ###

    n_pedestrian = 4800
    n_nonpedestrian = 5000
    n_total = n_pedestrian + n_nonpedestrian

    pedestrian_imgs = []
    nonpedestrian_imgs = []

    for fn in pedestrian_fn_list:
        img_path = os.path.join(pedestrianpath, fn)
        img = pmimage.imread(img_path)
        img = numpy.array(img)
        pedestrian_imgs.append(img)

    for fn in nonpedestrian_fn_list:
        img_path = os.path.join(nonpedestrianpath, fn)
        img = pmimage.imread(img_path)
        img = numpy.array(img)
        nonpedestrian_imgs.append(img)

    pedestrian_imgs = numpy.array(pedestrian_imgs).reshape(n_pedestrian, -1)

    nonpedestrian_imgs = numpy.array(nonpedestrian_imgs).reshape(n_nonpedestrian, -1)

    data_images = numpy.concatenate((pedestrian_imgs, nonpedestrian_imgs), axis=0)

    data_labels = [1] * n_pedestrian + [0] * n_nonpedestrian
    data_labels = numpy.array(data_labels)
    data_labels = data_labels.reshape((n_total,1))

    return data_images,data_labels

def GetPedestrianDataset():

    train_data_images, train_data_labels = GetPedestrianData('/home/adrgil/datasets/pedestrian/1/')
    test_data_images, test_data_labels = GetPedestrianData('/home/adrgil/datasets/pedestrian/T1/')

    train_dataset = IndexableDataset({
            'features': train_data_images,
            'targets': train_data_labels
    })
    test_dataset = IndexableDataset({
        'features': test_data_images,
        'targets': test_data_labels
    })

    return train_dataset,test_dataset