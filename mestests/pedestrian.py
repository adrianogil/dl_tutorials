#!/usr/bin/env python

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping, ForceFloatX

from dl_tutorials.blocks.extensions.plot import (
    PlotManager, Plotter, DisplayImage
)
from dl_tutorials.blocks.extensions.display import (
    ImageDataStreamDisplay, WeightDisplay
)

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
import netpbdmfile

from fuel.datasets import IndexableDataset

from fuel.schemes import ShuffledScheme

import logging
from argparse import ArgumentParser
import copy
import theano
from theano import tensor
import blocks

def GetPedestrianData(paths):

    ### Creating Forest Dataset ###
    pedestrian_imgs = []
    nonpedestrian_imgs = []

    for path in paths:
        pedestrianpath=path+'ped_examples/'
        nonpedestrianpath=path+'non-ped_examples/'

        pedestrian_fn_list = os.listdir(pedestrianpath)
        nonpedestrian_fn_list = os.listdir(nonpedestrianpath)

        for fn in pedestrian_fn_list:
            img_path = os.path.join(pedestrianpath, fn)
            img = netpbdmfile.imread(img_path)
            img = numpy.array(img)
            pedestrian_imgs.append(img)

        for fn in nonpedestrian_fn_list:
            img_path = os.path.join(nonpedestrianpath, fn)
            print img_path
            img = netpbdmfile.imread(img_path)
            img = numpy.array(img)
            nonpedestrian_imgs.append(img)

    n_pedestrian = 2*4800
    n_nonpedestrian = 2*5000
    n_total = n_pedestrian + n_nonpedestrian

    pedestrian_imgs = numpy.array(pedestrian_imgs).reshape((n_pedestrian, -1))

    nonpedestrian_imgs = numpy.array(nonpedestrian_imgs).reshape((n_nonpedestrian, -1))

    data_images = numpy.concatenate((pedestrian_imgs, nonpedestrian_imgs), axis=0)

    data_labels = [1] * n_pedestrian + [0] * n_nonpedestrian
    data_labels = numpy.array(data_labels)
    data_labels = data_labels.reshape((n_total,1))

    return data_images,data_labels

def GetPedestrianDataset():

    train_data_images, train_data_labels = GetPedestrianData(['/Users/adrianogil/workspace/datasets/pedestrian/1/','/Users/adrianogil/workspace/datasets/pedestrian/2/'])
    test_data_images, test_data_labels = GetPedestrianData(['/Users/adrianogil/workspace/datasets/pedestrian/T1/','/Users/adrianogil/workspace/datasets/pedestrian/T2/'])

    train_dataset = IndexableDataset({
            'features': train_data_images,
            'targets': train_data_labels
    })
    test_dataset = IndexableDataset({
        'features': test_data_images,
        'targets': test_data_labels
    })

    return train_dataset,test_dataset

# Getting around having tuples as argument and output
class TupleMapping(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, args):
        return (self.fn(args[0]), )


def main(dataset_name='sklearn', num_epochs=100):
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')

    def random_flip(data):
            feat, targ = data
            flip_x = 2 * numpy.random.randint(2) - 1
            flip_y = 2 * numpy.random.randint(2) - 1

            new_feat = feat.reshape((-1,18,36,1))[:,::flip_x, ::flip_y, :].reshape((-1, 648))

            return new_feat, targ

    logistic_regressor = LogisticRegressor(input_dim=648)
    probs = logistic_regressor.get_probs(features=x)
    params = logistic_regressor.get_params()
    cost = logistic_regressor.get_cost(probs=probs, targets=y).mean()
    cost.name = 'cost'
    misclassification = logistic_regressor.get_misclassification(
        probs=probs, targets=y
    ).mean()
    misclassification.name = 'misclassification'

    # train_dataset, test_dataset = build_2d_datasets(dataset_name=dataset_name)
    train_dataset, test_dataset = GetPedestrianDataset()

    algorithm = blocks.algorithms.GradientDescent(
        cost=cost,
        params=params,
        step_rule=blocks.algorithms.Scale(learning_rate=0.1))

    train_data_stream = ForceFloatX(
        data_stream=DataStream(
            dataset=train_dataset,
            iteration_scheme=SequentialScheme(
                examples=train_dataset.num_examples,
                batch_size=20,
            )
        )
    )

    # train_data_stream = Mapping(
    #         data_stream=train_data_stream,
    #         mapping=random_flip
    #     )

    test_data_stream = ForceFloatX(
        data_stream=DataStream(
            dataset=test_dataset,
            iteration_scheme=SequentialScheme(
                examples=test_dataset.num_examples,
                batch_size=20,
            )
        )
    )

    model = Model(cost)

    extensions = []
    extensions.append(Timing())
    extensions.append(FinishAfter(after_n_epochs=num_epochs))
    extensions.append(DataStreamMonitoring(
        [cost, misclassification],
        test_data_stream,
        prefix='test'))
    extensions.append(TrainingDataMonitoring(
        [cost, misclassification],
        prefix='train',
        after_epoch=True))

    scoring_function = theano.function([x], probs)

    plotters = []
    plotters.append(Plotter(
        channels=[['test_cost', 'test_misclassification',
                   'train_cost', 'train_misclassification']],
        titles=['Costs']))
    score_train_stream = Mapping(data_stream=copy.deepcopy(train_data_stream),
                                 mapping=TupleMapping(scoring_function),
                                 add_sources=('scores',))
    score_test_stream = Mapping(data_stream=copy.deepcopy(test_data_stream),
                                mapping=TupleMapping(scoring_function),
                                add_sources=('scores',))
    
    display_train = ImageDataStreamDisplay(
        data_stream=copy.deepcopy(train_data_stream),
        image_shape=(18, 36, 1),
        axes=(0, 1, 'c'),
        shift=-127.5,
        rescale=1./127.5
    )

    weight_display = WeightDisplay(
        weights=params[0],
        transpose=(1, 0),
        image_shape=(18, 36, 1),
        axes=(0, 1, 'c'),
        shift=-0.5,
        rescale=2.
    )

    images_displayer = DisplayImage(
        image_getters=[display_train, weight_display],
        titles=['Training examples', 'Weights']
    )
    plotters.append(images_displayer)

    extensions.append(PlotManager('Pedestrian Dataset', 
                                  plotters=plotters,
                                  after_epoch=False,
                                  every_n_epochs=50,
                                  after_training=True))
    extensions.append(Printing())
    main_loop = MainLoop(model=model,
                         data_stream=train_data_stream,
                         algorithm=algorithm,
                         extensions=extensions)

    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a logistic regression on"
                            " a 2D dataset.")
    parser.add_argument("--num-epochs", type=int, default=1000,
                        help="Number of training epochs to do.")
    parser.add_argument("--dataset", default="sklearn", nargs="?",
                        help=("Dataset to use."))
    args = parser.parse_args()
    main(dataset_name=args.dataset, num_epochs=args.num_epochs)

