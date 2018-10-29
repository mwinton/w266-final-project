# standard modules
import argparse
from importlib import reload
from keras.callbacks import EarlyStopping
import mlflow
import mlflow.keras
from pprint import pprint

# our own imports
from vqa_options import ModelOptions
from vqa_utils import FakeData
from vqa_models import StackedAttentionNetwork

if __name__ == '__main__':

    # parse command line parameters and flags
    parser = argparse.ArgumentParser(description='VQA Stacked Attention Networks',
                        epilog='W266 Final Project (Fall 2018) by Rachel Ho, Ram Iyer, Mike Winton')
    parser.add_argument('-t', '--train', action='store_true',
                        help='train a new model')
    parser.add_argument('-s', '--score', action='store_true',
                        help='score against labeled test set')
    parser.add_argument('-p', '--predict', action='store_true',
                        help = 'predict using existing model')
    parser.add_argument('-f', '--fake', action='store_true',
                        help = 'run fake data through pipeline')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help = 'turn on verbose output')
    parser.add_argument('-b', '--batch_size', type=int,
                        help = 'set batch size (int)')
    parser.add_argument('-e', '--epochs', type=int,
                        help = 'set max number of epochs (int)')
    args = parser.parse_args()
    if args.verbose:
        pprint(args)
    
    # load model options from config file
    options = ModelOptions().get_options()
    
    # override default for max_epochs if specified
    if args.epochs:
        options['max_epochs'] = args.epochs
        
    # override default for batch_size if specified
    if args.batch_size:
        options['batch_size'] = args.batch_size
        
    # print all options before building graph
    if args.verbose:
        # TODO: implement mlflow logging of params
        options['verbose'] = args.verbose
        pprint(options)

    # always build graph
    san = StackedAttentionNetwork(options)
    san.build_graph(options)

    # build fake data if flag was set for a test run
    if args.fake:
        options['fake'] = args.fake
        (images_x_train, sentences_x_train, y_train,
         images_x_test, sentences_x_test, y_test) = FakeData(options).get_fakes()

    # train if flag was set
    if args.train:
        san.train(options, x=[images_x_train, sentences_x_train], y=y_train)

    # evaluate if flag was set
    if args.score:
        score=san.evaluate(options, x=[images_x_test, sentences_x_test], y=y_test)
        # TODO: implement mlflow logging of score
        
    # predict if flag was set
    if args.predict:
        san.predict(options)
        
    print('Completed!')
    