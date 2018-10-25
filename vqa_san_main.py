# standard modules
import argparse
from importlib import reload
import mlflow
import mlflow.keras
from pprint import pprint

# our own imports
from vqa_options import ModelOptions
from StackedAttentionNetwork import StackedAttentionNetwork

if __name__ == '__main__':

    # parse command line parameters and flags
    parser = argparse.ArgumentParser(description='VQA Stacked Attention Networks',
                        epilog='W266 Final Project (Fall 2018) by Rachel Ho, Ram Iyer, Mike Winton')
    parser.add_argument('-t', '--train', action='store_true',
                        help='train a new model')
    parser.add_argument('-p', '--predict', action='store_true',
                        help = 'predict using existing model')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help = 'turn on verbose output')
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
        
    # print all options before building graph
    if args.verbose:
        options['verbose'] = args.verbose
        pprint(options)

    # always build graph
    san = StackedAttentionNetwork(options)
    san.build_graph(options)
    
    # train if requested
    if args.train:
        san.train(options)

    # predict if requested
    if args.predict:
        san.predict(options)
        
    print('Completed!')
    