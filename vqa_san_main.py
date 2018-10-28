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
    parser.add_argument('-p', '--predict', action='store_true',
                        help = 'predict using existing model')
    parser.add_argument('-f', '--fake', action='store_true',
                        help = 'run fake data through pipeline')
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
    print(san.summary())

    # build fake data if flag was set for a test run
    if args.fake:
        options['fake'] = args.fake
        train_images_x, train_sentences_x, train_y = FakeData(options)

    # train if requested
    if args.train:
        #set early stopping monitor to stop training when it won't improve anymore
        early_stopping_monitor = EarlyStopping(patience=3)
        san.fit(x=[train_images_x, train_sentences_x],
                y=train_y,
                batch_size=options.get('batch_size', 1),
                epochs=options.get('max_epochs', 5),
                verbose=2 if verbose else 0,
                # validation_split=0.2,
                callbacks=[early_stopping_monitor]
               )

    # predict if requested
    if args.predict:
        san.predict(options)
        
    print('Completed!')
    