import argparse
import pickle
import json
import sys

import h5py
import numpy as np
import os
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint

sys.path.append('..')

from vqa.dataset.types import DatasetType
from vqa.dataset.dataset import VQADataset, MergeDataset

from vqa.model.library import ModelLibrary
from vqa.model.vqa_options import ModelOptions 

# ------------------------------ GLOBALS ------------------------------
# Constants
ACTIONS = ['train', 'val', 'test', 'eval']


# Defaults
DEFAULT_MODEL = 1 
#DEFAULT_MODEL = 1  max(ModelLibrary.get_valid_model_nums())
DEFAULT_ACTION = 'train'


# ------------------------------- SCRIPT FUNCTIONALITY -------------------------------

def main(options):


    print('Action: ' + options['action_type'])
    print('Model number: {}'.format(options['model_num']))
    print('Extended: {}'.format(options['extended']))

    if (options['max_train_size'] != None):
        print('Training Set Size: {}'.format(options['max_train_size']))
    if (options['max_val_size'] != None):
       print('Validation set size: {}'.format(options['max_val_size']))

    # set paths for weights and results.
    options = ModelOptions.set_local_paths(options)

    # set numpy random seed for deterministic results
    np.random.seed(2018)
    
    # Always load train dataset to obtain the question_max_len from it
    train_dataset = load_dataset(DatasetType.TRAIN,options)
    options['max_sentence_len'] = train_dataset.question_max_len

    # Load model
    vqa_model = ModelLibrary.get_model(options)

    # Load dataset depending on the action to perform
    if action == 'train':
        dataset = train_dataset
        val_dataset = load_dataset(DatasetType.VALIDATION,options)
        if extended:
            extended_dataset = MergeDataset(train_dataset, val_dataset)
            train(vqa_model, extended_dataset, options)
        else:
            train(vqa_model, dataset, options,val_dataset=val_dataset)

    elif action == 'val':
        dataset = load_dataset(DatasetType.VALIDATION,options)
        validate(vqa_model, dataset, options)

    elif action == 'test':
        dataset = load_dataset(DatasetType.TEST,options)
        test(vqa_model, dataset, options)

    elif action == 'eval':
        dataset = load_dataset(DatasetType.EVAL,options)
        test(vqa_model, dataset, options)

    else:
        raise ValueError('The action type is unrecognized')


def load_dataset(dataset_type, options):
    

    dataset_path = ModelOptions.get_dataset_path(options,dataset_type)

    try:
        with open(dataset_path, 'rb') as f:
            print('Loading dataset...')
            dataset = pickle.load(f)
            print('Dataset loaded')
            samples = dataset.samples

            if(max_size == None):
                dataset.max_sample_size = len(samples)
            else:
                dataset.max_sample_size = max_size

            # check to make sure the samples list is sorted by image indices
            if( all(samples[i].image.features_idx <= samples[i+1].image.features_idx for i in range(len(samples)-1))) :
                 print("Passed sorted sample array check")
            else:
                 assert(0)

    except IOError:
        print('Creating dataset...')
        dataset = VQADataset(dataset_type, options)
        print('Preparing dataset...')
        dataset.prepare()
        print('Dataset size: %d' % dataset.size())
        print('Dataset ready.')

        print('Saving dataset...')
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('Dataset saved')

    return dataset


def train(model, dataset, options, val_dataset=None):
    if (not extended) and (not val_dataset):
        raise ValueError('If not using the extended dataset, a validation dataset must be provided')


    model_num = options["model_num"]
    model_weights_path = options["weights_path"]
    losses_path = options["results_path"]

    max_train_size = options["max_train_size"]
    max_val_size   = options['max_val_size']

    loss_callback = LossHistoryCallback(losses_path)
    save_weights_callback = CustomModelCheckpoint(model_weights_path, options['weights_dir_path'], model_num)
    stop_callback = EarlyStopping(patience=options['early_stop_patience'])
    batch_size = options['batch_size']
    max_epochs = options['max_epochs']


    if(max_train_size != None):
        samples_per_train_epoch = min(max_train_size,dataset.size()) 
        samples_per_train_epoch = max(batch_size,samples_per_train_epoch)
    else:
        samples_per_train_epoch = dataset.size()

    if(val_dataset != None):
        if(max_val_size !=None ):
            samples_per_val_epoch = min(max_val_size,val_dataset.size()) 
            samples_per_val_epoch = max(batch_size,samples_per_val_epoch)
        else:
            samples_per_val_epoch = val_dataset.size()

    print('Start training...')
    if not extended:
        model.fit_generator(dataset.batch_generator(), steps_per_epoch=samples_per_train_epoch//batch_size, epochs=max_epochs,
                            callbacks=[save_weights_callback, loss_callback, stop_callback],
                            validation_data=val_dataset.batch_generator(), 
                            validation_steps=samples_per_val_epoch//batch_size,max_queue_size=20)
    else:
        model.fit_generator(dataset.batch_generator(batch_size, split='train'), steps_per_epoch=dataset.train_size()/batch_size,
                            epochs=num_epochs, callbacks=[save_weights_callback, loss_callback, stop_callback],
                            validation_data=dataset.batch_generator(batch_size, split='val'),
                            validation_steps=dataset.val_size()/batch_size,max_queue_size=20)
    print('Trained')


def validate(model, dataset, options):

    weights_path = options['weights_path']
    batch_size   = options['batch_size']
    print('Loading weights...')
    model.load_weights(weights_path)
    print('Weights loaded')
    print('Start validation...')
    result = model.evaluate_generator(dataset.batch_generator(batch_size), val_samples=dataset.size())
    print('Validated. Loss: {}'.format(result))

    return result


def test(model, dataset, options):


    weights_path = options['weights_path']
    results_path = options['results_path']

    print('Loading weights...')
    model.load_weights(weights_path)
    print('Weights loaded')
    print('Predicting...')
    images, questions = dataset.get_dataset_input()
    results = model.predict([images, questions], options['batch_size'])
    print('Answers predicted')

    print('Transforming results...')
    results = np.argmax(results, axis=1)  # Max index evaluated on rows (1 row = 1 sample)
    results = list(results)
    print('Results transformed')

    print('Building reverse word dictionary...')
    word_dict = {idx: word for word, idx in dataset.tokenizer.word_index.iteritems()}
    print('Reverse dictionary build')

    print('Saving results...')
    results_dict = [{'answer': word_dict[results[idx]], 'question_id': sample.question.id}
                    for idx, sample in enumerate(dataset.samples)]
    with open(results_path, 'w') as f:
        json.dump(results_dict, f)
    print('Results saved')


# ------------------------------- CALLBACKS -------------------------------

class LossHistoryCallback(Callback):
    def __init__(self, results_path):
        super(LossHistoryCallback, self).__init__()
        self.train_losses = []
        self.val_losses = []
        self.results_path = results_path

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        try:
            with h5py.File(self.results_path, 'a') as f:
                if 'train_losses' in f:
                    del f['train_losses']
                if 'val_losses' in f:
                    del f['val_losses']
                f.create_dataset('train_losses', data=np.array(self.train_losses))
                f.create_dataset('val_losses', data=np.array(self.val_losses))
        except (TypeError, RuntimeError):
            print('Couldn\'t save losses')


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, weights_dir_path, model_num, monitor='val_loss', verbose=0, save_best_only=False,
                 mode='auto'):
        super(CustomModelCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                                                    save_best_only=save_best_only, mode=mode)
        self.model_num = model_num
        self.weights_dir_path = weights_dir_path
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.last_epoch = epoch

    def on_train_end(self, logs={}):
        try:
            os.symlink(self.weights_dir_path + 'model_weights_{}.{}.hdf5'.format(self.model_num, self.last_epoch),
                       self.weights_dir_path + 'model_weights_{}'.format(self.model_num))
        except OSError:
            # If the symlink already exist, delete and create again
            os.remove(self.weights_dir_path + 'model_weights_{}'.format(self.model_num))
            # Recreate
            os.symlink(self.weights_dir_path + 'model_weights_{}.{}.hdf5'.format(self.model_num, self.last_epoch),
                       'model_weights_{}'.format(self.model_num))
            pass


# ------------------------------- ENTRY POINT -------------------------------

if __name__ == '__main__':

    # parse command line parameters and flags
    parser = argparse.ArgumentParser(description='VQA Stacked Attention Networks',
                        epilog='W266 Final Project (Fall 2018) by Rachel Ho, Ram Iyer, Mike Winton')

    parser.add_argument('-f', '--fake', action='store_true',
                        help = 'run fake data through pipeline')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help = 'turn on verbose output')
    parser.add_argument('-b', '--batch_size', type=int,
                        help = 'set batch size (int)')
    parser.add_argument('-e', '--epochs', type=int,
                        help = 'set max number of epochs (int)')

    parser.add_argument(
        '-m',
        '--model',
        type=int,
        choices=ModelLibrary.get_valid_model_nums(),
        default=DEFAULT_MODEL,
        help='Specify the model architecture to interact with. Each model architecture has a model number associated.'
             'By default, the model will be the last architecture created, i.e., the model with the biggest number'
    )
    parser.add_argument(
        '-a',
        '--action',
        choices=ACTIONS,
        default=DEFAULT_ACTION,
        help='Which action should be perform on the model. By default, training will be done'
    )
    parser.add_argument(
        '--extended',
        action='store_true',
        help='Add this flag if you want to use the extended dataset, this is, use part of the validation dataset to'
             'train your model. Only valid for the --action=train'
    )
    parser.add_argument("--max_train_size",type=int,help="maximum number of training samples to use")
    parser.add_argument("--max_val_size",type=int,help="maximum number of validation samples to use")

    # Start script
    args = parser.parse_args()
    if args.verbose:
        pprint(args)

   # load model options from config file
    model_options = ModelOptions().get_options()
    
    # override default for max_epochs if specified
    if args.epochs:
        model_options['max_epochs'] = args.epochs
        
    # override default for batch_size if specified
    if args.batch_size:
        model_options['batch_size'] = args.batch_size
        

    # parse args with defaults
    model_options['model_num'] = args.model 
    model_options['action'] = args.action
        
    # print all options before building graph
    if args.verbose:
        # TODO: implement mlflow logging of params
        model_options['verbose'] = args.verbose
        pprint(options)

    """
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

    """     

    main(model_options)

    print('Completed!')

