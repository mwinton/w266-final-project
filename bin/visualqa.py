# Main file to launch runs.
# Adapted from https://github.com/imatge-upc/vqa-2016-cvprw, Issey Masuda Mora	

import argparse
import datetime
import h5py
import json
import mlflow
import numpy as np
import os
import pickle
import pprint
import sys

# import matplotlib this way to run without a display
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, TensorBoard

sys.path.append('..')

from vqa.dataset.types import DatasetType
from vqa.dataset.dataset import VQADataset, MergeDataset

from vqa.experiments.experiment_select import ExperimentLibrary
from vqa.model.model_select import ModelLibrary
from vqa.model.options import ModelOptions 

# ------------------------------ GLOBALS ------------------------------
# Constants
ACTIONS = ['train', 'val', 'test', 'eval']


# Defaults
DEFAULT_MODEL = "baseline"
DEFAULT_EXPERIMENT = 0
DEFAULT_ACTION = 'train'


# ------------------------------- SCRIPT FUNCTIONALITY -------------------------------

def main(options):


    print('Action: ' + options['action_type'])
    print('Model name: {}'.format(options['model_name']))
    print('Extended: {}'.format(options['extended']))

    if (options['max_train_size'] != None):
        print('Training Set Size: {}'.format(options['max_train_size']))
    if (options['max_val_size'] != None):
       print('Validation set size: {}'.format(options['max_val_size']))

    # set paths for weights and results.
    options = ModelOptions.set_local_paths(options)

    # set numpy random seed for deterministic results
    seed = 2018
    np.random.seed(seed)
    
    # open mlflow context for logging
    if (options['logging']):
        mlflow.start_run()
        mlflow.log_param('random_seed', seed)
#             # TODO: monitor for next release of MLFlow (beyond 0.7.0).
#             # https://pypi.org/project/mlflow/#history
#             # mlflow.set_experiment(options['experiment_id'])
        print('Enabled logging to MLFlow server for experiment_id = {}...'.format(options['experiment_id']))

    # Always load train dataset to obtain the one hot encoding indices 
    # and  max_sentence_len from it
    train_dataset = load_dataset(DatasetType.TRAIN,options)
    options['max_sentence_len'] = train_dataset.max_sentence_len

    answer_one_hot_mapping = train_dataset.answer_one_hot_mapping

    # Load model
    vqa_model = ModelLibrary.get_model(options)
    
    # Save time-stamped model json file
    d = datetime.datetime.now().isoformat()
    json_path = options['saved_models_path'] + 'model_{}_{}.json'.format(options['experiment_id'], d)
    with open(json_path, 'w') as json_file:
        json_file.write(vqa_model.to_json())

    # Load dataset depending on the action to perform
    action = options['action_type']
    if action == 'train':
        if options['logging']:
            # log Keras model configuration
            mlflow.log_artifact(json_path)
            # log non-empty model parameters (else mlflow crashes)
            for key, val in options.items():
                if val != '' and val != None:
                    mlflow.log_param(key, val)
            print('Logged experiment params to MLFlow...')
                    
        dataset = train_dataset
        val_dataset = load_dataset(DatasetType.VALIDATION,options,answer_one_hot_mapping)
        if options['extended']:
            extended_dataset = MergeDataset(train_dataset, val_dataset)
            train(vqa_model, extended_dataset, options)
        else:
            train(vqa_model, dataset, options, val_dataset=val_dataset)
        
    elif action == 'val':
        dataset = load_dataset(DatasetType.VALIDATION,options,answer_one_hot_mapping)
        validate(vqa_model, dataset, options)

    elif action == 'test':
        dataset = load_dataset(DatasetType.TEST,options,answer_one_hot_mapping)
        test(vqa_model, dataset, options)

    elif action == 'eval':
        dataset = load_dataset(DatasetType.EVAL,options,answer_one_hot_mapping)
        test(vqa_model, dataset, options)

    else:
        raise ValueError('The action type is unrecognized')

    if options['logging']:
        mlflow.end_run()
        print('Closed MLFlow logging context...')


def load_dataset(dataset_type, options,answer_one_hot_mapping = None):
    
    """
        Load the dataset from disk if available. If not, build it from the questions/answers json and image embeddings
        If this is the training dataset, retrieve the answer one hot mapping from disk or re-create it.
    """ 

    dataset_path = ModelOptions.get_dataset_path(options,dataset_type)
    # if this isn't a training dataset, the answer one hot indices are expected to be available
    if (dataset_type != DatasetType.TRAIN):
        assert(answer_one_hot_mapping != None) 

    try:
        with open(dataset_path, 'rb') as f:
            print('Loading dataset...')
            dataset = pickle.load(f)
            print('Dataset loaded')
            dataset.samples = sorted(dataset.samples, key=lambda sample: sample.image.features_idx)
            samples = dataset.samples

            if dataset_type == DatasetType.TRAIN:
                max_size = options['max_train_size'] 
            elif dataset_type == DatasetType.VALIDATION:
                max_size = options["max_val_size"]   

            if(max_size == None):
                dataset.max_sample_size = len(samples)
            else:
                dataset.max_sample_size = max_size

            if dataset_type==DatasetType.TRAIN :
                answer_one_hot_mapping = dataset.answer_one_hot_mapping

            # check to make sure the samples list is sorted by image indices
            if( all(samples[i].image.features_idx <= samples[i+1].image.features_idx
                    for i in range(len(samples)-1))) :

                 print("Passed sorted sample array check")
            else:
                 assert(0)

            # log dataset size to MLFlow
            if options['logging']:
                mlflow.log_param('dataset_size', len(samples))
                mlflow.log_param('training_set_size', max_size)

            print("{} loaded from disk. Dataset size {}, Processing {} samples "
                                   .format(dataset_type, len(samples), max_size))

    except IOError:

        # If dataset does not exist create it and save it for future runs.   

        print('Creating dataset...')
        dataset = VQADataset(dataset_type, options)
        print('Preparing dataset...')

        # if the one-hot mapping is not provided, generate one
        dataset.prepare(answer_one_hot_mapping)

        print('Dataset size: %d' % dataset.size())
        print('Dataset ready.')

        print('Saving dataset...')
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('Dataset saved')

    return dataset

def plot_train_metrics(train_stats, options, plot_type='epochs'):
    """
        Generate and save figure with plot of train and validation losses.
        Currently, plot_type='epochs' is the only option supported.
        
        train_stats is a Keras History object.  History.history is a dict containing lists
    """
    
    # extract data from history dict
    train_losses = train_stats.history['loss']
    val_losses = train_stats.history['val_loss']    
    train_acc = train_stats.history['acc']
    val_acc = train_stats.history['val_acc']

    # define filenames
    d = datetime.datetime.now().isoformat()
    loss_fig_path = options['results_dir_path'] + \
        'loss_curves/losses_{}_{}_{}_{}.png'.format(plot_type, options['model_name'], options['experiment_id'], d)
    acc_fig_path = options['results_dir_path'] + \
        'acc_curves/accuracies_{}_{}_{}_{}.png'.format(plot_type, options['model_name'], options['experiment_id'], d)
    
    if plot_type == 'epochs':
        # generate and save loss plot
        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.xticks(np.arange(0, len(train_losses), step=1))
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.title('Model: {}; Experiment: {}\nRun time: {}'.format(options['model_name'], options['experiment_id'], d))
        plt.legend(('Training', 'Validation'))
        plt.savefig(loss_fig_path)
        print('Saved loss plot at: ', loss_fig_path)
        
        # clear axes and figure to reset for next plot
        plt.cla()
        plt.clf()
        
        # generate and save accuracy plot
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.xticks(np.arange(0, len(train_acc), step=1))
        plt.xlabel('Epoch Numbeer')
        plt.ylabel('Accuracy')
        plt.title('Model: {}; Experiment: {}\nRun time: {}'.format(options['model_name'], options['experiment_id'], d))
        plt.legend(('Training', 'Validation'))
        plt.savefig(acc_fig_path)
        print('Saved accuracy plot at: ', acc_fig_path)
    else:
        # Keras history object doesn't capture batch-level data; only epoch
        raise TypeError('Plot type {} does not exist'.format(plot_type))
    
    return loss_fig_path, acc_fig_path


def train(model, dataset, options, val_dataset=None):

    extended = options['extended']
    if (not extended) and (not val_dataset):
        raise ValueError('If not using the extended dataset, a validation dataset must be provided')

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']
    max_train_size = options['max_train_size']
    max_val_size   = options['max_val_size']

    losses_path = options['losses_path']
    model_weights_path = options['weights_path']
    model_weights_dir_path = options['weights_dir_path']
    model_name = options['model_name']
    early_stop_patience = options['early_stop_patience']

    # define callbacks to plug into Keras training
    loss_callback = LossHistoryCallback(losses_path)
    save_weights_callback = CustomModelCheckpoint(model_weights_path, model_weights_dir_path, model_name)
    stop_callback = EarlyStopping(patience=early_stop_patience)
    tb_logs_path = options['tb_logs_root'] + 'final/{}'.format(datetime.datetime.now())
    tensorboard_callback = TensorBoard(log_dir=tb_logs_path,
#                                        histogram_freq=1,
                                       batch_size=batch_size,
                                       write_graph=True,
                                       write_images=True)
    callbacks = [save_weights_callback, loss_callback, stop_callback, tensorboard_callback]

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

    # flag to tell batch_generator not to yield image data
    if model_name == 'text_cnn':
        is_text_only = True
    else:
        is_text_only = False
        
    print('Start training...')
    if not extended:
        train_stats = model.fit_generator(dataset.batch_generator(is_text_only), steps_per_epoch=samples_per_train_epoch//batch_size,
                            epochs=max_epochs, callbacks=callbacks,
                            validation_data=val_dataset.batch_generator(is_text_only), 
                            validation_steps=samples_per_val_epoch//batch_size,max_queue_size=20)
    else:
        train_stats = model.fit_generator(dataset.batch_generator(batch_size, split='train'), 
                            steps_per_epoch=dataset.train_size()/batch_size,
                            epochs=num_epochs, callbacks=callbacks,
                            validation_data=dataset.batch_generator(batch_size, split='val'),
                            validation_steps=dataset.val_size()//batch_size,max_queue_size=20)

    # save loss and accuracy plots
    loss_fig_path, acc_fig_path = plot_train_metrics(train_stats, options)
    
    if options['logging']:
        # each list contains one entry per epoch; log final value in each list to mlflow
        # NOTE: mlflow `log_metric` API can be called multiple times per run, so we could
        # iterate through the entire list to log values from each epoch if desired
        mlflow.log_metric('train_acc', train_stats.history['acc'][-1])
        mlflow.log_metric('val_acc', train_stats.history['val_acc'][-1])
        mlflow.log_metric('train_loss', train_stats.history['loss'][-1])
        mlflow.log_metric('val_loss', train_stats.history['val_loss'][-1])
        mlflow.log_artifact(loss_fig_path)
        mlflow.log_artifact(acc_fig_path)
        
    print('Trained!')


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


# TODO: Needs to be modified for one hot encoding of answers
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
    answer_dict = {idx: word for word, idx in dataset.answer_one_hot_mapping.items()}
    print('Reverse dictionary built')

    print('Saving results...')
    results_dict = [{'answer': answer_dict[results[idx]], 'question_id': sample.question.id}
                    for idx, sample in enumerate(dataset.samples)]
    with open(results_path, 'w') as f:
        json.dump(results_dict, f)
    print('Results saved')


# ------------------------------- CALLBACKS -------------------------------

class LossHistoryCallback(Callback):
    """
       Registering keras callbacks to be called during training iterations
       Records the losses for each batch/epoch and stores it to file
    """
    def __init__(self, results_path):
        super(LossHistoryCallback, self).__init__()
        self.train_losses = []
        self.val_losses = []
        self.results_path = results_path

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        print("Loss history: saving in file {}".format(self.results_path))
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
    """
        Save the model weights at the end of each epoch.
    """
    def __init__(self, weights_path, weights_dir_path, model_name,
                 monitor='val_loss', verbose=0, save_best_only=False, mode='auto'):

        super(CustomModelCheckpoint, self).__init__(filepath=weights_path, monitor=monitor,
                                                    verbose=verbose, save_best_only=save_best_only, mode=mode)
        self.weights_path = weights_path
        self.weights_dir_path = weights_dir_path
        self.model_name = model_name
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.last_epoch = epoch

    def on_train_end(self, logs={}):
        """
           symlink to the last epoch weights, for easy reference to the final epoch.
        """
        
        final_epoch = self.last_epoch + 1
        wt_file = self.weights_path.format(epoch=final_epoch)
        symlink = self.weights_dir_path + 'model_weights_{}_latest'.format(self.model_name)
        print('DEBUG: wt_file = ', wt_file)
        print('DEBUG: symlink = ', symlink)
        
        # TODO: symlinking needs to point to absolute path.  Can't just use '../'
        try:
            os.symlink(wt_file, symlink)
        except FileExistsError:
            # If the symlink already exist, delete and create again
            os.remove(self.weights_dir_path + 'model_weights_{}_latest'.format(self.model_name))
            os.symlink(wt_file, symlink)

# ------------------------------- ENTRY POINT -------------------------------

if __name__ == '__main__':

    # parse command line parameters and flags
    parser = argparse.ArgumentParser(description='VQA Stacked Attention Networks',
                        epilog='W266 Final Project (Fall 2018) by Rachel Ho, Ram Iyer, Mike Winton')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help = 'turn on verbose output')
    parser.add_argument('--no_logging', action='store_true',
                        help = 'turn off logging to MLFlow server')

    parser.add_argument('-b', '--batch_size', type=int,
                        help = 'set batch size (int)')
    parser.add_argument('-e', '--epochs', type=int,
                        help = 'set max number of epochs (int)')
    parser.add_argument("--max_train_size", type=int,
                        help="maximum number of training samples to use")
    parser.add_argument("--max_val_size", type=int,
                        help="maximum number of validation samples to use")

    parser.add_argument(
        '-m',
        '--model',
        type=str.lower,
        choices=ModelLibrary.get_valid_model_names(),
        default=DEFAULT_MODEL,
        help='Specify the model architecture to interact with. Each model architecture has a model name associated.'
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
    parser.add_argument(
        '-x',
        '--experiment',
        type=int,
        default=DEFAULT_EXPERIMENT,
        choices=ExperimentLibrary.get_valid_experiment_ids(),
        help='Specify the experiment configuration ID. Omitting argument or selecting 0 means no experiment.'
    )

    # Start script
    args = parser.parse_args()

    # load model options from config file
    model_options = ModelOptions().get_options()
    
    # disable logging to MLFlow server; usually only used for debugging
    if args.no_logging:
        model_options['logging'] = False

    # override default for max_epochs if specified
    if args.epochs:
        model_options['max_epochs'] = args.epochs
        
    # override default for batch_size if specified
    if args.batch_size:
        model_options['batch_size'] = args.batch_size

    # parse args with defaults
    model_options['model_name'] = args.model 
    model_options['action_type'] = args.action

    model_options['max_train_size'] = args.max_train_size
    model_options['max_val_size']   = args.max_val_size

    # set the optimizer params (learning rate, etc...)
    ModelOptions.set_optimizer_params(model_options)
    
    # process experiments last
    # load experiment attributes from json (overrides model defaults and CLI args)
    if args.experiment:
        model_options = ExperimentLibrary.get_experiment(args.experiment, model_options)
    
    # print all options before building graph
    if args.verbose:
        model_options['verbose'] = args.verbose
        pprint.pprint(model_options)

#     if model_options['logging']:
#         # create a new experiment in MLFlow if one doesn't exist
#         mlflow.set_experiment(model_options['experiment_name'])

    main(model_options)

    print('Completed!')

