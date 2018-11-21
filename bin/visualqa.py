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
from vqa.dataset.dataset import VQADataset

from vqa.experiments.experiment_select import ExperimentLibrary
from vqa.model.model_select import ModelLibrary
from vqa.model.options import ModelOptions 

# ------------------------------ GLOBALS ------------------------------
# Constants
ACTIONS = ['train', 'test']


# Defaults
DEFAULT_MODEL = "baseline"
DEFAULT_EXPERIMENT = 0
DEFAULT_ACTION = 'train'


# ------------------------------- SCRIPT FUNCTIONALITY -------------------------------

def main(options):


    print('Action: ' + options['action_type'])
    print('Model name: {}'.format(options['model_name']))

    if options['action_type'] == 'train':
        if (options['max_train_size'] != None):
            print('Training Set Size: {}'.format(options['max_train_size']))
        if (options['max_val_size'] != None):
           print('Validation set size: {}'.format(options['max_val_size']))
    elif options['action_type'] == 'test':
        if (options['max_test_size'] != None):
           print('Test set size: {}'.format(options['max_test_size']))
        
    # set paths for weights and results.
    options = ModelOptions.set_local_paths(options)

    # set numpy random seed for deterministic results
    seed = 2018
    np.random.seed(seed)
    
    # open mlflow context for logging
    if (options['logging']):
        # set experiment before starting run; else MLFlow default expt will be used
        expt_name = '{}_{}'.format(options['experiment_name'], options['experiment_id'])
        mlflow.set_experiment(expt_name)
        mlflow.start_run()
        mlflow.log_param('random_seed', seed)
        print('Enabled logging to MLFlow server for experiment_name = \"{}\"...'.format(expt_name))

    # Always load train dataset to obtain the one hot encoding indices 
    # and  max_sentence_len from it
    print('Training dataset must be loaded, even for testing (it contains OHE indices).')
    train_dataset = load_dataset(DatasetType.TRAIN,options)
    options['max_sentence_len'] = train_dataset.max_sentence_len
    answer_one_hot_mapping = train_dataset.answer_one_hot_mapping

    # Load model
    # NOTE: cannot be loaded until after dataset because it needs the vocab size
    if options['model_name'] == 'san':
        vqa_model, attention_model = ModelLibrary.get_model(options)
    else:
        vqa_model = ModelLibrary.get_model(options)
        attention_model = None
    
    # Save time-stamped model json file
    d = options['run_timestamp']
    json_path = options['saved_models_path'] + 'model_{}_expt{}_{}.json' \
        .format(options['model_name'], options['experiment_id'], d)
    with open(json_path, 'w') as json_file:
        json_file.write(vqa_model.to_json())

    # log non-empty model parameters (else mlflow crashes)
    for key, val in options.items():
        if val != '' and val != None:
            mlflow.log_param(key, val)
    print('Logged experiment params to MLFlow...')

    # Load dataset depending on the action to perform
    action = options['action_type']
    if action == 'train':
        if options['logging']:
            # log Keras model configuration
            mlflow.log_artifact(json_path)
                    
        dataset = train_dataset
        val_dataset = load_dataset(DatasetType.VALIDATION,options,answer_one_hot_mapping)
        train(vqa_model, dataset, options, val_dataset=val_dataset)
        
    elif action == 'test':
        # test set needs to be tokenized with the same tokenizer that was used in the training set
        dataset = load_dataset(DatasetType.TEST,options,answer_one_hot_mapping, tokenizer=train_dataset.tokenizer)
        test(vqa_model, dataset, options, attention_model)

    else:
        raise ValueError('The action type is unrecognized')

    if options['logging']:
        mlflow_uri = os.environ['MLFLOW_TRACKING_URI']
        mlflow_expt_id = mlflow.active_run().info.experiment_id
        mlflow_run_uuid = mlflow.active_run().info.run_uuid
        mlflow_url = '{}/#/experiments/{}/runs/{}'.format(mlflow_uri, mlflow_expt_id, mlflow_run_uuid)
        mlflow.end_run()
        print('MLFlow logs for this run are available at ->', mlflow_url)


def load_dataset(dataset_type, options, answer_one_hot_mapping=None, tokenizer=None):
    
    """
        Load the dataset from disk if available. If not, build it from the questions/answers json and image embeddings
        If this is the training dataset, retrieve the answer one hot mapping from disk or re-create it.
    """ 

    dataset_path = ModelOptions.get_dataset_path(options,dataset_type)
    # if this isn't a training dataset, the answer one hot indices and tokenizer are expected to be available
    if (dataset_type != DatasetType.TRAIN):
        assert(answer_one_hot_mapping != None) 
        assert(tokenizer != None)

    # If pickle file is older than dataset.py, delete and recreate
    print('Checking timestamp on dataset -> {}'.format(dataset_path))
    dataset_py_path = os.path.abspath('../vqa/dataset/dataset.py')
    if os.path.isfile(dataset_path) and \
    os.path.getmtime(dataset_path) < os.path.getmtime(dataset_py_path):
        to_delete = input('\nWARNING: Dataset (which also contains the Tokenizer) is outdated.  Remove it (y/n)? ')
        if len(to_delete) > 0 and to_delete[:1] == 'y':
            os.remove(dataset_path)
            print('Dataset was outdated. Removed -> ', dataset_path)
        else:
            print('Continuing with pre-existing dataset.')

    try:
        with open(dataset_path, 'rb') as f:
            print('Loading dataset from {}'.format(dataset_path))
            dataset = pickle.load(f)
            print('Dataset loaded')

        options['n_vocab'] = dataset.vocab_size
            
        dataset.samples = sorted(dataset.samples, key=lambda sample: sample.image.features_idx)
        samples = dataset.samples

        if dataset_type == DatasetType.TRAIN:
            max_size = options['max_train_size'] 
            if options['logging']:
                mlflow.log_param('train_dataset_size', len(samples))
                mlflow.log_param('max_train_size', max_size)
        elif dataset_type == DatasetType.VALIDATION:
            max_size = options["max_val_size"]   
            if options['logging']:
                mlflow.log_param('val_dataset_size', len(samples))
                mlflow.log_param('max_val_size', max_size)
        elif dataset_type == DatasetType.TEST:
            max_size = options["max_test_size"]   
#             if options['logging']:
#                 # TODO: log this at better point for val_test_split
#                 mlflow.log_param('test_dataset_size', len(samples))
#                 mlflow.log_param('max_test_size', max_size)
        else:
            max_size = None

        if(max_size == None):
            dataset.max_sample_size = len(samples)
        else:
            dataset.max_sample_size = min(max_size,len(samples))

        # check to make sure the samples list is sorted by image indices
        if(all(samples[i].image.features_idx <= samples[i+1].image.features_idx \
               for i in range(len(samples)-1))):
            print("Passed sorted sample array check")
        else:
            assert(0)

        print("{} loaded from disk. Dataset size {}, processing limited to max_size = {}." \
              .format(dataset_type, len(samples), max_size))

    except IOError:

        # If dataset file does not exist create and save it for future runs.   
        print('Creating dataset...')
        dataset = VQADataset(dataset_type, options)

        # as part of preparation, if one-hot mapping is not provided, generate it.
        # both are expected to be provided if this is a test set
        print('Preparing dataset...')
        dataset.prepare(answer_one_hot_mapping, tokenizer)

        # TODO: fix the n_vocab logic when we're ready to do standalone test sets.  Currently,
        # n_vocab will never get set if a training set isn't processeed first.

        # n_vocab isn't set until it's calculated for training dataset
        if dataset_type==DatasetType.TRAIN:
            options['n_vocab'] = dataset.vocab_size
        else:
            dataset.vocab_size = options['n_vocab']

        print('Dataset prepared. Samples: {}. Vocab size: {}'.format(dataset.size(), options['n_vocab']))
        print('Saving dataset...')
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('Dataset saved to {}'.format(dataset_path))

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
    d = options['run_timestamp']
    loss_fig_path = options['results_dir_path'] + \
        'loss_curves/losses_{}_{}_expt{}_{}.png'.format(plot_type, options['model_name'], options['experiment_id'], d)
    acc_fig_path = options['results_dir_path'] + \
        'acc_curves/accuracies_{}_{}_expt{}_{}.png'.format(plot_type, options['model_name'], options['experiment_id'], d)
    
    # make sure directories exist before trying to save to them
    loss_fig_dir = os.path.dirname(os.path.abspath(loss_fig_path))
    print("Saving loss plots to directory -> ", loss_fig_dir)
    if not os.path.isdir(loss_fig_dir):
        os.mkdir(loss_fig_dir)
    acc_fig_dir = os.path.dirname(os.path.abspath(acc_fig_path))
    print("Saving accuracy plots to directory -> ", acc_fig_dir)
    if not os.path.isdir(acc_fig_dir):
        os.mkdir(acc_fig_dir)
    
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
        plt.xlabel('Epoch Number')
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

    if not val_dataset:
        raise ValueError('A validation dataset must be provided')

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']
    max_train_size = options['max_train_size']
    max_val_size   = options['max_val_size']

    losses_path = options['losses_path']
    model_weights_path = options['weights_path']
    print('Saving weights from final epoch to ->', model_weights_path)
    model_weights_dir_path = options['weights_dir_path']
    model_name = options['model_name']
    experiment_id = options['experiment_id']
    early_stop_patience = options['early_stop_patience']

    # define callbacks to plug into Keras training
    loss_callback = LossHistoryCallback(losses_path)
    save_weights_callback = CustomModelCheckpoint(model_weights_path, model_weights_dir_path, model_name, experiment_id)
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
    # flag to tell batch_generator not to yield sentence data
    if model_name == 'vggnet_only':
        is_img_only = True
    else:
        is_img_only = False
        
    print('Start training...')
    train_stats = model.fit_generator(dataset.batch_generator(is_text_only, is_img_only),
                                      steps_per_epoch=samples_per_train_epoch//batch_size,
                                      epochs=max_epochs, callbacks=callbacks,
                                      validation_data=val_dataset.batch_generator(is_text_only, is_img_only), 
                                      validation_steps=samples_per_val_epoch//batch_size,max_queue_size=20)

    # save loss and accuracy plots to local disk
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
    
    # Save y_proba for validation set to disk.  To do this it we have to run test() 
    # with the validation dataset; it doesn't appear possible to directly export
    # the keras layer.output tensor to a numpy array.  keras.backend.eval raises an exception.
    if options.get('predict_on_validation_set', False) and val_dataset != None:
        print('Generating and saving predictions for validation dataset...')
        # change action type and set paths for weights and results
        options['action_type'] = 'test'
        options = ModelOptions.set_local_paths(options)
        # change dataset_type to prevent shuffling during batch generation; otherwise 
        # it won't be possible to compare to true lablels
        val_dataset.dataset_type = DatasetType.TEST
        test(model, val_dataset, options)
        val_dataset.dataset_type = DatasetType.VALIDATION

def test(model, dataset, options, attention_model=None):

    weights_path = options['weights_path']
    results_json_path  = options['results_json_path']
    probabilities_path = options['probabilities_path']
    batch_size    = options['batch_size']
    max_test_size = options['max_test_size']

    print('Loading weights from {}...'.format(weights_path))
    model.load_weights(weights_path)
    print('Weights loaded')
    print('Predicting...')

    if(max_test_size != None):
        test_dataset_size = min(max_test_size, len(dataset.samples)) 
    else:
        test_dataset_size = len(dataset.samples)

    results = model.predict_generator(dataset.batch_generator(),
                                      steps=test_dataset_size//batch_size + 1,
                                      verbose=1)

    #resize results as it might have been padded for being an exact multiple of batch size
    results = results[:test_dataset_size]
    dataset.samples = dataset.samples[:test_dataset_size]

    print('Answers predicted for {} samples'.format(test_dataset_size))
    
    # define filename for y_proba file
    d = options['run_timestamp']
    y_proba_path = options['results_dir_path'] + \
        'pred_probs/y_proba_{}_expt{}_{}.p'.format(options['model_name'], options['experiment_id'], d)

    # make sure directory exists before trying to save to it
    y_proba_dir = os.path.dirname(os.path.abspath(y_proba_path))
    print('Saving predicted probabilities (shape = {}) to directory -> {}'.format(results.shape, y_proba_dir))
    if not os.path.isdir(y_proba_dir):
        os.mkdir(y_proba_dir)
    
    # save to disk (and also to MLFlow if logging is enabled)
    pickle.dump(results, open(y_proba_path, 'wb'))
    if options['logging']:
        mlflow.log_artifact(y_proba_path)
    print('Resulting predicted probabilities saved -> ', y_proba_path)

    print('Transforming probabilities to predicted labels (argmax)...')
    y_pred_ohe = list(np.argmax(results, axis=1))  # Max index evaluated on rows (1 row = 1 sample)

    print('Building reverse word dictionary from one-hot answer mappings...')
    ohe_to_answer_str = {idx: word for word, idx in dataset.answer_one_hot_mapping.items()}

    if options['val_test_split']:
        print('Saving results (questions, true answers, and predictions)...')
        no_answers = 0
        results_dict = []
        for idx, sample in enumerate(dataset.samples):
            if not hasattr(sample,'answer'):
                no_answers += 1
            else:
                results_dict.append({'predicted_answer': ohe_to_answer_str[y_pred_ohe[idx]], 
                                     'question_id': sample.question.id,
                                     'question_str': sample.question.question_str,
                                     'question_type': sample.answer.question_type,
                                     'image_id': sample.question.image_id,
                                     'answer_id': sample.answer.id,
                                     'answer_str': sample.answer.answer_str,
                                     'answer_type': sample.answer.answer_type,
                                     'annotations': sample.answer.annotations
                                    })
        print('Discarded {} samples without answers...'.format(no_answers))
    else:
        print('Saving results (questions and predictions)...')
        results_dict = [{'predicted_answer': ohe_to_answer_str[y_pred_ohe[idx]], 
                         'question_id': sample.question.id,
                         'question_str': sample.question.question_str,
                         'image_id': sample.question.image_id
                        }
                        for idx, sample in enumerate(dataset.samples)]
    with open(results_json_path, 'w') as f:
        json.dump(results_dict, f)
    if options['logging']:
        mlflow.log_artifact(results_json_path)
    print('Results saved to -> ', results_json_path)

    # save attention probabilities to disk
    if not attention_model == None:
        # list will have one numpy array for each attention_layer output by the model
        attention_probabilities = attention_model \
            .predict_generator(dataset.batch_generator(), steps=test_dataset_size//batch_size + 1, verbose=1)

        print('Attention probabilities extracted from {} attention layers'.format(len(attention_probabilities)))
        with h5py.File(probabilities_path, 'a') as f:
            # len(attention_probability) should = n_attention_layers
            for i in range(len(attention_probabilities)):
                dataset_name = 'attention_probabilites{}'.format(i)
                f.create_dataset(dataset_name, data=attention_probabilities[i])
        print('Attention_probabilities saved ->', probabilities_path)
        if options['logging']:
            mlflow.log_artifact(probabilities_path)

    
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
    def __init__(self, weights_path, weights_dir_path, model_name, experiment_id, 
                 monitor='val_loss', verbose=0, save_best_only=False, mode='auto'):

        super(CustomModelCheckpoint, self).__init__(filepath=weights_path, monitor=monitor,
                                                    verbose=verbose, save_best_only=save_best_only, mode=mode)
        self.weights_path = weights_path
        self.weights_dir_path = weights_dir_path
        self.model_name = model_name
        self.experiment_id = experiment_id
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        # save after every epoch to enable restarting at that epoch after a crash
        super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.last_epoch = epoch

    def on_train_end(self, logs={}):
        """
           symlink to the last epoch weights, for easy reference to the final epoch.
        """
        
        final_epoch = self.last_epoch + 1 # Keras doesn't increment during final epoch
        for i in range(1, final_epoch + 1):
            wt_file = os.path.abspath(self.weights_path.format(epoch=i))
            if i < final_epoch:
                print('Deleting temporary weight file ->', wt_file)
                os.remove(wt_file)
            else:
                symlink = os.path.abspath(self.weights_dir_path + 'model_weights_{}_expt{}_latest' \
                                          .format(self.model_name, self.experiment_id))

        if options['logging']:
            print("Transferring model weights to mlflow..")
            mlflow.log_artifact(wt_file)
            print("Done..")
       
        try:
            os.symlink(wt_file, symlink)
        except FileExistsError:
            # If the symlink already exist, delete and create again
            os.remove(symlink)
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
    parser.add_argument('--predict_on_validation_set', action='store_true',
                        help = 'after training, run `model.predict()` on validation dataset')

    parser.add_argument('-b', '--batch_size', type=int,
                        help = 'set batch size (int)')
    parser.add_argument('-e', '--epochs', type=int,
                        help = 'set max number of epochs (int)')
    parser.add_argument("--max_train_size", type=int,
                        help="maximum number of training samples to use")
    parser.add_argument("--max_val_size", type=int,
                        help="maximum number of validation samples to use")
    parser.add_argument("--max_test_size", type=int,
                        help="maximum number of test samples to use")

    parser.add_argument(
        '-d',
        '--dataset',
        type=str.lower,
        choices=['v1', 'v2'],
        default='v1',
        help='Specify the VQA dataset to use (v1 or v2).'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str.lower,
        choices=ModelLibrary.get_valid_model_names(),
        default=DEFAULT_MODEL,
        help='Specify the model architecture to interact with. Each model ' + \
             'architecture has a model name associated.'
    )
    parser.add_argument(
        '-o',
        '--optimizer',
        type=str.lower,
        choices=['adam', 'rmsprop', 'sgd'],
        default='sgd',
        help='Specify the optimizer to use (adam, rmsprop, sgd).'
    )
    parser.add_argument(
        '-a',
        '--action',
        choices=ACTIONS,
        default=DEFAULT_ACTION,
        help='Which action should be perform on the model. By default, training will be done'
    )
    parser.add_argument(
        '-x',
        '--experiment',
        type=int,
        default=DEFAULT_EXPERIMENT,
        help='Specify the experiment configuration ID. Omitting argument or ' + \
             'selecting 0 means no experiment. Program will look for a corresponding file ' + \
             'at \"\experiments\<username>_experiment_<id>.json\".'
    )

    # Start script
    args = parser.parse_args()

    # load model options from config file
    options = ModelOptions().get_options()
    
    # disable logging to MLFlow server; usually only used for debugging
    if args.no_logging:
        options['logging'] = False

    # override default for max_epochs if specified
    if args.epochs:
        options['max_epochs'] = args.epochs
        
    # override default for batch_size if specified
    if args.batch_size:
        options['batch_size'] = args.batch_size

    # parse args with defaults; this is used in prefixing only for v2
    if args.dataset != 'v1':
        options['dataset'] = args.dataset
        
    options['model_name'] = args.model 
    options['optimizer'] = args.optimizer
    options['action_type'] = args.action
    
    if args.action == 'train' and args.predict_on_validation_set:
        options['predict_on_validation_set'] = True

    options['max_train_size'] = args.max_train_size
    options['max_val_size']   = args.max_val_size
    options['max_test_size']   = args.max_test_size

    # set the optimizer params (learning rate, etc...)
    ModelOptions.set_optimizer_params(options)
    
    # process experiments last
    # load experiment attributes from json (overrides model defaults and CLI args)
    if args.experiment:
        options = ExperimentLibrary.get_experiment(args.experiment, options)
    
    # define run_timestamp to be used in all saved artifacts
    run_timestamp = datetime.datetime.now().isoformat('-', timespec='seconds')
    options['run_timestamp'] = run_timestamp
    
    # print all options before building graph
    if args.verbose:
        options['verbose'] = args.verbose
        pprint.pprint(options)

    main(options)

    print('Completed!')

