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

# ------------------------------ GLOBALS ------------------------------
# Constants
ACTIONS = ['train', 'val', 'test', 'eval']
VOCABULARY_SIZE = 20000
NUM_EPOCHS = 40
BATCH_SIZE = 128

# Paths
DATA_PATH = '../data/'
DATA_ROOT = '/home/ram_iyer/vqa_data/'
PREPROCESSED_PATH = DATA_PATH + 'preprocessed/'
TOKENIZER_PATH = PREPROCESSED_PATH + 'tokenizer.p'
FEATURES_DIR_PATH = DATA_ROOT + 'images/mscoco/embeddings/'
WEIGHTS_DIR_PATH = '../models/weights/'
RESULTS_DIR_PATH = '../results/'

# Config
CONFIG_TRAIN = {
    'dataset_type': DatasetType.TRAIN,
    'dataset_path': PREPROCESSED_PATH + 'train_dataset.p',
    'questions_path': DATA_ROOT + 'questions/train/v2_OpenEnded_mscoco_train2014_questions.json',
    'annotations_path': DATA_ROOT + 'annotations/train/v2_mscoco_train2014_annotations.json'
}
CONFIG_VAL = {
    'dataset_type': DatasetType.VALIDATION,
    'dataset_path': PREPROCESSED_PATH + 'validate_dataset.p',
    'questions_path': DATA_ROOT + 'questions/val/v2_OpenEnded_mscoco_val2014_questions.json',
    'annotations_path': DATA_ROOT + '/annotations/val/v2_mscoco_val2014_annotations.json'
}
CONFIG_TEST = {
    'dataset_type': DatasetType.TEST,
    'dataset_path': PREPROCESSED_PATH + 'test_dataset.p',
    'questions_path': DATA_ROOT + 'questions/v2_OpenEnded_mscoco_test2015_questions.json',
    'annotations_path': None
}
CONFIG_EVAL = {
    'dataset_type': DatasetType.EVAL,
    'dataset_path': PREPROCESSED_PATH + 'eval_dataset.p',
    'questions_path': DATA_ROOT + 'questions/val/v2_mscoco_val2014_annotations.json',
    'annotations_path': None
}

# Defaults
DEFAULT_MODEL = 1 
#DEFAULT_MODEL = 1  max(ModelLibrary.get_valid_model_nums())
DEFAULT_ACTION = 'train'


# ------------------------------- SCRIPT FUNCTIONALITY -------------------------------

def main(action, model_num, extended,max_train_size,max_val_size):
    print('Action: ' + action)
    print('Model number: {}'.format(model_num))
    print('Extended: {}'.format(extended))
    if (max_train_size != None):
        print('Training Set Size: {}'.format(max_train_size))
    if (max_val_size != None):
       print('Validation set size: {}'.format(max_val_size))


    # set numpy random seed for deterministic results
    np.random.seed(2018)
    
    # Always load train dataset to obtain the question_max_len from it
    train_dataset = load_dataset(CONFIG_TRAIN['dataset_type'], CONFIG_TRAIN['dataset_path'],
                                 CONFIG_TRAIN['questions_path'], CONFIG_TRAIN['annotations_path'], FEATURES_DIR_PATH,
                                 TOKENIZER_PATH,max_train_size)
    question_max_len = train_dataset.question_max_len

    # Load model
    vqa_model = ModelLibrary.get_model(model_num, vocabulary_size=VOCABULARY_SIZE, question_max_len=question_max_len)

    # Load dataset depending on the action to perform
    if action == 'train':
        dataset = train_dataset
        val_dataset = load_dataset(CONFIG_VAL['dataset_type'], CONFIG_VAL['dataset_path'],
                                   CONFIG_VAL['questions_path'], CONFIG_VAL['annotations_path'], FEATURES_DIR_PATH,
                                   TOKENIZER_PATH,max_val_size)
        if extended:
            extended_dataset = MergeDataset(train_dataset, val_dataset)
            weights_path = WEIGHTS_DIR_PATH + 'model_weights_' + str(model_num) + '_ext.{epoch:02d}.hdf5'
            losses_path = RESULTS_DIR_PATH + 'losses_{}_ext.h5'.format(model_num)
            train(vqa_model, extended_dataset, model_num, weights_path, losses_path,max_train_size,max_val_size, extended=True)
        else:
            weights_path = WEIGHTS_DIR_PATH + 'model_weights_' + str(model_num) + '.{epoch:02d}.hdf5'
            losses_path = RESULTS_DIR_PATH + 'losses_{}.h5'.format(model_num)
            train(vqa_model, dataset, model_num, weights_path, losses_path,max_train_size,max_val_size, val_dataset=val_dataset)

    elif action == 'val':
        dataset = load_dataset(CONFIG_VAL['dataset_type'], CONFIG_VAL['dataset_path'],
                               CONFIG_VAL['questions_path'], CONFIG_VAL['annotations_path'], FEATURES_DIR_PATH,
                               TOKENIZER_PATH,max_val_size)
        weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(model_num)
        validate(vqa_model, dataset, weights_path)
    elif action == 'test':
        dataset = load_dataset(CONFIG_TEST['dataset_type'], CONFIG_TEST['dataset_path'],
                               CONFIG_TEST['questions_path'], CONFIG_TEST['annotations_path'], FEATURES_DIR_PATH,
                               TOKENIZER_PATH)
        if not extended:
            weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(model_num)
            results_path = RESULTS_DIR_PATH + 'test2015_results_{}.json'.format(model_num)
        else:
            weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}_ext'.format(model_num)
            results_path = RESULTS_DIR_PATH + 'test2015_results_{}_ext.json'.format(model_num)
        test(vqa_model, dataset, weights_path, results_path)
    elif action == 'eval':
        dataset = load_dataset(CONFIG_EVAL['dataset_type'], CONFIG_EVAL['dataset_path'],
                               CONFIG_EVAL['questions_path'], CONFIG_EVAL['annotations_path'], FEATURES_DIR_PATH,
                               TOKENIZER_PATH)
        if not extended:
            weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(model_num)
            results_path = RESULTS_DIR_PATH + 'val2014_results_{}.json'.format(model_num)
        else:
            weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}_ext'.format(model_num)
            results_path = RESULTS_DIR_PATH + 'val2014_results_{}_ext.json'.format(model_num)
        test(vqa_model, dataset, weights_path, results_path)
    else:
        raise ValueError('The action you provided do not exist')


def load_dataset(dataset_type, dataset_path, questions_path, annotations_path, features_dir_path, tokenizer_path, max_size=None):
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
        dataset = VQADataset(dataset_type, questions_path, annotations_path, features_dir_path,
                             tokenizer_path, max_sample_size=max_size ,vocab_size=VOCABULARY_SIZE)
        print('Preparing dataset...')
        dataset.prepare()
        print('Dataset size: %d' % dataset.size())
        print('Dataset ready.')

        print('Saving dataset...')
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('Dataset saved')

    return dataset


def train(model, dataset, model_num, model_weights_path, losses_path,max_train_size,max_val_size, val_dataset=None, extended=False):
    if (not extended) and (not val_dataset):
        raise ValueError('If not using the extended dataset, a validation dataset must be provided')

    loss_callback = LossHistoryCallback(losses_path)
    save_weights_callback = CustomModelCheckpoint(model_weights_path, WEIGHTS_DIR_PATH, model_num)
    stop_callback = EarlyStopping(patience=5)

    if(max_train_size != None):
        samples_per_train_epoch = min(max_train_size,dataset.size()) 
        samples_per_train_epoch = max(BATCH_SIZE,samples_per_train_epoch)
    else:
        samples_per_train_epoch = dataset.size()

    if(val_dataset != None):
        if(max_val_size !=None ):
            samples_per_val_epoch = min(max_val_size,val_dataset.size()) 
            samples_per_val_epoch = max(BATCH_SIZE,samples_per_val_epoch)
        else:
            samples_per_val_epoch = val_dataset.size()

    print('Start training...')
    if not extended:
        model.fit_generator(dataset.batch_generator(BATCH_SIZE), steps_per_epoch=samples_per_train_epoch//BATCH_SIZE, epochs=NUM_EPOCHS,
                            callbacks=[save_weights_callback, loss_callback, stop_callback],
                            validation_data=val_dataset.batch_generator(BATCH_SIZE), 
                            validation_steps=samples_per_val_epoch//BATCH_SIZE,max_queue_size=20,use_multiprocessing=True)
    else:
        model.fit_generator(dataset.batch_generator(BATCH_SIZE, split='train'), steps_per_epoch=dataset.train_size()/BATCH_SIZE,
                            epochs=NUM_EPOCHS, callbacks=[save_weights_callback, loss_callback, stop_callback],
                            validation_data=dataset.batch_generator(BATCH_SIZE, split='val'),
                            validation_steps=dataset.val_size()/BATCH_SIZE,max_queue_size=20,use_multiprocessing=True)
    print('Trained')


def validate(model, dataset, weights_path):
    print('Loading weights...')
    model.load_weights(weights_path)
    print('Weights loaded')
    print('Start validation...')
    result = model.evaluate_generator(dataset.batch_generator(BATCH_SIZE), val_samples=dataset.size())
    print('Validated. Loss: {}'.format(result))

    return result


def test(model, dataset, weights_path, results_path):

    ## Not implemented - TODO check if needed
    """
    print('Loading weights...')
    model.load_weights(weights_path)
    print('Weights loaded')
    print('Predicting...')
    images, questions = dataset.get_dataset_input()
    results = model.predict([images, questions], BATCH_SIZE)
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
    """

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
    parser = argparse.ArgumentParser(description='Main entry point to interact with the VQA module')
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
        '-e',
        '--extended',
        action='store_true',
        help='Add this flag if you want to use the extended dataset, this is, use part of the validation dataset to'
             'train your model. Only valid for the --action=train'
    )
    parser.add_argument("--max_train_size",type=int,help="maximum number of training samples to use")
    parser.add_argument("--max_val_size",type=int,help="maximum number of validation samples to use")
    # Start script
    args = parser.parse_args()
    main(args.action, args.model, args.extended,args.max_train_size,args.max_val_size)
