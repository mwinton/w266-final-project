import getpass
import datetime
import os

from collections import OrderedDict
from keras.optimizers import Adam, RMSprop, SGD

from .. dataset.types import DatasetType

class ModelOptions(object):

    def __init__(self):
        ''' init function.  This class stores hyperparameters for the model. '''
        
        self.options = OrderedDict()
        
        # Base file system paths
        user_name = getpass.getuser()
        self.options['user_name'] = user_name
        data_root = "/home/" + user_name + "/vqa_data/"
        self.options['data_root'] = data_root
        self.options['tb_logs_root'] = "/home/" + user_name + "/logs/"
        self.options['experiments_path'] =  os.path.abspath("../vqa/experiments") + '/'

        # Image embedding root
        image_embed_root = data_root+'images/mscoco/embeddings/vgg16/'
        self.options['images_embed_train_path'] = image_embed_root + 'train.hdf5'
        self.options['images_embed_val_path']   = image_embed_root + 'val.hdf5'
        self.options['images_embed_test_path'] = image_embed_root + 'test.hdf5'
        
#         # Text embedding root
#          self.options['glove_path'] = ''

        # Both VQA v1 and v2 use the same images and question types, so OK to set here
        self.options['images_train_root_path'] = data_root+'images/mscoco/train2014/'
        self.options['images_val_root_path']   = data_root+'images/mscoco/val2014/'
        self.options['images_test_root_path']   = data_root+'images/mscoco/test2015/'
        self.options['question_types_path']  = data_root+'questiontypes/mscoco_question_types.txt'

#         # to be polulated later once all options are parsed
#         self.options['weights_path']  = ""
#         self.options['results_path'] = ""
#         self.options['losses_path']  = ""

        # Which dataset to use (use empty string for easier string comparisons)
        self.options['dataset'] = ''

        # Model selection parameter
        self.options['model_name'] = 'san'    # default is Yang's SAN model
        
        # Type of action to be performed
        self.options['action_type'] = "train"

        # Type of validation/test split
        # if true, a repeatable half of the original validation set is split into the training set
        self.options['val_test_split'] = True

        # Experiment to be performed
        self.options['experiment_id'] = 0                  # default is no experiment
        self.options['experiment_name'] = 'Default_Expt'   # name to display in MLFlow
        
        # Image model parameters
        self.options['n_image_embed'] = 512     # VGGNet
        self.options['n_image_regions'] = 196   # 14x14 regions
        self.options['mscoco_dim'] = 256        # x, y dimension of photos
        self.options['vggnet_input_dim'] = 448  # expected x, y dim of VGGNet
        self.options['image_depth'] = 3         # 3 color channels (RGB)
        self.options['start_with_image_embed'] = True

        # Text model parameters
        # Keras tokenizer: 18364 (Q+A) or 13681 (Q-only); Yang: 13746
        # self.options['n_vocab'] = 18364           
        # self.options['max_sentence_len'] = 22     # actual max is 22, so don't override it   
        
        self.options['n_sent_embed'] = 500          # TODO: change this when we use GloVe
        self.options['sent_init_type'] = 'uniform'  # TODO: experiment with GloVe
        self.options['sent_init_range'] = 0.01

        self.options['n_filters_unigram'] = 256
        self.options['n_filters_bigram'] = 512
        self.options['n_filters_trigram'] = 512

        # Attention layer parameters
        # Some of the simpler models don't use these, so don't set by default.  That allows code to 
        # check whether attention layers are part of the model by looking at options['n_attention_layers']
        if self.options['model_name'] == 'san':
            self.options['n_attention_layers'] = 2
            self.options['n_attention_input'] = self.options['n_filters_unigram'] \
                                              + self.options['n_filters_bigram'] \
                                              + self.options['n_filters_trigram']
            self.options['n_attention_features'] = 512
            self.options['attention_merge_type'] = 'addition'
            self.options['attention_dropout_ratio'] = 0.5  # Yang

        # Classification layer parameters
        self.options['n_answer_classes'] = 1001  # 1000 real classes + <unk>  

        # Training parameters
        self.options['batch_size'] = 100
        self.options['max_epochs'] = 50
        self.options['early_stop_patience'] = 5
        self.options['max_train_size'] = None # interpreted as full size of training set unless overridden
        self.options['max_val_size'] = None # interpreted as full validation set size unless overridden
        self.options['max_test_size'] = None # interpreted as full test set size unless overridden
        self.options['extended']  = False # use train+val set for training

        # When changing the optimize, also update set_optimizer_params()
        self.options['optimizer'] = 'sgd'       

        # Regularization / weight decay parameters (assumed to go with an l2 regularizer)
        self.options['regularizer'] = True
        self.options['weight_decay'] = 0.0005            # in Keras we have to apply layer-by-layer

        # MLFlow logging parameters
        if 'MLFLOW_TRACKING_URI' in os.environ:
            self.options['logging'] = True
        else:
            self.options['logging'] = False
        self.options['verbose'] = False
        self.options['log_interval'] = 500
        self.options['display_interval'] = 1000    
    
    def get_options(self):
        ''' return ordered dict containing all model options '''
        return self.options

    @staticmethod
    def get_dataset_path(options,datasetType):
        """
           returns the dataset path for the given datasetType
        """
        selector = {
            DatasetType.TRAIN      : options["train_dataset_path"],
            DatasetType.VALIDATION : options["val_dataset_path"],
            DatasetType.TEST       : options["test_dataset_path"],
            DatasetType.EVAL       : options["eval_dataset_path"]
        }
        return selector.get(datasetType)

    @staticmethod
    def get_questions_path(options,datasetType):
        """
           returns the questions path for the given datasetType
        """
        selector = {
            DatasetType.TRAIN      : options["questions_train_path"],
            DatasetType.VALIDATION : options["questions_val_path"],
            DatasetType.TEST       : options["questions_test_path"],
            DatasetType.EVAL       : options["questions_val_path"]
        }
        if (options['val_test_split']):
            selector[DatasetType.TEST] = options['questions_val_path']
        return selector.get(datasetType)

    @staticmethod
    def get_pairs_path(options, datasetType):
        """
           returns the questions path for the given datasetType (if using v2 dataset)
        """
        if options['dataset'] == 'v2':
            selector = {
                DatasetType.TRAIN      : options["pairs_train_path"],
                DatasetType.VALIDATION : options["pairs_val_path"],
                DatasetType.TEST       : None,
                DatasetType.EVAL       : None
            }
            if (options['val_test_split']):
                selector[DatasetType.TEST] = options['pairs_val_path']

            return selector.get(datasetType)
        else:
            return None
        
    @staticmethod
    def get_annotations_path(options,datasetType):
        """
           returns the dataset path for the given datasetType
        """
        selector = {
            DatasetType.TRAIN      : options["annotations_train_path"],
            DatasetType.VALIDATION : options["annotations_val_path"],
            DatasetType.TEST       : None,
            DatasetType.EVAL       : None
        }
        if (options['val_test_split']):
            selector[DatasetType.TEST] = options['annotations_val_path']

        return selector.get(datasetType)


    @staticmethod
    def get_images_embed_path(options,datasetType):
        """
           returns the dataset path for the given datasetType
        """
        selector = {
            DatasetType.TRAIN      : options["images_embed_train_path"],
            DatasetType.VALIDATION : options["images_embed_val_path"],
            DatasetType.TEST       : options["images_embed_test_path"],
            DatasetType.EVAL       : options["images_embed_val_path"]
        }
        if (options['val_test_split']):
            selector[DatasetType.TEST] = options['images_embed_val_path']

        return selector.get(datasetType)

    @staticmethod
    def get_images_path(options,datasetType):
        """
           returns the dataset path for the given datasetType
        """
        selector = {
            DatasetType.TRAIN      : options["images_train_root_path"],
            DatasetType.VALIDATION : options["images_val_root_path"],
            DatasetType.TEST       : options["images_test_root_path"],
            DatasetType.EVAL       : options["images_val_root_path"]
        }
        if (options['val_test_split']):
            selector[DatasetType.TEST] = options['images_val_root_path']
            
        return selector.get(datasetType)

    @staticmethod
    def set_local_paths(options):
        """
            returns the weights and losses paths based on the dataset and 
            the -extended option
        """
        action    = options['action_type']
        model_name = options["model_name"]
        extended  = options['extended']
        data_root = options['data_root']
        
        # VQAv2 json files have a `v2_` filename prefix
        prefix = ''
        if options['dataset'] == 'v2': prefix = 'v2_'

        # Also neeed to annotate extended dataset
        suffix = ''
        if extended: suffix = "_ext"

        # Need to prefix v2 files with `v2_`
        options['questions_train_path'] = data_root + \
            'questions/' + prefix + 'OpenEnded_mscoco_train2014_questions.json'
        options['questions_val_path']   = data_root + \
            'questions/' + prefix + 'OpenEnded_mscoco_val2014_questions.json'
        options['questions_test_path']  = data_root + \
            'questions/' + prefix + 'OpenEnded_mscoco_test2015_questions.json'

        # complementary pairs are only relevant for v2 dataset
        if options['dataset'] == 'v2':
            options['pairs_train_path'] = data_root + \
                'pairs/' + prefix + 'mscoco_train2014_complementary_pairs.json'
            options['pairs_val_path'] = data_root + \
                'pairs/' + prefix + 'mscoco_val2014_complementary_pairs.json'
        
        options['annotations_train_path'] = data_root + \
            'annotations/' + prefix + 'mscoco_train2014_annotations.json'
        options['annotations_val_path']   = data_root + \
            'annotations/' + prefix + 'mscoco_val2014_annotations.json'

        # Files created during train/val/test phases
        # NOTE: os.path.abspath drops trailing slash, so need to re-add
        options['local_data_path']  =  os.path.abspath("../data/preprocessed") + '/'
        options['weights_dir_path'] =  os.path.abspath("../saved_models/weights") + '/'
        options['results_dir_path'] =  os.path.abspath("../results") + '/' 
        options['saved_models_path'] = os.path.abspath('../saved_models/json') + '/'

        # create directories if they don't exist
        os.makedirs(options['local_data_path'],   exist_ok=True)
        os.makedirs(options['weights_dir_path'],  exist_ok=True)
        os.makedirs(options['results_dir_path'],  exist_ok=True)
        os.makedirs(options['saved_models_path'], exist_ok=True) 

        local_data_path = options['local_data_path']

        # We also need to prefix our generated pickle files by dataset
        options['tokenizer_path']     = local_data_path + prefix + 'tokenizer.p'
        options['train_dataset_path'] = local_data_path + prefix + 'train_dataset.p'
        options['val_dataset_path']   = local_data_path + prefix + 'validate_dataset.p'
        options['test_dataset_path']  = local_data_path + prefix + 'test_dataset.p'
        options['eval_dataset_path']  = local_data_path + prefix + 'eval_dataset.p'

        weights_dir_path = options['weights_dir_path']
        results_dir_path = options['results_dir_path']
        
        # get run- and experiment-dependent filename annotations
        d = options['run_timestamp']
        expt = options.get('experiment_id', 0)
        
        if (action == "train"):
            # timestamp the weights; later we create a symlink to the most recent set (for prediction)
            weights_dir_path = weights_dir_path + prefix + 'model_weights_{}{}_expt{}_{}' \
                .format(model_name, suffix, expt, d)
            # Keras requires that we must use named `epoch` placeholder in format string
            options["weights_path"] = weights_dir_path + '.{epoch:02d}.hdf5'
            
            # timestamp the losses_path for logging purposes
            options['losses_path'] = results_dir_path + prefix + 'losses_{}{}_expt{}_{}.hdf5' \
                .format(model_name, suffix, expt, d)
            
        elif (action == "val" ):
            options["weights_path"] = weights_dir_path + prefix + 'model_weights_{}{}_expt{}_{}' \
                .format(model_name, suffix, expt, d)

        elif (action == "test"):
            options['weights_path'] = weights_dir_path + prefix + 'model_weights_{}{}_expt{}_latest' \
                .format(model_name, suffix, expt )
            options['results_path'] = results_dir_path + prefix + 'test2015_results_{}{}_expt{}_{}.json' \
                .format(model_name, suffix, expt, d)
        
        else:
            # action type is eval
            options['weights_path'] = weights_dir_path + prefix + 'model_weights_{}{}_expt{}_{}' \
                .format(model_name, suffix, expt, d)
            options['results_path'] = results_dir_path + prefix + 'val2014_results_{}{}_expt{}_{}.json' \
                .format(model_name, suffix, expt, d)

        return options
 
    @staticmethod
    def set_optimizer_params(options, optimizer=None):
        """
            Sets optimizer-specific parameters to the options object
        """
        if not optimizer == None:
            options['optimizer'] = optimizer
        
        if options['optimizer'] == 'sgd':
            
            # these work pretty well for text-only (43% val accuracy)
            options['sgd_learning_rate'] = 0.1     # Yang
            options['sgd_momentum'] = 0.9          # Yang (and Keras default)
            options['sgd_decay_rate'] = 0.0        # Yang (no decay)
            options['sgd_grad_clip'] = 0.1         # Yang

            # Keras: lr = self.lr * (1. / (1. + self.decay * self.iterations))
#             options['sgd_learning_rate'] = 0.1         # Yang
#             options['sgd_grad_clip'] = 0.1         # Yang
            
            # Unused parameters
            # options['sgd_decay_rate'] = 0.999    # never used in Yang's code
            # options['sgd_word_embed_lr'] = 80    # never used in Yang's code
            # options['sgd_gamma'] = 1             # Yang's code used gamma=1, so fixed learning rate
            # options['sgd_smooth'] = 1e-8         # never used in Yang's code
            # options['sgd_step'] = 10             # TBD whether we need this
            # options['sgd_step_start'] = 100      # TBD whether we need this
        elif options['optimizer'] == 'adam':
            options['adam_learning_rate'] = 0.1
            options['adam_beta_1'] = 0.9                # matches Keras default
            options['adam_beta_2'] = 0.999              # matches Keras default
        elif options['optimizer'] == 'rmsprop':
            options['rmsprop_learning_rate'] = 0.1      # Keras recommends not changing other params
        else:
            raise TypeError('Invalid optimizer specified.  Only \'sgd\' is currently supported.')
            
        return options

    @staticmethod
    def get_optimizer(options):
        """
            Returns a Keras optimizer instance of the configured type
        """
        if options['optimizer'] == 'sgd':
            optimizer = SGD(lr=options['sgd_learning_rate'],
                                             momentum=options['sgd_momentum'],
                                             decay=options['sgd_decay_rate'],
                                             clipnorm=options['sgd_grad_clip']
                                            )
        elif options['optimizer'] == 'adam':
            optimizer = Adam(lr=options['adam_learning_rate'],
                                              beta_1=options['adam_beta_1'],
                                              beta_2=options['adam_beta_2'])
        elif options['optimizer'] == 'rmsprop':
            optimizer = RMSprop(lr=options['rmsprop_learning_rate'])
        else:
            raise TypeError('Invalid optimizer specified.')
            
        return optimizer

