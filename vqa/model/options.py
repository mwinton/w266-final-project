import getpass
import datetime
import os

from collections import OrderedDict
from keras.optimizers import SGD

from .. dataset.types import DatasetType

class ModelOptions(object):

    def __init__(self):
        ''' init function.  This class stores hyperparameters for the model. '''
        
        self.options = OrderedDict()
        
        # File system paths
        user_name = getpass.getuser()
        data_root = "/home/"+user_name+"/vqa_data/"
        self.options['data_root'] = data_root
        self.options['tb_logs_root'] = "/home/" + user_name + "/logs/"

        self.options['user_name'] = user_name

        self.options['images_train_root_path'] = data_root+'images/mscoco/train2014/'
        self.options['images_val_root_path']   = data_root+'images/mscoco/val2014/'
        self.options['images_test_root_path']   = data_root+'images/mscoco/test2015/'

        self.options['questions_train_path'] = data_root+'questions/OpenEnded_mscoco_train2014_questions.json'
        self.options['questions_val_path']   = data_root+'questions/OpenEnded_mscoco_val2014_questions.json'
        self.options['questions_test_path']  = data_root+'questions/OpenEnded_mscoco_test2015_questions.json'

        self.options['question_types_path']  = data_root+'questiontypes/mscoco_question_types.txt'

        self.options['annotations_train_path'] = data_root+'annotations/mscoco_train2014_annotations.json'
        self.options['annotations_val_path']   = data_root+'annotations/mscoco_val2014_annotations.json'

        image_embed_root = data_root+'images/mscoco/embeddings/vgg16/'
        self.options['images_embed_train_path'] = image_embed_root+'train.hdf5'
        self.options['images_embed_val_path']   = image_embed_root+'val.hdf5'
        self.options['images_embed_test_path'] = image_embed_root+'test.hdf5'
        
#         self.options['glove_path'] = ''

        ## files created during train/val/test phases
        self.options['local_data_path']  =  "../data/preprocessed/"
        self.options['saved_models_path'] = '../saved_models/json/'
        self.options['weights_dir_path'] =  "../saved_models/weights/"
        self.options['results_dir_path'] =  "../results/"

        ## create directories if they don't exist
        os.makedirs(self.options['local_data_path'],   exist_ok=True)
        os.makedirs(self.options['weights_dir_path'],  exist_ok=True)
        os.makedirs(self.options['results_dir_path'],  exist_ok=True)
        os.makedirs(self.options['saved_models_path'], exist_ok=True) 

        local_data_path = self.options['local_data_path']

        self.options['tokenizer_path']     = local_data_path+'tokenizer.p'
        self.options['train_dataset_path'] = local_data_path+'train_dataset.p'
        self.options['val_dataset_path']   = local_data_path+"validate_dataset.p"
        self.options['test_dataset_path']  = local_data_path+"test_dataset.p"
        self.options['eval_dataset_path']  = local_data_path+"eval_dataset.p"

        # to be polulated later once all options are parsed
        self.options['weights_path']  = ""
        self.options['results_path'] = ""
        self.options['losses_path']  = ""

        # Model selection parameter
        self.options['model_name'] = "baseline"  # default is the first/baseline model

        # Type of action to be performed
        self.options['action_type'] = "train"

        # Experiment to be performed
        self.options['experiment_id'] = 0                  # default is no experiment
        self.options['experiment_name'] = 'VQA_default'    # name to display in MLFlow
        
        # Image model parameters
        self.options['n_image_embed'] = 512     # VGGNet
        self.options['n_image_regions'] = 196   # 14x14 regions
        self.options['mscoco_dim'] = 256        # x, y dimension of photos
        self.options['vggnet_input_dim'] = 448  # expected x, y dim of VGGNet
        self.options['image_depth'] = 3         # 3 color channels (RGB)
        self.options['start_with_image_embed'] = True

        # Text model parameters
        # self.options['n_vocab'] = 18364           # Keras tokenizer: 18364 (Q+A) or 13681 (Q-only); Yang: 13746
        # self.options['max_sentence_len'] = 22     # actual max is 22, so don't override it   
        self.options['n_sent_embed'] = 500          # TODO: change this when we use GloVe
        self.options['sent_init_type'] = 'uniform'  # TODO: experiment with GloVe
        self.options['sent_init_range'] = 0.01
        self.options['keep_single_answer'] = True


        self.options['n_filters_unigram'] = 256
        self.options['n_filters_bigram'] = 512
        self.options['n_filters_trigram'] = 512

        # Attention layer parameters
        self.options['n_attention_input'] = self.options['n_filters_unigram'] \
                                          + self.options['n_filters_bigram'] \
                                          + self.options['n_filters_trigram']
        self.options['n_attention_features'] = 512
        self.options['n_attention_layers'] = 2
        self.options['attention_merge_type'] = 'addition'
        self.options['attention_dropout_ratio'] = 0.5

        # Classification layer parameters
        self.options['n_answer_classes'] = 1001  # 1000 real classes + <unk>  

        # Training parameters
        self.options['batch_size'] = 100
        self.options['max_epochs'] = 50
        self.options['early_stop_patience'] = 5
        self.options['max_train_size'] = None # interpreted as full size of training set unless overridden
        self.options['max_val_size'] = None # interpreted as full validation set size unless overridden
        self.options['extended']  = False # use train+val set for training
        # When changing the optimize, also update set_optimizer_params()
        self.options['optimizer'] = 'sgd'       

        # TODO: implement custom loss function
        # self.options['loss_function'] = 'neg_mean_log_prob_y'  # TODO: try cross-entropy -p*log(q)

        # Regularization / weight decay parameters
        self.options['weight_decay'] = 0.0005   # in Keras we have to apply layer-by-layer

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
        return selector.get(datasetType)

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

        weights_dir_path = options['weights_dir_path']
        results_dir_path = options['results_dir_path']

        if (extended):
            suffix = "_ext"
        else:
            suffix = ""
            
        # timestamp string to use in appropriate filenames
        # not timestamping weight files this point since they need to be re-read
        date_str = datetime.datetime.now().isoformat()
        
        if (action == "train"):
            # timestamp the weights; later we create a symlink to the most recent set (for prediction)
            options["weights_path"] = weights_dir_path + 'model_weights_' + \
                model_name + suffix + '_' + date_str + '.{epoch:02d}.hdf5'  # must use named `epoch` placeholder
            
            # timestamp the losses_path for logging purposes
            options['losses_path'] = results_dir_path+'losses_{}{}_{}.hdf5'.format(model_name, suffix, date_str)
            
        elif (action == "val" ):
            # TODO: update to handle timestamped files if necessary
            options["weights_path"] = weights_dir_path + 'model_weights_{}'.format(model_name)

        elif (action == "test"):
            # TODO: update to handle timestamped files if necessary
            options['weights_path'] = weights_dir_path + 'model_weights_{}{}'.format(model_name, suffix)
            options['results_path'] = results_dir_path + 'test2015_results_{}{}.json'.format(model_name, suffix)
        
        else:
            # action type is eval
            # TODO: update to handle timestamped files if necessary
            options['weights_path'] = weights_dir_path + 'model_weights_{}{}'.format(model_name, suffix)
            options['results_path'] = results_dir_path + 'val2014_results_{}{}.json'.format(model_name, suffix)

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
        else:
            raise TypeError('Invalid optimizer specified.  Only \'sgd\' is currently supported.')
            
        return options
