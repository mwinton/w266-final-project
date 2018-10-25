from collections import OrderedDict

class ModelOptions(object):

    def __init__(self):
        ''' init function.  This class stores hyperparameters for the model. '''
        
        self.options = OrderedDict()
        
        # File system paths
        self.options['images_train_root_path'] = 'images/mscoco/train2014'
        self.options['images_val_root_path'] = 'images/mscoco/val2014'

        self.options['questions_train_path'] = 'questions/OpenEnded_mscoco_train2014_questions.json'
        self.options['questions_val_path'] = 'questions/OpenEnded_mscoco_val2014_questions.json'

        self.options['question_types_path'] = 'questiontypes/mscoco_question_types.txt'

        self.options['annotations_train_path'] = 'annotations/mscoco_train2014_annotations.json'
        self.options['annotations_val_path'] = 'annotations/mscoco_val2014_annotations.json'

        self.options['image_embed_path'] = ''
        self.options['glove_path'] = ''

        # Image model parameters
        self.options['n_image_embed'] = 512  # VGGNet
        self.options['n_regions'] = 196    # 14x14 regions

        # Text model parameters
        self.options['n_vocab'] = 13746   # TODO: calculate this ourselves
        self.options['max_sentence_len'] = 20  # TODO: calculate this ourselves
        self.options['n_sent_embed'] = 500
        self.options['sent_init_type'] = 'uniform'  # TODO: experiment with GloVe
        self.options['sent_init_range'] = 0.01


        self.options['n_filters_unigram'] = 256
        self.options['n_filters_bigram'] = 512
        self.options['n_filters_trigram'] = 512

        # Attention layer parameters
        self.options['n_attention_input'] = self.options['num_channels_unigram'] \
                                       + self.options['num_channels_bigram'] \
                                       + self.options['num_channels_trigram']
        self.options['n_attention_layers'] = 2
        self.options['attention_merge_type'] = 'addition'
        self.options['attention_dropout_ratio'] = 0.5

        # Classification layer parameters
        self.options['n_answer_classes'] = 1000  # TODO: experiment with this (1000 words ~ 82% coverage)

        # Training parameters
        self.options['batch_size'] = 100
        self.options['max_epochs'] = 50

        # SGD training parameters
        self.options['optimizer'] = 'sgd'
        self.options['learning_rate'] = 0.1
        self.options['word_embed_lr'] = 80
        self.options['momentum'] = 0.9
        self.options['gamma'] = 1
        self.options['step'] = 10
        self.options['step_start'] = 100
        self.options['weight_decay'] = 0.0005
        self.options['decay_rate'] = 0.999
        self.options['drop_ratio'] = 0.5
        self.options['smooth'] = 1e-8
        self.options['grad_clip'] = 0.1

        # MLFlow logging parameters
        self.options['mlflow_tracking_uri'] = 'http://35.236.106.47:5000'  # Mike's GCE instance
        self.options['verbose'] = False
        self.options['log_interval'] = 500
        self.options['display_interval'] = 1000    
    
    def get_options(self):
        ''' return ordered dict containing all model options '''
        return self.options
    
