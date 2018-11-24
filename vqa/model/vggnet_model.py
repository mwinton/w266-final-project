# standard modules
from keras.applications.vgg16 import VGG16
import keras.activations
import keras.backend as kbe
from keras.callbacks import EarlyStopping
import keras.layers
from keras.layers import Activation, Add, Concatenate, Conv1D, Dense, Dropout, Embedding
from keras.layers import Input, GlobalMaxPooling1D, Lambda, Multiply, RepeatVector, Reshape
from keras.models import Model
from keras.regularizers import l2
import mlflow
import mlflow.keras
from pprint import pprint

# our own imports
from . options import ModelOptions

class VGGNetModel(object):
    
    def __init__ (self, options):
        ''' Initialize Model '''
        if options['verbose']:
            print('Initializing Image-only VGGNet model...')
        self.options = options
    
    def build_graph (self, options):
        ''' Build Keras graph '''
        verbose = options['verbose']
        print('Building graph...')

        batch_size = self.options['batch_size']
        n_attention_input = self.options['n_attention_input']
        
        # Instantiate a regularizer if weight-decay was specified
        self.regularizer = None
        if self.options.get('regularizer', False) == True:
            self.regularizer = l2(options['weight_decay'])
            if verbose: print('Using L2 regularizer with weight_decay={}...'.format(options['weight_decay']))
        else:
            print('No regularization applied')

        #
        # begin image pipeline
        # diagram: https://docs.google.com/drawings/d/1ZWRPmy4e2ACvqOsk4ttAEaWZfUX_qiQEb0DE05e8dXs/edit
        #
        
        image_input_dim = self.options['image_input_dim']
        image_input_depth = self.options['image_depth']

        # TODO: determine these dynamically from the VGG16 output
        n_image_regions = self.options['n_image_regions']
        n_image_embed = self.options['n_image_embed']

        # loading embeddings directly, so we can start with this layer
        layer_image_input = layer_reshaped_vgg16  = Input(batch_shape=(None,n_image_regions,n_image_embed),name="reshaped_vgg16")
        if verbose: print('layer_reshaped_vgg16 output shape:', layer_reshaped_vgg16.shape)
        
        # Single dense layer to transform dimensions to match sentence dims
        # in:  [batch_size, n_image_regions, image_output_depth]
        # out: [batch_size, n_image_regions, n_attention_input]
        layer_v_i = Dense(units=n_attention_input,
                          activation='tanh',
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=self.regularizer,
                          name='v_i'
                         )(layer_reshaped_vgg16)
        if verbose: print('layer_v_i output shape:', layer_v_i.shape)
                
                
        # NOTE: this max pooling layer is NOT used in the main model.  It's used here to
        # reduce the n_regions dimension to enable classification in this debugging model.
        # in:  [batch_size, n_image_regions, n_attention_input]
        # out: [batch_size, n_attention_input]
        layer_max_pooled_i = GlobalMaxPooling1D(name='max_pooled_i')(layer_v_i)
        if verbose: print('layer_max_pooled_i output shape:', layer_max_pooled_i.shape)

        # final classification
        # in:  [batch_size, n_attention_input]
        # out: [batch_size, n_answer_classes]
        n_answer_classes = self.options['n_answer_classes']
        layer_prob_answer = Dense(units=n_answer_classes,
                                  activation='softmax',
                                  use_bias=True,
                                  kernel_initializer='random_uniform',
                                  bias_initializer='zeros',
                                  kernel_regularizer=self.regularizer,
                                  name='prob_answer'
                                 )(layer_max_pooled_i)
        if verbose: print('layer_prob_answer output shape:', layer_prob_answer.shape)
        
        # do argmax to make predictions (or look for canned classifier)
        
        # assemble all these layers into model
        self.model = Model(inputs=[layer_reshaped_vgg16], outputs=layer_prob_answer)

        optimizer = ModelOptions.get_optimizer(options)
        print('Compiling model with {} optimizer...'.format(self.options['optimizer']))
        
        # compile model so that it's ready to train
        self.model.compile (optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        ''' wrapper around keras.Model.summary()'''
        self.model.summary()
