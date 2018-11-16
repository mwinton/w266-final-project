# standard modules
from keras.applications.vgg16 import VGG16
import keras.activations
import keras.backend as kbe
from keras.callbacks import EarlyStopping
import keras.layers
from keras.layers import Activation, Add, Concatenate, Conv1D, Dense, Dropout, Embedding
from keras.layers import Input, GlobalMaxPooling1D, Lambda, Multiply, RepeatVector, Reshape
from keras.models import Model
import mlflow
import mlflow.keras
from pprint import pprint

# our own imports
from . options import ModelOptions

class TextCNNModel(object):
    
    def __init__ (self, options):
        ''' Initialize Model '''
        if options['verbose']:
            print('Initializing Text-only CNN model...')
        self.options = options
    
    def build_graph (self, options):
        ''' Build Keras graph '''
        verbose = options['verbose']
        print('Building graph...')

        batch_size = self.options['batch_size']
        n_attention_input = self.options['n_attention_input']
        
        #
        # begin sentence pipeline
        # diagram: https://docs.google.com/drawings/d/1PJKOcQA73sUvH-w3UlLOhFH0iHOaPJ9-IRb7S75yw0M/edit
        #
        
        max_t = self.options['max_sentence_len']
        V = self.options['n_vocab']

        # sentence input receives sequences of [batch_size, max_time] integers between 1 and V
        layer_sent_input = Input(batch_shape=(None, max_t),
                                 dtype='int32',
                                 sparse=False,
#                                  sparse=True,  # if we start as sparse, have to convert to dense later
                                 name='sentence_input'
                                )
        if verbose: print('layer_sent_input shape:', layer_sent_input._keras_shape)

        # This embedding layer will encode the input sequence
        # in:  [batch_size, max_t]
        # out: [batch_size, max_t, n_text_embed]
        sent_embed_initializer = self.options.get('sent_init_type', 'uniform')
        sent_embed_dim = self.options['n_sent_embed']
        if sent_embed_initializer == 'uniform':
            layer_x = Embedding(input_dim=V, 
                                output_dim=sent_embed_dim,
                                input_length=max_t,
                                embeddings_initializer=sent_embed_initializer,
                                mask_zero=False,  # CNN layers don't seem to be able to deal with True
                                name='sentence_embedding'
                               )(layer_sent_input)
#                                )(kbe.to_dense(layer_sent_input))
        # TODO: implement GloVe option
        elif sent_embed_initializer == 'glove':
            pass
        if verbose: print('layer_x output shape:', layer_x.shape)
    
        # Unigram CNN layer
        # in:  [batch_size, max_t, n_text_embed]
        # out: [batch_size, max_t, n_filters_unigram]
        n_filters_unigram = self.options['n_filters_unigram']
        layer_conv_unigram = Conv1D(filters=n_filters_unigram,
                                    kernel_size=1,
                                    strides=1,
                                    padding='valid',  # this results in output length != input length
                                    activation='tanh',
                                    use_bias=True,
                                    kernel_initializer='random_uniform',
                                    bias_initializer='zeros',
                                    name='unigram_conv'
                                   )(layer_x)
        if verbose: print('layer_conv_unigram output shape:', layer_conv_unigram.shape)
        
        # Unigram max pooling
        # in:  [batch_size, n_unigrams, n_filters_unigram]
        # out: [batch_size, n_filters_unigram]
        layer_pooled_unigram = GlobalMaxPooling1D(name='unigram_max_pool')(layer_conv_unigram)
        if verbose: print('layer_pooled_unigram output shape:', layer_pooled_unigram.shape)

        # Bigram CNN layer
        # in:  [batch_size, max_t, n_text_embed]
        # out: [batch_size, max_t - 1, n_filters_bigram]
        n_filters_bigram = self.options['n_filters_bigram']
        layer_conv_bigram = Conv1D(filters=n_filters_bigram,
                                   kernel_size=2,
                                   strides=1,
                                   padding='valid',  # this results in output length != input length
                                   activation='tanh',
                                   use_bias=True,
                                   kernel_initializer='random_uniform',
                                   bias_initializer='zeros',
                                   name='bigram_conv'
                                  )(layer_x)
        if verbose: print('layer_conv_bigram output shape:', layer_conv_bigram.shape)
        
        # Bigram max pooling
        # in:  [batch_size, n_bigrams, n_filters_bigram]
        # out: [batch_size, n_filters_bigram]
        layer_pooled_bigram = GlobalMaxPooling1D(name='bigram_max_pool')(layer_conv_bigram)
        if verbose: print('layer_pooled_bigram output shape:', layer_pooled_bigram.shape)

        # Trigram CNN layer
        # in:  [batch_size, max_t, n_text_embed]
        # out: [batch_size, max_t - 2, n_filters_trigram]
        n_filters_trigram = self.options['n_filters_trigram']
        layer_conv_trigram = Conv1D(filters=n_filters_trigram,
                              kernel_size=3,
                              strides=1,
                              padding='valid',  # this results in output length != input length
                              activation='tanh',
                              use_bias=True,
                              kernel_initializer='random_uniform',
                              bias_initializer='zeros',
                              name='trigram_conv'
                             )(layer_x)
        if verbose: print('layer_conv_trigram output shape:', layer_conv_trigram.shape)
        
        # Trigram max pooling
        # in:  [batch_size, n_trigrams, n_filters_trigram]
        # out: [batch_size, n_filters_trigram]
        layer_pooled_trigram = GlobalMaxPooling1D(name='trigram_max_pool')(layer_conv_trigram)
        if verbose: print('layer_pooled_trigram output shape:', layer_pooled_trigram.shape)

        # Concatenate the n-gram max pooled tensors into our question vector
        # in:  [batch_size, n_filters_(uni|bi|tri)gram]
        # out: [batch_size, n_attention_input]
        layer_v_q = Concatenate(axis=1, name='v_q')(
            [layer_pooled_unigram, layer_pooled_bigram, layer_pooled_trigram])
        if verbose: print('layer_v_q output shape:', layer_v_q.shape)
                
        # final classification
        # in:  [batch_size, n_attention_input]
        # out: [batch_size, n_answer_classes]
        n_answer_classes = self.options['n_answer_classes']
        layer_prob_answer = Dense(units=n_answer_classes,
                                  activation='softmax',
                                  use_bias=True,
                                  kernel_initializer='random_uniform',
                                  bias_initializer='zeros',
                                  name='prob_answer'
                                 )(layer_v_q)
        if verbose: print('layer_prob_answer output shape:', layer_prob_answer.shape)
        
        # do argmax to make predictions (or look for canned classifier)
        
        # assemble all these layers into model
        self.model = Model(inputs=[layer_sent_input], outputs=layer_prob_answer)

        optimizer = ModelOptions.get_optimizer(options)
        print('Compiling model with {} optimizer...'.format(self.options['optimizer']))
        
        # compile model so that it's ready to train
        self.model.compile (optimizer=optimizer,
                            loss='categorical_crossentropy',  # can train if not using the sparse version
                            # TODO: to match Yang's paper we may need to write our own loss function
                            # see https://github.com/keras-team/keras/blob/master/keras/losses.py
                            metrics=['accuracy'])

    def summary(self):
        ''' wrapper around keras.Model.summary()'''
        self.model.summary()
