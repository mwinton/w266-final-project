from keras.applications.vgg16 import VGG16
import keras.activations
import keras.backend as kbe
from keras.callbacks import EarlyStopping
import keras.layers
from keras.layers import Concatenate, Conv1D, Dense, Dropout, Embedding
from keras.layers import Input, GlobalMaxPooling1D, Reshape, Softmax
from keras.layers import BatchNormalization 
from keras.models import Model
from keras.regularizers import l2
import mlflow
import mlflow.keras
import pickle
from pprint import pprint

from . options import ModelOptions

class NoAttentionNetwork(object):
    
    def __init__ (self, options):
        ''' Initialize SAN object '''
        if options['verbose']:
            print('Initializing SAN...')
        self.options = options
    
    def build_graph (self, options):
        ''' Build Keras graph '''
        verbose = options['verbose']
        print('Building graph...')

        # check options for alternate activation type
        activation_type = options.get('activation_type', 'tanh')

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
        
#         image_input_dim = self.options['vggnet_input_dim']
#         image_input_depth = self.options['image_depth']

        n_image_regions = self.options['n_image_regions']
        n_image_embed = self.options['n_image_embed']

        # pre-load image loading embeddings
        # in:  [batch_size, n_image_regions, image_output_depth]
        layer_reshaped_vgg16 = Input(batch_shape=(None, n_image_regions, n_image_embed), name="reshaped_vgg16")
        if verbose: print('layer_reshaped_vgg16 output shape:', layer_reshaped_vgg16.shape)

        #
        # begin sentence pipeline
        # diagram: https://docs.google.com/drawings/d/1PJKOcQA73sUvH-w3UlLOhFH0iHOaPJ9-IRb7S75yw0M/edit
        #
        
        # these are both set when the dataset is prepared
        max_t = self.options['max_sentence_len']
        V = self.options['n_vocab']
        if verbose: print('input vocab size:', V)

        # sentence input receives sequences of [batch_size, max_time] integers between 1 and V
        layer_sent_input = Input(batch_shape=(None, max_t),
                                 dtype='int32',
                                 sparse=False,
                                 name='sentence_input'
                                )
        if verbose: print('layer_sent_input shape:', layer_sent_input._keras_shape)

        # This embedding layer will encode the input sequence
        # in:  [batch_size, max_t]
        # out: [batch_size, max_t, n_text_embed]
        # default to randomly initialized embeddings (rather than GloVe)
        sent_embed_initializer = self.options['sent_init_type']
        sent_embed_dim = self.options['n_sent_embed']
        if sent_embed_initializer == 'random':
            layer_x = Embedding(input_dim=V, 
                                output_dim=sent_embed_dim,
                                input_length=max_t,
                                embeddings_initializer='uniform',
                                mask_zero=False,  # CNN layers don't seem to be able to deal with True
                                name='sentence_embedding'
                               )(layer_sent_input)
        elif sent_embed_initializer == 'glove':
            trainable = options['sent_embed_trainable']
            glove_matrix = pickle.load(open(options['glove_matrix_path'], 'rb'))
            if sent_embed_dim != glove_matrix.shape[1]:
                # if options don't match the matrix shape, override with actual (but logs may be wrong)
                sent_embed_dim = self.options['n_sent_embed'] = glove_matrix.shape[1]
            print('Loaded GloVe embedding matrix')
            layer_x = Embedding(input_dim=V, 
                                output_dim=sent_embed_dim, 
                                input_length=max_t,
                                weights=[glove_matrix],
                                trainable=trainable,
                                name='sentence_embedding'
                               )(layer_sent_input)
        
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
                                    kernel_regularizer=self.regularizer,
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
                                   kernel_regularizer=self.regularizer,
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
                              kernel_regularizer=self.regularizer,
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
        
        #
        # replace attention layers in original model with simple concatenation of features
        #
        
        # Single dense layer to reduce sentence dimensions for attention
        # in:  [batch_size, n_attention_input]
        # out: [batch_size, n_attention_features]
        n_attention_features = self.options['n_attention_features']
        layer_v_q = Dense(units=n_attention_features,
                                activation=activation_type,
                                use_bias=True,
                                kernel_initializer='random_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=self.regularizer,
                                name='v_q_reduced'
                               )(layer_v_q)
        if verbose: print('layer_v_q output shape', layer_v_q.shape)

        # need to expand the rank of the tensor to concatenate with image vector
        # in:  [batch_size, n_attention_features]
        # out: [batch_size, 1, n_attention_features]
        layer_v_q = Reshape((1, n_attention_features))(layer_v_q)

        # apply batch normalization to get image and sentence features on same scale; then concatenate
        layer_v_i_norm  = BatchNormalization(name='batch_norm_image')(layer_reshaped_vgg16)
        layer_v_q_norm  = BatchNormalization(name='batch_norm_sent')(layer_v_q)
        layer_all_feats = Concatenate(axis=1, name='all_feats')([layer_v_i_norm, layer_v_q_norm])
        if verbose: print('layer_all_feats output shape:', layer_all_feats.shape)
        
        # apply dropout after final concatenation layer
        # (repurposing `attention_dropout_ratio` option for this experimeent)
        # In Keras, dropout is automatically disabled in test mode 
        attention_dropout_ratio = self.options['attention_dropout_ratio']
        layer_dropout = Dropout(rate=attention_dropout_ratio, name='dropout')(layer_all_feats)
        if verbose: print('layer_dropout output shape:', layer_dropout.shape)
         
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
                                 )(layer_dropout)
        if verbose: print('layer_prob_answer output shape:', layer_prob_answer.shape)
        
        # assemble all these layers into model
        self.model = Model(inputs=[layer_reshaped_vgg16, layer_sent_input], outputs=layer_prob_answer)

        optimizer = ModelOptions.get_optimizer(options)
        print('Compiling model with {} optimizer...'.format(self.options['optimizer']))
        
        # compile model so that it's ready to train
        self.model.compile (optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        ''' wrapper around keras.Model.summary()'''
        self.model.summary()
