# standard modules
from keras.applications.vgg16 import VGG16
import keras.activations
import keras.backend as kbe
from keras.callbacks import EarlyStopping
import keras.layers
from keras.layers import Activation, Add, Concatenate, Conv1D, Dense, Dropout, Embedding, Softmax
from keras.layers import Input, GlobalMaxPooling1D, Lambda, Multiply, RepeatVector, Reshape
from keras.layers import BatchNormalization 
from keras.models import Model
from keras.regularizers import l2
import mlflow
import mlflow.keras
from pprint import pprint

# our own imports
from . options import ModelOptions

class StackedAttentionNetwork(object):
    
    def __init__ (self, options):
        ''' Initialize SAN object '''
        if options['verbose']:
            print('Initializing SAN...')
        self.options = options
    
    def _sum_axis_1(self, x):
        ''' need to wrap backend ops in function and call from Lambda layer'''
        return kbe.sum(x, axis=1)
        
    def _repeat_elements(self, x, n, axis=-1):
        ''' need to wrap backend ops in function and call from Lambda layer'''
        return kbe.repeat_elements(x, n, axis=axis)
        
    def _build_attention_subgraph (self, options, idx, layer_v_i, layer_v_q):

        verbose = options['verbose']
        print('Building attention subgraph...')

        # Single dense layer to reduce image dimensions for attention
        # in:  [batch_size, n_image_regions, n_attention_input]
        # out: [batch_size, n_image_regions, n_attention_features]
        n_attention_features = options['n_attention_features']
        layer_attn_image = Dense(units=n_attention_features,
                                   activation='tanh',
                                   use_bias=True,
                                   kernel_initializer='random_uniform',
                                   bias_initializer='zeros',
                                   kernel_regularizer=self.regularizer,
                                   name='attention_image_%d' % (idx)
                                  )(layer_v_i)
        if verbose: print('attention_image_%d' % (idx), layer_attn_image.shape)

        # Adding a batch norm step before image and sentence vectors are added
        # This comment refers to subsequent steps for batchNorm in the model as well,
        # where we do a normalization before adding image and sentence vectors.
        # This was significant in getting accuracy improvements.
        # One reason for this might be that standardization of the image and question
        # vectors keep them in equal footing in terms of their respective magnitudes

        layer_attn_image  = BatchNormalization(name='batch_norm_image_%d' % (idx))(layer_attn_image)

        
        # Single dense layer to reduce sentence dimensions for attention
        # in:  [batch_size, n_attention_input]
        # out: [batch_size, n_attention_features]
        n_attention_features = self.options['n_attention_features']
        layer_attn_sent = Dense(units=n_attention_features,
                                activation='tanh',
                                use_bias=True,
                                kernel_initializer='random_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=self.regularizer,
                                name='attention_sent_%d' % (idx)
                               )(layer_v_q)
        if verbose: print('attention_sent_%d' % (idx), layer_attn_sent.shape)

        # Adding a batch norm before image and sentence vectors are added
        layer_attn_sent = BatchNormalization(name='batch_norm_sent_%d' % (idx))(layer_attn_sent)
        
        # Need to expand and repeat the sentence vector to be added to each image region
        # in:   [batch_size, n_attention_features]
        # out:  [batch_size, n_image_regions, n_attention_features]
        n_image_regions = options['n_image_regions']
        layer_attn_sent = RepeatVector(n_image_regions,
                                       name='expanded_attn_sent_%d' % (idx))(layer_attn_sent)
        if verbose: print('expanded_attn_sent_%d' % (idx), layer_attn_sent.shape)

        # combine the image and sentence tensors
        # in (image):     [batch_size, n_image_regions, n_attention_features]
        # in (sentence):  [batch_size, n_attention_features]
        # out:            [batch_size, n_image_regions, n_attention_features]
        attention_merge_type = self.options['attention_merge_type']
        if attention_merge_type == 'addition':  # Yang's paper did simple matrix + vector addition
            layer_h_a = Add(name='h_a_%d' % (idx))([layer_attn_sent, layer_attn_image])
        else:
            # TODO: add option to combine some other way
            pass
        if verbose: print('h_a_%d' % (idx), layer_h_a.shape)
        
        # Single dense layer to reduce axis=2 to 1 dimension for softmax (one per image region)
        # in:   [batch_size, n_image_regions, n_attention_features]
        # out:  [batch_size, n_image_regions, 1]
        layer_pre_softmax = Dense(units=1,
                                  activation='tanh',
                                  use_bias=True,
                                  kernel_initializer='random_uniform',
                                  bias_initializer='zeros',
                                  kernel_regularizer=self.regularizer,
                                  name='pre_softmax_%d' % (idx)
                                 )(layer_h_a)
        if verbose: print('layer_pre_softmax_%d' % (idx), layer_pre_softmax.shape)
        
        # Calculate softmax
        # in:   [batch_size, n_image_regions, 1]
        # out:  [batch_size, n_image_regions, 1]
        layer_attn_prob_dist = Softmax(axis=1, name='layer_prob_attn_%d' % (idx))(layer_pre_softmax)
        if verbose: print('layer_attn_prob_dist_%d' % (idx), layer_attn_prob_dist.shape)
        
        # Need to expand and repeat the attention vector to be multiplied by each image region
        # in:  [batch_size, n_image_regions, 1]
        # out:  [batch_size, n_image_regions, n_attention_input]
        n_attention_input = options['n_attention_input']
        layer_prob_expanded = Lambda(self._repeat_elements, 
                                     name='layer_prob_expanded_%d' % (idx),
                                     arguments={'n':n_attention_input})(layer_attn_prob_dist)
        if verbose: print('layer_prob_expanded_%d' % (idx), layer_prob_expanded.shape)

        # Refined query vector
        # in:   [batch_size, n_image_regions, n_attention_input]
        # out:  [batch_size, n_attention_input]
        layer_v_tilde = Multiply()([layer_v_i, layer_prob_expanded])
        layer_v_tilde = Lambda(self._sum_axis_1, name='v_tilde_%d' % (idx))(layer_v_tilde)
        if verbose: print('v_tilde_%d' % (idx), layer_v_tilde.shape)

        # Adding a batch norm befsre image and sentence vectors are added
        layer_v_tilde = BatchNormalization(name='batch_norm_v_tilde_%d' % (idx)) (layer_v_tilde)
        layer_v_q     = BatchNormalization(name='batch_norm_v_q_%d' % (idx))(layer_v_q)

        layer_v_q_refined = Add(name='v_q_refined_%d' % (idx))([layer_v_tilde, layer_v_q])
        if verbose: print('v_q_refined_%d' % (idx), layer_v_q_refined.shape)
        
        return layer_v_q_refined

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
        
        image_input_dim = self.options['vggnet_input_dim']
        image_input_depth = self.options['image_depth']

        # TODO: determine these dynamically from the VGG16 output
        n_image_regions = self.options['n_image_regions']
        n_image_embed = self.options['n_image_embed']

        if options['start_with_image_embed']:
            # if loading embeddings directly, we can start with this layer
            layer_image_input = layer_reshaped_vgg16  = Input(batch_shape=(None,n_image_regions,n_image_embed),name="reshaped_vgg16")
            
            if verbose: print('layer_reshaped_vgg16 output shape:', layer_reshaped_vgg16.shape)
        else:
            # image input as [batch_size, image_input_dim, image_input_dim, image_input_depth] of floats
            layer_image_input = Input(batch_shape=(None, image_input_dim, image_input_dim, image_input_depth),
                                      dtype='float32',
                                      sparse=False,
                                      name='image_input'
                                     )
            if verbose: print('layer_image_input shape:', layer_image_input._keras_shape)
        
            # Runs VGGNet16 model and extracts last pooling layeer
            # in:  [batch_size, image_input_dim, image_input_dim, image_input_depth]
            # out: [batch_size, image_output_dim, image_output_dim, image_output_depth]
            image_model_initializer = self.options.get('image_init_type', None)
            model_vgg16 = VGG16(include_top=False,
                                weights=image_model_initializer,  # None = random initialization
                                input_tensor=layer_image_input,
                                input_shape=(image_input_dim, image_input_dim, image_input_depth),
                                pooling=None  # output is 4D tensor from last convolutional layer
                                # TODO: check the order of returned tensor dimensions
                               )
            if verbose: print('model_vgg16 output shape:', model_vgg16.output_shape)
        

            # Reshaped image output to flatten the image region vectors
            # in:  [batch_size, image_output_dim, image_output_dim, n_image_embed]
            # out: [batch_size, n_image_regions, n_image_embed]
            layer_reshaped_vgg16 = Reshape((n_image_regions, n_image_embed),  # excludes batch size
                                           name='reshaped_vgg16'
                                          )(model_vgg16.output)  # model.output gives a tensor
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
        sent_embed_initializer = self.options.get('sent_init_type', 'random') 
        sent_embed_dim = self.options['n_sent_embed']
        if sent_embed_initializer == 'random':
            layer_x = Embedding(input_dim=V, 
                                output_dim=sent_embed_dim,
                                input_length=max_t,
                                embeddings_initializer=sent_embed_initializer,
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
        # begin attention layers
        # diagram: https://docs.google.com/drawings/d/1EDpuHGZHA_BjR0kE23B6UsjccvHr0z-uAB6F-CKLop0/edit
        #
        
        # build multi-layer attention stack
        # image in:     [batch_size, n_image_regions, n_attention_input]
        # sentence in:  [batch_size, n_attention_input]
        # out:          [batch_size, n_attention_input]
        n_attention_layers = options['n_attention_layers']          
        for idx in range(n_attention_layers):
            layer_v_q = self._build_attention_subgraph(options, idx, layer_v_i, layer_v_q)
               
        # apply dropout after final attention layer
        # In Keras, dropout is automatically disabled in test mode 
        attention_dropout_ratio = self.options['attention_dropout_ratio']
        layer_dropout_v_q = Dropout(rate=attention_dropout_ratio, name='dropout_v_q')(layer_v_q)
        if verbose: print('layer_dropout_v_q output shape:', layer_dropout_v_q.shape)
        
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
                                 )(layer_dropout_v_q)
        if verbose: print('layer_prob_answer output shape:', layer_prob_answer.shape)
        
        # do argmax to make predictions (or look for canned classifier)
        
        # assemble all these layers into model
        self.model = Model(inputs=[layer_image_input, layer_sent_input], outputs=layer_prob_answer)

        optimizer = ModelOptions.get_optimizer(options)
        print('Compiling model with {} optimizer...'.format(self.options['optimizer']))
        
        # compile model so that it's ready to train
        self.model.compile (optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # build attention layer model with one output for each attention layer, and connect to
        # the main model in order to extract attention probabilities output
        attention_layers = []
        for idx in range(n_attention_layers):
            attention_layers.append(self.model.get_layer('layer_prob_attn_%d' % (idx)).output)
        self.attention_layer_model = Model(inputs=self.model.input, outputs=attention_layers)

        # build attention layer model with one output for each attention layer, and connect to
        # the main model in order to extract attention probabilities output
        attention_layers = []
        for idx in range(n_attention_layers):
            attention_layers.append(self.model.get_layer('layer_prob_attn_%d' % (idx)).output)
        self.attention_layer_model = Model(inputs=self.model.input, outputs=attention_layers)
    
    def summary(self):
        ''' wrapper around keras.Model.summary()'''
        self.model.summary()
