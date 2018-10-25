# standard modules
from keras.applications.vgg16 import VGG16
import keras.layers
from keras.layers import Conv1D, Dense, Embedding, Input, GlobalMaxPooling1D, Reshape
from keras.models import Model
import mlflow
import mlflow.keras
from pprint import pprint

# our own imports
from vqa_options import ModelOptions

class StackedAttentionNetwork(object):
    
    def __init__ (self, options):
        ''' Initialize SAN object '''
        if options['verbose']:
            print('Initializing SAN...')
        self.options = options
    
    def build_question_subgraph (self, options):
        pass
    
    def build_image_subgraph (self, options):
        pass
    
    def build_attention_subgraph (self, options):
        pass
    
    def build_graph (self, options):
        ''' Build Keras graph '''
        if options['verbose']:
            print('Building graph...')

        batch_size = self.options['batch_size']
        n_attention_input = self.options['n_attention_input']
        
        #
        # begin image pipeline
        # diagram: https://docs.google.com/drawings/d/1ZWRPmy4e2ACvqOsk4ttAEaWZfUX_qiQEb0DE05e8dXs/edit
        #
        
        # TODO: make sure images are rescaled from 256x256 -> 448x448 during preprocessing
        
        image_input_dim = self.options['vggnet_input_dim']
        image_input_depth = self.options['image_depth']

        # image input as [batch_size, image_input_dim, image_input_dim, image_input_depth] of floats
        layer_image_input = Input(batch_shape=(None, image_input_dim, image_input_dim, image_input_depth),
                                  dtype='float32',
                                  sparse=False,
                                  name='image_input'
                                 )
        
        # Runs VGGNet16 model and extracts last pooling layeer
        # in:  [batch_size, image_input_dim, image_input_dim, image_input_depth]
        # out: [batch_size, image_output_dim, image_output_dim, n_image_regions]
        image_model_initializer = self.options.get('image_init_type', None)
        layer_vgg16 = VGG16(include_top=False,
                            weights=image_model_initializer,  # None = random initialization
                            input_tensor=layer_image_input,
                            input_shape=(image_input_dim, image_input_dim, image_input_depth),
                            pooling=None  # output is 4D tensor from last convolutional layer
                            # TODO: check the order of returned tensor dimensions
                           )
        
        n_image_regions = self.options['n_regions']

        # Reshaped image output to flatten the image region vectors
        # in:  [batch_size, image_output_dim, image_output_dim, image_output_depth]
        # out: [batch_size, n_image_regions, image_output_depth]
        layer_vgg16 = Reshape((batch_size, -1, n_image_regions))(layer_vgg16)
        
        # Single-layer perceptron to transform dimensions to match sentence dims
        # in:  [batch_size, n_image_regions, image_output_depth]
        # out: [batch_size, n_image_regions, n_attention_input]
        layer_v_i = Dense(units=n_attention_input,
                          activation='tanh',
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          name='v_i'
                         )(layer_vgg16)
        
        
        #
        # begin sentence pipeline
        # diagram: https://docs.google.com/drawings/d/1PJKOcQA73sUvH-w3UlLOhFH0iHOaPJ9-IRb7S75yw0M/edit
        #
        
        max_t = self.options['max_sentence_len']
        V = self.options['n_vocab']

        # sentence input receives sequences of [batch_size, max_time] integers between 1 and V
        layer_sent_input = Input(batch_shape=(None, max_t),
                                 dtype='int32',
                                 sparse=True,
                                 name='sentence_input'
                                )

        # This embedding layer will encode the input sequence
        # in:  [batch_size, max_t]
        # out: [batch_size, max_t, n_text_embed]
        sent_embed_initializer = self.options.get('sent_init_type', 'uniform')
        sent_embed_dim = self.options['n_sent_embed']
        if sent_embed_initializer == 'uniform':
            layer_x = Embedding(input_dim=V, 
                                output_dim=sent_embed_dim,
                                input_length=max_t,
                                embedding_initializer=sent_embed_initializer,
                                mask_zero=True,
                                name='sentence_embedding'
                               )(layer_sent_input)
        # TODO: implement GloVe option
        elif sent_embed_initializer == 'glove':
            pass
    
        # Unigram CNN layer
        # in:  [batch_size, max_t, n_text_embed]
        # out: [batch_size, n_unigrams, n_filters_unigram]
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
        
        # Unigram max pooling
        # in:  [batch_size, n_unigrams, n_filters_unigram]
        # out: [batch_size, n_filters_unigram]
        layer_pooled_unigram = GlobalMaxPooling1D(name='unigram_max_pool')(layer_conv_unigram)

        # Bigram CNN layer
        # in:  [batch_size, max_t, n_text_embed]
        # out: [batch_size, n_bigrams, n_filters_bigram]
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
        
        # Bigram max pooling
        # in:  [batch_size, n_bigrams, n_filters_bigram]
        # out: [batch_size, n_filters_bigram]
        layer_pooled_bigram = GlobalMaxPooling1D(name='bigram_max_pool')(layer_conv_bigram)

        # Trigram CNN layer
        # in:  [batch_size, max_t, n_text_embed]
        # out: [batch_size, n_trigrams, n_filters_trigram]
        n_filters_trigram = self.options['n_filters_trigram']
        layer_conv_trigram = Conv1D(filters=n_filters_trigram,
                              kernel_size=2,
                              strides=1,
                              padding='valid',  # this results in output length != input length
                              activation='tanh',
                              use_bias=True,
                              kernel_initializer='random_uniform',
                              bias_initializer='zeros',
                              name='trigram_conv'
                             )(layer_x)
        
        # Trigram max pooling
        # in:  [batch_size, n_trigrams, n_filters_trigram]
        # out: [batch_size, n_filters_trigram]
        layer_pooled_trigram = GlobalMaxPooling1D(name='trigram_max_pool')(layer_conv_trigram)

        # Concatenate the n-gram max pooled tensors into our question vector
        # in:  [batch_size, n_filters_(uni|bi|tri)gram]
        # out: [batch_size, n_attention_input]
        layer_v_q = layers.Concatenate(axis=1, name='v_q')(layer_conv_unigram, 
                                                           layer_conv_bigram,
                                                           layer_conv_trigram,
                                                           name='v_q'
                                                          )
        #
        # begin attention layers
        # diagram: https://docs.google.com/drawings/d/1EDpuHGZHA_BjR0kE23B6UsjccvHr0z-uAB6F-CKLop0/edit
        #
        
        
        # 
        # begin classification layers
        #
        
        
    def train (self, options):
        ''' Train graph '''
        if options['verbose']:
            print('Training...')
    
    def predict (self, options):
        ''' Make predictions '''
        if options['verbose']:
            print('Predicting...')