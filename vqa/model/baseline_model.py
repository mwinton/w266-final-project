# standard modules
import keras.activations
import keras.layers
from keras.layers import concatenate, Dense, Dropout, Embedding, Input, LSTM, RepeatVector, Reshape
from keras.models import Model

# our own imports
from . options import ModelOptions

class BaselineModel(object):
    
    def __init__ (self, options):
        ''' Initialize BaselineModel object '''
        if options['verbose']:
            print('Initializing Baseline Model...')
        self.options = options
    
    def build_graph (self, options):
        ''' Build simple Keras graph to exercise end-to-end code path'''

        verbose = options['verbose']
        if verbose: print('Building graph...')
        
        n_image_regions  = options['n_image_regions'] # 196 (14x14)
        n_image_embed    = options['n_image_embed']   # 512 channels VGGNet
        max_sentence_len = options['max_sentence_len']
        vocabulary_size  = options['n_vocab']
        sentence_embed_size  = options['n_sent_embed']
        n_classes = options['n_answer_classes']

        # Send image through a dense layer
        image_input = Input(batch_shape=(None, n_image_regions, n_image_embed))
        image_reshape  = Reshape((n_image_regions * n_image_embed,))(image_input)
        image_flat  = Dense(1024, activation="relu")(image_reshape)
        image_repeat = RepeatVector(n=max_sentence_len, name='image_repeat')(image_flat)
        if verbose: print('image_repeat output shape:', image_repeat.shape)
            
        # Sentence embedding with dropout
        question_input = Input(shape=(max_sentence_len,), dtype='int32')
        question_embedded = Embedding(input_dim=vocabulary_size, output_dim=sentence_embed_size,
                                      input_length=max_sentence_len)(question_input)  # Can't use masking
        question_embedded = Dropout(0.5)(question_embedded)
        if verbose: print('question_embedded output shape:', question_embedded.shape)

        # Concatenate the image and question, with dropout and dense layer
        merged = concatenate([image_repeat, question_embedded])  # Merge for layers merge for tensors
        lstm = LSTM(sentence_embed_size, return_sequences=False)(merged)
        if verbose: print('lstm output shape:', lstm.shape)
            
        lstm = Dropout(0.5)(lstm)
        output = Dense(units=n_classes, activation='softmax')(lstm)
        if verbose: print('softmax output shape:', output.shape)

        # Build and compile
        self.model = Model(inputs=[image_input, question_input], outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        ''' wrapper around keras.Model.summary()'''
        self.model.summary()
