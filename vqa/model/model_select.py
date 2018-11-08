
## Adapted from https://github.com/imatge-upc/vqa-2016-cvprw, Issey Masuda Mora 


from keras.layers import Input, Embedding, merge, LSTM, Dropout, Dense, RepeatVector, BatchNormalization, \
    TimeDistributed, Reshape, concatenate
from keras.models import Model, model_from_json
from keras.optimizers import Adam

from vqa import BASE_DIR
from . san_model import StackedAttentionNetwork


class ModelLibrary:
    # ---------------------------------- CONSTANTS --------------------------------
    # Model identifiers
    MODEL_BASELINE = "baseline"        # Base model
    MODEL_SAN      = "san"       # SAN Model  (To reproduce Yang's paper results)


    # Path
    MODELS_PATH = BASE_DIR + 'saved_models/'

    # ---------------------------------- FUNCTIONS --------------------------------

    def __init__(self):
        pass

    @classmethod
    def get_valid_model_names(cls):
        valid_names = [cls.__dict__[key] for key in cls.__dict__.keys() if key.startswith('MODEL_')]
        valid_names.sort()
        print("Valid Model Names",valid_names)
        return valid_names

    @staticmethod
    def get_model(options):

        model_name = options['model_name']

        if model_name == ModelLibrary.MODEL_BASELINE:
            return ModelLibrary.get_baseline_model(options)
        elif model_name == ModelLibrary.MODEL_SAN:
            return ModelLibrary.get_san_model(options)
        else:
            print("Model not registered")
  

    @staticmethod
    def get_baseline_model(options):
        model_name = ModelLibrary.MODEL_BASELINE
        model_path = ModelLibrary.MODELS_PATH + 'model_{}.json'.format(model_name)
        try:
            with open(model_path, 'r') as f:
                print('Loading model...')
                vqa_model = model_from_json(f.read())
                print('Model loaded')
                print('Compiling model...')
                vqa_model.compile(optimizer='adam', loss='categorical_crossentropy')
                print('Model compiled')
        except IOError:
            print('Creating model...')
            # Image

            n_image_regions  = options['n_image_regions'] # 196 (14x14)
            n_image_embed    = options['n_image_embed']   # 512 channels VGGNet
            max_sentence_len = options['max_sentence_len']
            vocabulary_size  = options['n_vocab']
            sentence_embed_size  = options['n_sent_embed']
            n_answer_classes   = options['n_answer_classes']

            image_input = Input(batch_shape=(None,n_image_regions,n_image_embed))
            image_reshape  = Reshape((n_image_regions * n_image_embed,))(image_input)
            image_flat  = Dense(1024,activation="relu")(image_reshape)
            image_repeat = RepeatVector(n=max_sentence_len)(image_flat)

            # Question
            question_input = Input(shape=(max_sentence_len,), dtype='int32')
            question_embedded = Embedding(input_dim=vocabulary_size, output_dim=sentence_embed_size,
                                          input_length=max_sentence_len)(question_input)  # Can't use masking
            question_embedded = Dropout(0.5)(question_embedded)

            # Merge
            merged = concatenate([image_repeat, question_embedded])  # Merge for layers merge for tensors
            x = LSTM(sentence_embed_size, return_sequences=False)(merged)
            x = Dropout(0.5)(x)
            output = Dense(units=n_answer_classes, activation='softmax')(x)

            vqa_model = Model(inputs=[image_input, question_input], outputs=output)
            print('Model created')

            print('Compiling model...')
            vqa_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            vqa_model.summary()
            print('Model compiled')

            """
            print('Saving model...')
            model_json = vqa_model.to_json()
            with open(model_path, 'w') as f:
                f.write(model_json)
            print('Model saved')
            """

        return vqa_model

    @staticmethod
    def get_san_model(options):
        model_name = ModelLibrary.MODEL_SAN
        model_path = ModelLibrary.MODELS_PATH + 'model_{}.json'.format(model_name)

        optimizer     = options['optimizer']
        loss_function = options['loss_function']

        try:
            with open(model_path, 'r') as f:
                print('Loading model...')
                vqa_model = model_from_json(f.read())
                print('Model loaded')
                print('Compiling model...')
                vqa_model.compile(optimizer=optimizer, loss=loss_function)
                print('Model compiled')
        except IOError:
            print('Creating model...')
            # Image

            san  = StackedAttentionNetwork(options)
            san.build_graph(options)

            vqa_model = san.model

            print('Model created')

            vqa_model.summary()
            print('Model compiled')

            """
            print('Saving model...')
            model_json = vqa_model.to_json()
            with open(model_path, 'w') as f:
                f.write(model_json)
            print('Model saved')
            """

        return vqa_model


