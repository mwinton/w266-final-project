from keras.layers import Input, Embedding, merge, LSTM, Dropout, Dense, RepeatVector, BatchNormalization, \
    TimeDistributed, Reshape, concatenate
from keras.models import Model, model_from_json
from keras.optimizers import Adam

from vqa import BASE_DIR


class ModelLibrary:
    # ---------------------------------- CONSTANTS --------------------------------
    # Model identifiers
    MODEL_ONE = 1       # Base model
    MODEL_TWO = 2       # Batch normalization
    MODEL_THREE = 3     # Modified learning rate
    MODEL_FOUR = 4      # Multi-word answer
    MODEL_FIVE = 5      # Sentence embedding
    MODEL_SIX = 6       # Sentence embedding + batch norm

    # Generic model parameters
    EMBED_HIDDEN_SIZE = 100

    # Path
    MODELS_PATH = BASE_DIR + 'models/'

    # ---------------------------------- FUNCTIONS --------------------------------

    def __init__(self):
        pass

    @classmethod
    def get_valid_model_nums(cls):
        valid_nums = [cls.__dict__[key] for key in cls.__dict__.keys() if key.startswith('MODEL_')]
        valid_nums.sort()
        return valid_nums

    @staticmethod
    def get_model(model_num, vocabulary_size=20000, question_max_len=22, answer_max_len=5):
        if model_num == ModelLibrary.MODEL_ONE:
            return ModelLibrary.get_model_one(vocabulary_size, question_max_len)
        elif model_num == ModelLibrary.MODEL_TWO:
            return ModelLibrary.get_model_two(vocabulary_size, question_max_len)
        elif model_num == ModelLibrary.MODEL_THREE:
            return ModelLibrary.get_model_three(vocabulary_size, question_max_len)
        elif model_num == ModelLibrary.MODEL_FOUR:
            return ModelLibrary.get_model_four(vocabulary_size, question_max_len, answer_max_len)
        elif model_num == ModelLibrary.MODEL_FIVE:
            return ModelLibrary.get_model_five(vocabulary_size, question_max_len)
        elif model_num == ModelLibrary.MODEL_SIX:
            return ModelLibrary.get_model_six(vocabulary_size, question_max_len)

    @staticmethod
    def get_model_one(vocabulary_size=20000, question_max_len=22):
        model_num = ModelLibrary.MODEL_ONE
        model_path = ModelLibrary.MODELS_PATH + 'model_{}.json'.format(model_num)
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
            image_input = Input(batch_shape=(None,196,512))
            image_reshape  = Reshape((196*512,))(image_input)
            image_flat  = Dense(1024,activation="relu")(image_reshape)
            image_repeat = RepeatVector(n=question_max_len)(image_flat)

            # Question
            question_input = Input(shape=(question_max_len,), dtype='int32')
            question_embedded = Embedding(input_dim=vocabulary_size, output_dim=ModelLibrary.EMBED_HIDDEN_SIZE,
                                          input_length=question_max_len)(question_input)  # Can't use masking
            question_embedded = Dropout(0.5)(question_embedded)

            # Merge
            merged = concatenate([image_repeat, question_embedded])  # Merge for layers merge for tensors
            x = LSTM(ModelLibrary.EMBED_HIDDEN_SIZE, return_sequences=False)(merged)
            x = Dropout(0.5)(x)
            output = Dense(output_dim=vocabulary_size, activation='softmax')(x)

            vqa_model = Model(input=[image_input, question_input], output=output)
            print('Model created')

            print('Compiling model...')
            vqa_model.compile(optimizer='adam', loss='categorical_crossentropy')
            vqa_model.summary()
            print('Model compiled')

            #print('Saving model...')
            #model_json = vqa_model.to_json()
            #with open(model_path, 'w') as f:
            #    f.write(model_json)
            #print('Model saved')

        return vqa_model

    @staticmethod
    def get_model_two(vocabulary_size=20000, question_max_len=22):
        model_num = ModelLibrary.MODEL_TWO
        model_path = ModelLibrary.MODELS_PATH + 'model_{}.json'.format(model_num)
        # Optimizer
        adam = Adam(lr=1e-4)
        try:
            with open(model_path, 'r') as f:
                print('Loading model...')
                vqa_model = model_from_json(f.read())
                print('Model loaded')
                print('Compiling model...')
                vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
                print('Model compiled')
        except IOError:
            print('Creating model...')
            # Image
            image_input = Input(shape=(1024,))
            image_repeat = RepeatVector(n=question_max_len)(image_input)

            # Question
            question_input = Input(shape=(question_max_len,), dtype='int32')
            question_embedded = Embedding(input_dim=vocabulary_size, output_dim=ModelLibrary.EMBED_HIDDEN_SIZE,
                                          input_length=question_max_len)(question_input)  # Can't use masking
            question_embedded = Dropout(0.5)(question_embedded)

            # Merge
            merged = merge([image_repeat, question_embedded], mode='concat')  # Merge for layers merge for tensors
            x = BatchNormalization()(merged)
            x = LSTM(ModelLibrary.EMBED_HIDDEN_SIZE, return_sequences=False)(x)
            x = Dropout(0.5)(x)
            output = Dense(output_dim=vocabulary_size, activation='softmax')(x)

            vqa_model = Model(input=[image_input, question_input], output=output)
            print('Model created')

            print('Compiling model...')
            vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
            print('Model compiled')

            print('Saving model...')
            model_json = vqa_model.to_json()
            with open(model_path, 'w') as f:
                f.write(model_json)
            print('Model saved')

        return vqa_model

    @staticmethod
    def get_model_three(vocabulary_size=20000, question_max_len=22):
        model_num = ModelLibrary.MODEL_THREE
        model_path = ModelLibrary.MODELS_PATH + 'model_{}.json'.format(model_num)
        # Optimizer
        adam = Adam(lr=1e-4)
        # Create/load model
        try:
            with open(model_path, 'r') as f:
                print('Loading model...')
                vqa_model = model_from_json(f.read())
                print('Model loaded')
                print('Compiling model...')
                vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
                print('Model compiled')
        except IOError:
            print('Creating model...')
            # Image
            image_input = Input(shape=(1024,))
            image_repeat = RepeatVector(n=question_max_len)(image_input)

            # Question
            question_input = Input(shape=(question_max_len,), dtype='int32')
            question_embedded = Embedding(input_dim=vocabulary_size, output_dim=ModelLibrary.EMBED_HIDDEN_SIZE,
                                          input_length=question_max_len)(question_input)  # Can't use masking
            question_embedded = Dropout(0.5)(question_embedded)

            # Merge
            merged = merge([image_repeat, question_embedded], mode='concat')  # Merge for layers merge for tensors
            x = LSTM(ModelLibrary.EMBED_HIDDEN_SIZE, return_sequences=False)(merged)
            x = Dropout(0.5)(x)
            output = Dense(output_dim=vocabulary_size, activation='softmax')(x)

            vqa_model = Model(input=[image_input, question_input], output=output)
            print('Model created')

            print('Compiling model...')
            vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
            print('Model compiled')

            print('Saving model...')
            model_json = vqa_model.to_json()
            with open(model_path, 'w') as f:
                f.write(model_json)
            print('Model saved')

        return vqa_model

    @staticmethod
    def get_model_four(vocabulary_size=20000, question_max_len=22, answer_max_len=5):
        model_num = ModelLibrary.MODEL_FOUR
        model_path = ModelLibrary.MODELS_PATH + 'model_{}.json'.format(model_num)
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
            image_input = Input(shape=(1024,))
            image_repeat = RepeatVector(n=question_max_len)(image_input)

            # Question
            question_input = Input(shape=(question_max_len,), dtype='int32')
            question_embedded = Embedding(input_dim=vocabulary_size, output_dim=ModelLibrary.EMBED_HIDDEN_SIZE,
                                          input_length=question_max_len)(question_input)  # Can't use masking
            question_embedded = Dropout(0.5)(question_embedded)

            # Merge question and image branches
            merged = merge([image_repeat, question_embedded], mode='concat')  # Merge for layers merge for tensors
            x = LSTM(ModelLibrary.EMBED_HIDDEN_SIZE, return_sequences=False)(merged)
            x = Dropout(0.5)(x)
            x = RepeatVector(n=answer_max_len)(x)
            # Merge the feedback (output) with the previous tensor
            feedback_input = Input(shape=(answer_max_len, vocabulary_size))
            x = merge([x, feedback_input], mode='concat')
            # Generate the multi-word answer
            x = LSTM(ModelLibrary.EMBED_HIDDEN_SIZE, return_sequences=True)(x)
            x = Dropout(0.5)(x)
            output = TimeDistributed(Dense(output_dim=vocabulary_size, activation='softmax'))(x)

            vqa_model = Model(input=[image_input, question_input, feedback_input], output=output)
            print('Model created')

            print('Compiling model...')
            vqa_model.compile(optimizer='adam', loss='categorical_crossentropy')
            print('Model compiled')

            print('Saving model...')
            model_json = vqa_model.to_json()
            with open(model_path, 'w') as f:
                f.write(model_json)
            print('Model saved')

        return vqa_model

    @staticmethod
    def get_model_five(vocabulary_size=20000, question_max_len=22):
        model_num = ModelLibrary.MODEL_FIVE
        model_path = ModelLibrary.MODELS_PATH + 'model_{}.json'.format(model_num)
        # Params
        lstm_hidden_units = 256
        # Optimizer
        adam = Adam(lr=1e-4)
        # Create/load model
        try:
            with open(model_path, 'r') as f:
                print('Loading model...')
                vqa_model = model_from_json(f.read())
                print('Model loaded')
                print('Compiling model...')
                vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
                print('Model compiled')
        except IOError:
            print('Creating model...')
            # Image
            image_input = Input(shape=(1024,))
            image_features = Dense(output_dim=lstm_hidden_units, activation='relu')(image_input)
            image_features = Dropout(0.5)(image_features)

            # Question
            question_input = Input(shape=(question_max_len,), dtype='int32')
            question_embedded = Embedding(input_dim=vocabulary_size, output_dim=ModelLibrary.EMBED_HIDDEN_SIZE,
                                          input_length=question_max_len)(question_input)  # Can't use masking
            question_embedded = Dropout(0.5)(question_embedded)
            sentence_embedded = LSTM(lstm_hidden_units, return_sequences=False)(question_embedded)
            sentence_embedded = Dropout(0.5)(sentence_embedded)

            # Merge
            merged = merge([image_features, sentence_embedded], mode='sum')  # Merge for layers, merge for tensors
            output = Dense(output_dim=vocabulary_size, activation='softmax')(merged)

            vqa_model = Model(input=[image_input, question_input], output=output)
            print('Model created')

            print('Compiling model...')
            vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
            print('Model compiled')

            print('Saving model...')
            model_json = vqa_model.to_json()
            with open(model_path, 'w') as f:
                f.write(model_json)
            print('Model saved')

        return vqa_model

    @staticmethod
    def get_model_six(vocabulary_size=20000, question_max_len=22):
        model_num = ModelLibrary.MODEL_SIX
        model_path = ModelLibrary.MODELS_PATH + 'model_{}.json'.format(model_num)
        # Params
        lstm_hidden_units = 256
        # Optimizer
        adam = Adam(lr=1e-5)
        # Create/load model
        try:
            with open(model_path, 'r') as f:
                print('Loading model...')
                vqa_model = model_from_json(f.read())
                print('Model loaded')
                print('Compiling model...')
                vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
                print('Model compiled')
        except IOError:
            print('Creating model...')
            # Image
            image_input = Input(shape=(1024,))
            image_features = Dense(output_dim=lstm_hidden_units, activation='relu')(image_input)
            image_features = Dropout(0.5)(image_features)

            # Question
            question_input = Input(shape=(question_max_len,), dtype='int32')
            question_embedded = Embedding(input_dim=vocabulary_size, output_dim=ModelLibrary.EMBED_HIDDEN_SIZE,
                                          input_length=question_max_len)(question_input)  # Can't use masking
            question_embedded = Dropout(0.5)(question_embedded)
            sentence_embedded = LSTM(lstm_hidden_units, return_sequences=False)(question_embedded)
            sentence_embedded = Dropout(0.5)(sentence_embedded)

            # Merge
            merged = merge([image_features, sentence_embedded], mode='sum')  # Merge for layers, merge for tensors
            merged = BatchNormalization()(merged)
            output = Dense(output_dim=vocabulary_size, activation='softmax')(merged)

            vqa_model = Model(input=[image_input, question_input], output=output)
            print('Model created')

            print('Compiling model...')
            vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
            print('Model compiled')

            print('Saving model...')
            model_json = vqa_model.to_json()
            with open(model_path, 'w') as f:
                f.write(model_json)
            print('Model saved')

        return vqa_model
