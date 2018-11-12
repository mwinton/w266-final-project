
## Adapted from https://github.com/imatge-upc/vqa-2016-cvprw, Issey Masuda Mora 


from keras.layers import Input, Embedding, merge, LSTM, Dropout, Dense, RepeatVector, BatchNormalization, \
    TimeDistributed, Reshape, concatenate
from keras.models import Model, model_from_json

from . baseline_model import BaselineModel
from . san_model import StackedAttentionNetwork
from . text_cnn_model import TextCNNModel
from . vggnet_model import VGGNetModel

class ModelLibrary:
    # ---------------------------------- CONSTANTS --------------------------------

    # Model identifiers
    MODEL_BASELINE = "baseline"     # Base model
    MODEL_SAN      = "san"          # SAN Model  (To reproduce Yang's paper results)
    MODEL_TEXT_CNN = "text_cnn"     # Text-only CNN Model
    MODEL_VGGNET   = "vggnet_only"  # Image-only VGGNet Model

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
        elif model_name == ModelLibrary.MODEL_TEXT_CNN:
            return ModelLibrary.get_text_cnn_model(options)
        elif model_name == ModelLibrary.MODEL_VGGNET:
            return ModelLibrary.get_vggnet_only_model(options)
        else:
            print("Model not registered")
  

    @staticmethod
    def get_baseline_model(options):
        
        print('Creating model...')
        blm  = BaselineModel(options)
        blm.build_graph(options)

        vqa_model = blm.model
        print('Model created and compiled')
        vqa_model.summary()

        return vqa_model

    @staticmethod
    def get_san_model(options):

        print('Creating model...')
        san  = StackedAttentionNetwork(options)
        san.build_graph(options)

        vqa_model = san.model
        print('Model created and compiled')
        vqa_model.summary()

        return vqa_model

    @staticmethod
    def get_text_cnn_model(options):

        print('Creating model...')
        tcnn  = TextCNNModel(options)
        tcnn.build_graph(options)

        vqa_model = tcnn.model
        print('Model created and compiled')
        vqa_model.summary()

        return vqa_model

    @staticmethod
    def get_vggnet_only_model(options):

        print('Creating model...')
        vgg  = VGGNetModel(options)
        vgg.build_graph(options)

        vqa_model = vgg.model
        print('Model created and compiled')
        vqa_model.summary()

        return vqa_model



