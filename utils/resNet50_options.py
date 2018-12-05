from collections import OrderedDict
import os

class resNet50Options(object):
    """
        Class containing parameters needed in the ResNet50 model
    """

    def __init__(self):
        """
            Init function setting the values of the options.
        """
        
        self.options = OrderedDict()

        self.options["model_name"] = 'resNet50'
                
        # Image model parameters
        self.options['input_dim'] = 448         # expected x, y dim of resNet 
        self.options['image_depth'] = 3         # 3 color channels (RGB)
        self.options['n_image_embed'] = 512     # resNet
        self.options['n_image_regions'] = 196   # 14x14 regions

        # Training batch size 
        self.options['batch_size'] = 15   # small to avoid OOM errors

    def get_options(self):
        """
            Returns: OrderedDict containing all image model options
        """

        return self.options

