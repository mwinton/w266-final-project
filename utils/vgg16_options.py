from collections import OrderedDict
import os

class VGG16Options(object):
    """
        Class containing parameters needed in the VGGNet16 model
    """

    def __init__(self):
        """
            Init function setting the values of the options.
        """
        
        self.options = OrderedDict()

        self.options["model_name"] = 'vgg16'
                
        # Image model parameters
        self.options['input_dim'] = 448         # expected x, y dim of VGGNet
        self.options['image_depth'] = 3         # 3 color channels (RGB)
        self.options['n_image_embed'] = 512     # VGGNet
        self.options['n_image_regions'] = 196   # 14x14 regions

        # Training batch size 
        self.options['batch_size'] = 15   # small to avoid OOM errors

    def get_options(self):
        """
            Returns: OrderedDict containing all image model options
        """

        return self.options

