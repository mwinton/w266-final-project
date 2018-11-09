from collections import OrderedDict
import os

class VGG16Options(object):

    def __init__(self):
        ''' init function.  This class stores options for generating VGG16 embeddings. '''
        
        self.options = OrderedDict()
                
        # Image model parameters
        self.options['vggnet_input_dim'] = 448  # expected x, y dim of VGGNet
        self.options['image_depth'] = 3         # 3 color channels (RGB)
        self.options['n_image_embed'] = 512     # VGGNet
        self.options['n_image_regions'] = 196   # 14x14 regions

        # Training batch size 
        self.options['batch_size'] = 15   # small to avoid OOM errors

    def get_options(self):
        ''' return ordered dict containing all model options '''
        return self.options

