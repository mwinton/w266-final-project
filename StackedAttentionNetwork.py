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
    
    def build_graph (self, options):
        ''' Build Keras graph '''
        if options['verbose']:
            print('Building graph...')
    
    def train (self, options):
        ''' Train graph '''
        if options['verbose']:
            print('Training...')
    
    def predict (self, options):
        ''' Make predictions '''
        if options['verbose']:
            print('Predicting...')