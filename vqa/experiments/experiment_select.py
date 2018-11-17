import json

from vqa import BASE_DIR
from vqa.model.model_select import ModelLibrary
from vqa.model.options import ModelOptions 

class ExperimentLibrary:
    # ---------------------------------- CONSTANTS --------------------------------
    # Model identifiers
    EXPERIMENT_0   = 0  # no experiment

    # Path
    EXPERIMENTS_PATH = BASE_DIR + 'vqa/experiments/'

    # ---------------------------------- FUNCTIONS --------------------------------

    def __init__(self):
        pass

    @staticmethod
    def get_experiment(id, options):
        '''
            Load corresponding json file and override options defaults.  Any key-value pairs in the json
            file will be added to the options object, overriding existing values.
        '''
        
        print('Setting up experiment {}'.format(id))
        options['experiment_id'] = id
        expt_path = '{}{}_experiment_{}.json'.format(options['experiments_path'], options['user_name'], id)

        # experiment 0 means no changes to options.py
        if id == ExperimentLibrary.EXPERIMENT_0:
            return options
        
        print('\nLoading experiment json from ->', expt_path)
        expt_json = json.load(open(expt_path))
        
        # if there's no json experiment_id attribute, get it from the CLI `id` arg
        if not 'experiment_id' in expt_json:
            options['experiment_id'] = id

        # raise an exception if CLI `id` arg (ie. filename) and json attribute don't match
        if id != expt_json.get('experiment_id', id):
            raise ValueError('If \"experiment_id\" json attribute is present, it must agree with the ID in the filename.')

        # make sure a valid model was selected
        if expt_json.get('model_name', None) not in ModelLibrary.get_valid_model_names():
            raise KeyError('Valid \"model_name\" must be specified in the experiment json file. Choices: {}'.format
                           (ModelLibrary.get_valid_model_names()))

        if 'optimizer' in expt_json:
            # set default optimizer params (learning rate, etc...) for the selected optimizer
            # It's possible they will still get overwritten, but this allows experiment to select
            # a different optimizer without having to specify all of its parameters
            options['optimizer'] = expt_json['optimizer']
            ModelOptions.set_optimizer_params(options)
        
        # update existing values, or create new ones (with warning)
        for key, val in expt_json.items():
            if key in options:
                options[key] = val
                print('Updated: options[\'{}\'] = {}'.format(key, val))
            else:
                options[key] = val
                print('WARNING: new parameter, options[\'{}\'] = {}. Was this intentional?'.format(key, val))
            
        # Make sure every experiment has a name; needed for MLFlow logging
        if options.get('experiment_name', None) == None:
            options['experiment_name'] = 'experiment_{}'.format(id)
        else:
            options['experiment_name'] = options['experiment_name'].replace(' ', '_')

        return options
