import json

from vqa import BASE_DIR
from vqa.model.model_select import ModelLibrary

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
        expt_path = '{}experiment_{}.json'.format(options['experiments_path'], id)
        
        if id != ExperimentLibrary.EXPERIMENT_0:
            print('Trying to load json from ->', expt_path)
            expt_json = json.load(open(expt_path))
            if not 'experiment_id' in expt_json:
                raise KeyError('Unique integer \"experiment_id\" must be specified in the experiment json file.')
            if expt_json.get('model_name', None) not in ModelLibrary.get_valid_model_names():
                raise KeyError('Valid \"model_name\" must be specified in the experiment json file. Choices: {}'.format
                              (ModelLibrary.get_valid_model_names()))
            for key, val in expt_json.items():
                options[key] = val
            
        # Make sure every experiment has a name; needed for MLFlow logging
        if options.get('experiment_name', None) == None:
            options['experiment_name'] = 'experiment_{}'.format(id)
        else:
            options['experiment_name'] = options['experiment_name'].replace(' ', '_')

        return options
