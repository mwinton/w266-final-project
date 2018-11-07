from vqa import BASE_DIR

class ExperimentLibrary:
    # ---------------------------------- CONSTANTS --------------------------------
    # Model identifiers
    EXPERIMENT_0   = 0  # no experiment
    EXPERIMENT_1   = 1  # concise description of experiment

    # Path
    EXPERIMENTS_PATH = BASE_DIR + 'vqa/experiments/'

    # ---------------------------------- FUNCTIONS --------------------------------

    def __init__(self):
        pass

    @classmethod
    def get_valid_experiment_ids(cls):
        valid_ids = [cls.__dict__[key] for key in cls.__dict__.keys() if key.startswith('EXPERIMENT_')]
        valid_ids.sort()
        print("Valid Experiment IDs:", valid_ids)
        return valid_ids

    @staticmethod
    def get_experiment(id, options):
        '''
            Load corresponding json file and override options defaults
        '''
        
        print('Setting up experiment {}'.format(id))
        options['experiment_id'] = id
        
        if id != ExperimentLibrary.EXPERIMENT_0:
            # TODO: implement experiment loading
            pass
        
        return options
