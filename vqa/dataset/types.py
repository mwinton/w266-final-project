## Adapted from https://github.com/imatge-upc/vqa-2016-cvprw, Issey Masuda Mora 

from enum import Enum

class DatasetType(Enum):
    """
        Enumeration with the possible dataset types
    """

    TRAIN = 0
    VALIDATION = 1
    TEST = 2
