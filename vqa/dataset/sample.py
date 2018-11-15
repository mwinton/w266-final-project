## Datastructures to store questions/answers and images
## Adapted from https://github.com/imatge-upc/vqa-2016-cvprw, Issey Masuda Mora 

import numpy as np
import os

from scipy.misc import imread, imresize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from .types import DatasetType


class VQASample:
    """Class that handles a single VQA dataset sample, thus containing the questions, answer and image.

    Attributes:
        question (Question)
        image (Image)
        answer (Answer): only if dataset_type is different than TEST
        dataset_type (DatasetType): the type of dataset this sample belongs to
    """

    def __init__(self, question, image, answer=None, dataset_type=DatasetType.TRAIN):
        """Instantiate a VQASample.

        Args:
            question (Question): Question object with the question sample
            image (Image): Image object with, at least, the reference to the image path
            answer (Answer): Answer object with the answer sample. If dataset type is TEST, no answer is expected
            dataset_type (DatasetType): type of dataset this sample belongs to. The default is DatasetType.TRAIN
        """
        # Question
        if isinstance(question, Question):
            self.question = question
        else:
            raise TypeError('question has to be an instance of class Question')

        # Answer
        if dataset_type != DatasetType.TEST and dataset_type != DatasetType.EVAL:
            if isinstance(answer, Answer):
                self.answer = answer
            else:
                raise TypeError('answer has to be an instance of class Answer')

        # Image
        if isinstance(image, Image):
            self.image = image
        else:
            raise TypeError('image has to be an instance of class Image')

        # Dataset type
        if isinstance(dataset_type, DatasetType):
            self.sample_type = dataset_type
        else:
            raise TypeError('dataset_type has to be one of the DatasetType defined values')

    def get_input(self, max_sentence_len, mem=True):
        """Gets the prepared input to be injected into the NN.

        Args:
            max_sentence_len (int): The maximum length of the question. The question will be truncated if it's larger
                or padded with zeros if it's shorter

        Returns:
            A list with two items, each of one a NumPy array. The first element contains the question and the second
            one the image, both processed to be ready to be injected into the model
        """

        # Prepare question np array representation
        question = self.question.get_tokens()
        question = pad_sequences([question], max_sentence_len)[0]

        # Prepare image
        image = self.image.features

        if(image.shape[0] == 0) :
            print("Error, image_idx -> {} was not loaded in sample".format(self.image.features_idx))

        return image, question

    def get_output(self):
        if self.sample_type == DatasetType.TEST or self.sample_type == DatasetType.EVAL:
            raise TypeError('This sample is of type DatasetType.TEST or DatasetType.EVAL and thus does not have an '
                            'associated output')

        answer = self.answer.get_tokens()
        one_hot_ans = np.zeros(self.answer.n_answer_classes)

        if answer:
            idx = self.answer.one_hot_index
            # Just to make sure that all answers have appropriate indices assigned
            assert(idx != -1)
            # One-hot vector
            one_hot_ans[idx] = 1

        return one_hot_ans.astype(np.bool_)


class Question:
    """Class that holds the information of a single question of a VQA sample"""

    def __init__(self, question_id, question_str, image_id, tokenizer=None):
        """Instantiates a Question object.

        Args:
            question_id (int): unique question indentifier
            question_str (str): question as a string
            image_id (int): unique image identifier of the image related to this question
            tokenizer (Tokenizer): if given, the question will be tokenized with it
        """
        # Validate id
        try:
            self.id = int(question_id)
            if self.id < 0:
                raise ValueError('question_id has to be a positive integer')
        except:
            raise ValueError('question_id has to be a positive integer')

        # Validate image_id
        try:
            self.image_id = int(image_id)
            if self.id < 0:
                raise ValueError('image_id has to be a positive integer')
        except:
            raise ValueError('image_id has to be a positive integer')

        # Store empty complementary_id to potentially bee populated later
        self.complementary_id = None
        
        # Store question string
        self.question_str = question_str
        self._tokens_idx = []

        # Validate tokenizer class
        if tokenizer:
            if isinstance(tokenizer, Tokenizer):
                self.tokenizer = tokenizer
                self._tokens_idx = self.tokenizer.texts_to_sequences([self.question_str])[0]
            else:
                raise TypeError('The tokenizer param must be an instance of keras.preprocessing.text.Tokenizer')

    def tokenize(self, tokenizer=None):
        """
        Tokenizes the question using the specified tokenizer. If none is provided, it will use the one
        passed in the constructor.

        Returns:
            A list with integer indexes, each index representing a word in the question

        Raises:
            Error in case that a tokenizer hasn't been provided in the method or at any point before
        """

        # replace self.tokenizer if one was passed in
        if tokenizer:
            self.tokenizer = tokenizer

        # enforce that we must have a tokenizer
        if not hasattr(self, 'tokenizer'):
            raise TypeError('tokenizer cannot be of type None, you have to provide an instance of '
                            'keras.preprocessing.text.Tokenizer if you haven\'t provided one yet')

        # texts_to_sequences takes a list of strings and returns a list of sequences
        # because we only pass in one string, we only want the first sequence from returned list
        self._tokens_idx = self.tokenizer.texts_to_sequences([self.question_str])[0]
        
        # This temporary "hack" doesn't work because it returns strings not ints
        # self._tokens_idx = text_to_word_sequence(self.question_str)
        
        return self._tokens_idx

    def get_tokens(self):
        """Return the question index tokens based on the specified tokenizer"""

        return self._tokens_idx

    def get_tokens_length(self):
        """Returns the question length measured in number of tokens"""

        return len(self._tokens_idx)


class Answer:
    """Class that holds the information of a single answer of a VQA sample"""

    def __init__(self, answer_id, answer_str, question_id, image_id, question_type,
                 answer_type, n_answer_classes, tokenizer=None):
        """Instantiates an Answer object.

        Args:
            answer_id (int): unique answer indentifier
            answer (str): answer as a string
            question_id (int): unique question identifier of the question related to this answer
            image_id (int): unique image identifier of the image related to this answer
            question_type (str): type of question (e.g. 'what', 'how many')
            answer_type (str): type of answer (e.g. 'other')
            tokenizer (Tokenizer): if given, the question will be tokenized with it
        """

        # Validate id
        try:
            self.id = int(answer_id)
            if self.id < 0:
                raise ValueError('answer_id has to be a positive integer')
        except:
            raise ValueError('answer_id has to be a positive integer')

        # Validate question_id
        try:
            self.question_id = int(question_id)
            if self.id < 0:
                raise ValueError('question_id has to be a positive integer')
        except:
            raise ValueError('question_id has to be a positive integer')

        # Validate image_id
        try:
            self.image_id = int(image_id)
            if self.id < 0:
                raise ValueError('image_id has to be a positive integer')
        except:
            raise ValueError('image_id has to be a positive integer')

        self.n_answer_classes = n_answer_classes

        # keep question and answer types (no validation on strings)
        self.question_type = question_type
        self.answer_type = answer_type
        
        # will be set later when the top answers are known
        self.one_hot_index = -1 

        self.answer_str = answer_str
        self._tokens_idx = []

        # Validate tokenizer class
        if tokenizer:
            if isinstance(tokenizer, Tokenizer):
                self.tokenizer = tokenizer
                self._tokens_idx = self.tokenizer.texts_to_sequences([self.answer_str])[0]
            else:
                raise TypeError('The tokenizer param must be an instance of keras.preprocessing.text.Tokenizer')

    def tokenize(self, tokenizer=None):
        """Tokenizes the answer using the specified tokenizer. If none provided, it will use the one passed in the
        constructor.

        Returns:
            A list with integer indexes, each index representing a word in the question

        Raises:
            Error in case that a tokenizer hasn't been provided in the method or at any point before
        """

        if tokenizer:
            self.tokenizer = tokenizer
            self._tokens_idx = self.tokenizer.texts_to_sequences([self.answer_str])[0]
        elif self.tokenizer:
            self._tokens_idx = self.tokenizer.texts_to_sequences([self.answer_str])[0]
        else:
            raise TypeError('tokenizer cannot be of type None, you have to provide an instance of '
                            'keras.preprocessing.text.Tokenizer if you haven\'t provided one yet')
        return self._tokens_idx

    def get_tokens(self):
        """Return the question index tokens based on the specified tokenizer"""

        return self._tokens_idx


class Image:

    def __init__(self, image_id, features_idx):
        self.image_id = image_id
        self.features_idx = features_idx
        self.features = np.array([])

    def reset(self):
        del self.features
        self.features = np.array([])

    def load(self, images_features, offset ):

        if len(self.features):
            return self.features
        else:
            if (self.features_idx < offset):
                print("Error: sample index -> {}. offset -> {}".format(self.features_idx,offset))
        
            self.features = images_features[self.features_idx - offset]
            return self.features

