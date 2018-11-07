
## Datastructures to store questions/answers and images
## Adapted from https://github.com/imatge-upc/vqa-2016-cvprw, Issey Masuda Mora 



import os

import numpy as np

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

    def get_input(self, question_max_len, mem=True):
        """Gets the prepared input to be injected into the NN.

        Args:
            question_max_len (int): The maximum length of the question. The question will be truncated if it's larger
                or padded with zeros if it's shorter

        Returns:
            A list with two items, each of one a NumPy array. The first element contains the question and the second
            one the image, both processed to be ready to be injected into the model
        """

        # Prepare question
        question = self.question.get_tokens()
        question = pad_sequences([question], question_max_len)[0]

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
        one_hot_ans = np.zeros(self.answer.vocab_size)

        if answer:
            # TODO: extend to multiple word answers
            idx = answer[0]  # Get only one word
            # One-hot vector
            one_hot_ans[idx] = 1

        return one_hot_ans.astype(np.bool_)


class Question:
    """Class that holds the information of a single question of a VQA sample"""

    def __init__(self, question_id, question, image_id, vocab_size, tokenizer=None):
        """Instantiates a Question object.

        Args:
            question_id (int): unique question indentifier
            question (str): question as a string
            image_id (int): unique image identifier of the image related to this question
            vocab_size (int): size of the vocabulary
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

        # Validate vocab_size
        try:
            self.vocab_size = int(vocab_size)
            if self.vocab_size < 0:
                raise ValueError('vocab_size has to be a positive integer')
        except:
            raise ValueError('vocab_size has to be a positive integer')

        self.question = question
        self._tokens_idx = []

        # Validate tokenizer class
        if tokenizer:
            if isinstance(tokenizer, Tokenizer):
                self.tokenizer = tokenizer
                self._tokens_idx = self.tokenizer.texts_to_sequences([self.question])[0]
            else:
                raise TypeError('The tokenizer param must be an instance of keras.preprocessing.text.Tokenizer')

    def tokenize(self, tokenizer=None):
        """Tokenizes the question using the specified tokenizer. If none provided, it will use the one passed in the
        constructor.

        Returns:
            A list with integer indexes, each index representing a word in the question

        Raises:
            Error in case that a tokenizer hasn't been provided in the method or at any point before
        """

        if tokenizer:
            self.tokenizer = tokenizer
            self._tokens_idx = self.tokenizer.texts_to_sequences([self.question])[0]
        elif self.tokenizer:
            self._tokens_idx = self.tokenizer.texts_to_sequences([self.question])[0]
        else:
            raise TypeError('tokenizer cannot be of type None, you have to provide an instance of '
                            'keras.preprocessing.text.Tokenizer if you haven\'t provided one yet')

        # Update question_max_length if not provided

        return self._tokens_idx

    def get_tokens(self):
        """Return the question index tokens based on the specified tokenizer"""

        return self._tokens_idx

    def get_tokens_length(self):
        """Returns the question length measured in number of tokens"""

        return len(self._tokens_idx)


class Answer:
    """Class that holds the information of a single answer of a VQA sample"""

    def __init__(self, answer_id, answer, question_id, image_id, vocab_size, tokenizer=None):
        """Instantiates an Answer object.

        Args:
            answer_id (int): unique answer indentifier
            answer (str): answer as a string
            question_id (int): unique question identifier of the question related to this answer
            image_id (int): unique image identifier of the image related to this answer
            vocab_size (int): size of the vocabulary
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

        # Validate vocab_size
        try:
            self.vocab_size = int(vocab_size)
            if self.vocab_size < 0:
                raise ValueError('vocab_size has to be a positive integer')
        except:
            raise ValueError('vocab_size has to be a positive integer')

        self.answer = answer
        self._tokens_idx = []

        # Validate tokenizer class
        if tokenizer:
            if isinstance(tokenizer, Tokenizer):
                self.tokenizer = tokenizer
                self._tokens_idx = self.tokenizer.texts_to_sequences([self.answer])[0]
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
            self._tokens_idx = self.tokenizer.texts_to_sequences([self.answer])[0]
        elif self.tokenizer:
            self._tokens_idx = self.tokenizer.texts_to_sequences([self.answer])[0]
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

