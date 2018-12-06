## Datastructures to store questions/answers and images
## Adapted from https://github.com/imatge-upc/vqa-2016-cvprw, Issey Masuda Mora 

import numpy as np
import os

from scipy.misc import imread, imresize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
from .types import DatasetType
from ..model.options import ModelOptions
import nltk


class VQASample:
    """Class that handles a single VQA dataset sample, thus containing the questions, answer and image.

    Attributes:
        question (Question)
        image (Image)
        answer (Answer): only if dataset_type is different than TEST
        dataset_type (DatasetType): the type of dataset this sample belongs to
    """

    def __init__(self, question, image, answer=None, dataset_type=DatasetType.TRAIN, val_test_split=False):
        """Instantiate a VQASample.

        Args:
            question (Question): Question object with the question sample
            image (Image): Image object with, at least, the reference to the image path
            answer (Answer): Answer object with the answer sample. If dataset type is TEST, no answer is expected
            dataset_type (DatasetType): type of dataset this sample belongs to. The default is DatasetType.TRAIN
            val_test_split: boolean specifying whether half of validation dataset is reserved for "test"
            
        Returns:
            no return value
        """
        
        # Question
        if isinstance(question, Question):
            self.question = question
        else:
            raise TypeError('question has to be an instance of class Question')

        # Answer
        if (dataset_type != DatasetType.TEST) or \
        (dataset_type == DatasetType.TEST and val_test_split):
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


    def get_input(self, max_sentence_len, need_pos_tags = False ):
        """
            Gets the prepared input to be injected into the NN.

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

        if need_pos_tags:
            tags = self.question.get_pos_tags()
            tags = pad_sequences([tags], max_sentence_len)[0]


        # Prepare image
        image = self.image.features

        if(image.shape[0] == 0) :
            print("Error, image_idx -> {} was not loaded in sample".format(self.image.features_idx))

        if not need_pos_tags:
            return image, question
        else:
            return image, question, tags

    def get_output(self):
        """
            Provides the one-hot vector to be yielded by dataset's batch_generator (ie. the label used in model training)
            
            Returns:
                one-hot encoding vector for this sample
        """
        
        if self.sample_type == DatasetType.TEST:
            raise TypeError('This sample is of type DatasetType.TEST and thus does not have an associated output.')

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
    """
        Class that holds the information of a single question of a VQA sample
    """

    def __init__(self, question_id, question_str, image_id):
        """
            Instantiates a Question object.

            Args:
                question_id (int): unique question ID
                question_str (str): question as a string
                image_id (int): unique image ID of the associated image
                
            Returns:
                no return value
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

        # Store empty complement_id to potentially be populated later
        self.complement_id = None
        
        # Store question string
        self.question_str = question_str
        self._tokens_idx = []


    def tokenize(self, tokenizer, need_pos_tags):
        """
            Tokenizes the question using the specified tokenizer.

            Args:
                tokenizer: Keras tokenizer to use for tokenizing the sample
                need_pos_tags: boolean indicating whether to do POS tagging
                
            Returns:
                A list with integer indexes, each index representing a word in the question
        """

        # texts_to_sequences takes a list of strings and returns a list of sequences
        # because we only pass in one string, we only want the first sequence from returned list
        self._tokens_idx = tokenizer.texts_to_sequences([self.question_str])[0]
        if need_pos_tags:
            tagged_question = nltk.pos_tag(text_to_word_sequence(self.question_str))  
            self._tag_list = []
            for token,tag in tagged_question:
                # since tokenizer is only built on training set vocab, some words might be missing from validation/test set.
                if token in tokenizer.word_index:
                    if tag not in ModelOptions.tag_to_num:
                        print("For question {} \n Invalid tag {} found in {}".format(self.question_str,tag,tagged_question))
                        self._tag_list.append(0)
                    else:
                        self._tag_list.append(ModelOptions.tag_to_num[tag])

            if (len(self._tag_list) != len(self._tokens_idx)):
                print(" Mismatched token and tag lists \n")
                print("Question =>", self.question_str)
                print("tagged_question =>", tagged_question)
                print("Tag list ->",self._tag_list)
                print("Token list => ", self._tokens_idx)

            assert(len(self._tag_list) == len(self._tokens_idx))
        return self._tokens_idx

    def get_tokens(self):
        """
            Get the question index tokens based on the specified tokenizer
            
            Returns:
                list of token IDs
        """

        return self._tokens_idx

    def get_pos_tags(self):
        """
            Return the question pos tags for each of the words in the question
            
            Returns:
                list of POS tags
        """

        return self._tag_list

    def get_tokens_length(self):
        """
            Returns the question length measured in number of tokens
            
            Returns:
                integer number of tokens
        """

        return len(self._tokens_idx)


class Answer:
    """
        Class that holds the information of a single answer of a VQA sample
    """

    def __init__(self, answer_id, answer_str, question_id, image_id, question_type,
                 answer_type, annotations, n_answer_classes):
        """
            Instantiates an Answer object.

            Args:
                answer_id (int): unique answer indentifier
                answer_str(str): answer as a string
                question_id (int): unique question identifier of the question related to this answer
                image_id (int): unique image identifier of the image related to this answer
                question_type (str): type of question (e.g. 'what', 'how many')
                answer_type (str): type of answer (e.g. 'other')
                annotations (list): list of strings representing answers by 10 human raters
                n_answer_classes (int): number of class labels 

            Returns:
                no return value
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

        # keep question type, answer type, annotations (no validation on strings)
        self.question_type = question_type
        self.answer_type = answer_type
        self.annotations = annotations
        
        # will be set later when the top answers are known
        self.one_hot_index = -1 

        self.answer_str = answer_str
        self._tokens_idx = []

    def tokenize(self, tokenizer):
        """
            Tokenizes the answer using the specified tokenizer.  All words are tokenized, not just top k.
            
            Arg:
                tokenizer = Keras Tokenizer to use for tokenizing the answer

            Returns:
                A list with integer indexes, each index representing a word in the answer
        """

        # texts_to_sequences takes a list of strings and returns a list of sequences
        # because we only pass in one string, we only want the first sequence from returned list
        self._tokens_idx = tokenizer.texts_to_sequences([self.answer_str])[0]
        return self._tokens_idx

    def get_tokens(self):
        """
            Get the answer index tokens based on the specified tokenizer
            
            Returns:
                list of token IDs
       """

        return self._tokens_idx


class Image:
    """
        Class that holds the information of a single image of a VQA sample
    """

    def __init__(self, image_id, features_idx):
        """
            Instantiates an Image object.

            Args:
                image_id (int): unique image identifier of the image related to this answer
                features_idx: index of the image filename
                
            Returns:
                no return value

        """
        self.image_id = image_id
        self.features_idx = features_idx
        self.features = np.array([])

    def reset(self):
        """
            Method to reset an image if necessary.
            
            Returns:
                no return value
        """
        
        del self.features
        self.features = np.array([])

    def load(self, images_features, offset ):
        """
            Returns the image's features
            
            Args:
                images_features: list of image features
                offset: starting offset into the list
        """
        
        if len(self.features):
            return self.features
        else:
            if (self.features_idx < offset):
                print("Error: sample index -> {}. offset -> {}".format(self.features_idx,offset))
        
            self.features = images_features[self.features_idx - offset]
            return self.features

