# File to load the dataset and prepare batches for training 
# Adapted from https://github.com/imatge-upc/vqa-2016-cvprw, Issey Masuda Mora 
# 
# Added Memory management related code for Image Embeddings

import datetime
import h5py
import inspect
import json
import math
import numpy as np
import os
import pickle
import random
import scipy.io
import time

from collections import Counter, defaultdict

from keras.preprocessing.text import Tokenizer

from .sample import Question, Answer, Image, VQASample
from .types import DatasetType
from ..model.options import ModelOptions
from .process_tokens import process_sentence, process_answer


class VQADataset:
    """Class that holds a dataset with VQASample instances.

    Wrapper that eases the process of dataset management. To be able to use it, after object instantiation call the
    method prepare().

    Attributes:
        dataset_type (DatasetType):
        questions_path (str):
        images_path (str):
        answers_path (str): only if dataset_type is not DatasetType.TEST
        vocab_size (int):
    """

    def __init__(self, dataset_type, options):
        """Instantiate a new VQADataset that will hold the whole dataset.

        Args:
            dataset_type (DatasetType): type of dataset
            options: model options
        """

        self.options = options

        # Dataset Type
        if isinstance(dataset_type, DatasetType):
            self.dataset_type = dataset_type
        else:
            raise TypeError('dataset_type has to be one of the DatasetType enum values')

        # Questions file (VQA json file)
        questions_path = ModelOptions.get_questions_path(options,dataset_type)
        if os.path.isfile(questions_path):
            self.questions_path = questions_path
        else:
            raise ValueError('The file ' + questions_path + ' does not exist')

        # VQA dataset type (e.g. v2)
        self.dataset = options['dataset']
        self.val_test_split = options['val_test_split']
        
        # Complementary pairs path (VQA json file); only v2 dataset has pairs data
        self.pairs_path = ModelOptions.get_pairs_path(options, dataset_type)
        if self.dataset == 'v2' and not os.path.isfile(self.pairs_path):
            raise ValueError('Using v2 dataset, but file ' + self.pairs_path + ' does not exist')
        
        # Images path (directory below which all images are stored)
        images_path = ModelOptions.get_images_path(options,dataset_type)
        if os.path.isdir(images_path):
            self.images_path = images_path
        else:
            raise ValueError('The directory ' + images_path + ' does not exist')

        # Image features path (hdf5 file generated by create_image_embeddings.py)
        image_features_path = ModelOptions.get_images_embed_path(options,dataset_type)
        if os.path.isfile(image_features_path):
            self.image_features_path = image_features_path
        else:
            raise ValueError('The file ' + image_features_path + ' does not exist')

        # Answers file (VQA json file)
        answers_path = ModelOptions.get_annotations_path(options,dataset_type)
        if answers_path and (not os.path.isfile(answers_path)):
            raise ValueError('The file ' + answers_path + ' does not exist')
        elif (not answers_path) and (dataset_type != DatasetType.TEST):
            raise ValueError('Answers file needed for training')
        self.answers_path = answers_path

        # Number of answer classes
        self.n_answer_classes = self.options['n_answer_classes']
        
        # Sentence embedding params
        self.sent_init_type = self.options['sent_init_type']
        self.glove_matrix_path = self.options['glove_matrix_path']

        # Tokenizer path (pickle file previously generated in prepare() method)
        self.tokenizer_path = os.path.abspath(self.options['tokenizer_path'])
        # if directory doesn't exist, create it
        tokenizer_dir = os.path.dirname(self.tokenizer_path)
        if not os.path.isdir(tokenizer_dir):
            os.mkdir(tokenizer_dir)

        if self.dataset_type != DatasetType.TEST:
            print("Tokenizer path -> ", self.tokenizer_path)
            # If Tokenizer pickle file is older than dataset.py, delete Tokenizer and GloVe matrix
            dataset_py_path = os.path.abspath(inspect.stack()[0][1])
            if os.path.isfile(self.tokenizer_path) and \
            os.path.getmtime(self.tokenizer_path) < os.path.getmtime(dataset_py_path):
                to_delete = input('\nWARNING: Tokenizer is outdated.  Remove it (y/n)? ')
                if len(to_delete) > 0 and to_delete[:1] == 'y':
                    os.remove(self.tokenizer_path)
                    print('Tokenizer was outdated.  Removed ->', self.tokenizer_path)
                    os.remove(self.glove_matrix_path)
                    print('GloVe embedding matrix was outdated. Removed -> ', self.glove_matrix_path)
                else:
                    print('Continuing with pre-existing Tokenizer and GloVe embedding matrix.')
            
            # Load pre-trained Tokenizer if one exists
            if os.path.isfile(self.tokenizer_path):
                self.tokenizer = pickle.load(open(self.tokenizer_path, 'rb'))
                print("Loading existing Tokenizer and attaching to dataset...")
            # Create new Tokenizer, but it can't be used until it's trained in prepare() method
            else:
                print("Creating new (untrained) Tokenizer...")
                # TODO: determine if we need to set the oov_token param for the Tokenizer
                # NOTE: 0 is a reserved index that won't be assigned to any word.
                # NOTE: Tokenizer removes all punctuation, so contraction preprocessing isn't needed
                self.tokenizer = Tokenizer(num_words=None, lower=True)
        else:
            # for test set, it must be provided by training set, not loaded from disk
            self.tokenizer = None

        # If GloVe matrix pickle file is older than dataset.py, delete GloVe matrix
        if os.path.isfile(self.glove_matrix_path) and os.path.isfile(self.tokenizer_path) and \
        os.path.getmtime(self.glove_matrix_path) < os.path.getmtime(self.tokenizer_path):
            to_delete = input('\nWARNING: GloVe embedding matrix is outdated (older than Tokenizer). Remove it (y/n)? ')
            if len(to_delete) > 0 and to_delete[:1] == 'y':
                os.remove(self.glove_matrix_path)
                print('GloVe embedding matrix was outdated. Removed -> ', self.glove_matrix_path)
            else:
                print('Continuing with pre-existing GloVe embedding matrix.')

        # Check if max sentence length has been specified
        self.max_sentence_len = self.options.get('max_sentence_len', None)

        self.answer_one_hot_mapping = None

        if (dataset_type == DatasetType.TRAIN):
            self.max_sample_size = self.options['max_train_size']
        elif (dataset_type == DatasetType.VALIDATION):
            self.max_sample_size = self.options['max_val_size']
        elif (dataset_type == DatasetType.TEST):
            self.max_sample_size = self.options['max_test_size']
        else:
            self.max_sample_size = None

        # List with samples
        self.samples = []

        
    def prepare(self, answer_one_hot_mapping, tokenizer=None):
        """Prepares the dataset to be used.

        It will load all the questions and answers in memory and references to
        the images. It will also train a Tokenizer holding the word dictionary
        and both answers and questions will be tokenized and encoded using that
        Tokenizer.

        As a side result, this function also creates a one hot encoding index 
        for all the top (n_answer_classes) answer choices Any answer outside
        the scope of this is tagged with a special out of vocab index - index 0
        """

        # if this isn't a training dataset, the answer one hot indices are expected to be available
        if (self.dataset_type != DatasetType.TRAIN):
            assert(answer_one_hot_mapping != None)

        # Load Questions and Answers
        questions = self._create_questions_dict(self.questions_path)
        print('Questions dict created. Num entries: {}'.format(len(questions)))  
        
        # Add complementary pairs data (only exists for VQA v2 dataset)
        if self.dataset == 'v2':
            questions = self._set_question_complements(questions, self.pairs_path)

        answers = self._create_answers_dict(self.answers_path)
        print('Answers dict created. Num entries: {}'.format(len(answers)))
        
        # Load Image IDs from the VGGNet embeddings file
        image_ids = self._get_image_ids(self.image_features_path)
        images = self._create_images_dict(image_ids)
        print('Images dict created')

        # We only keep the n_answer_classes choices for answers as this
        # is formulated as a classification problem
        answers = self._encode_answers(answers, answer_one_hot_mapping)

        # Initialize Tokenizer and GloVe matrix
        if tokenizer is not None:
            self.tokenizer = tokenizer
            print('Using the Tokenizer that was provided by the training set.')
        if self.sent_init_type == 'glove':
            self._init_tokenizer(questions, answers, build_glove_matrix=True)
            print('Tokenizer trained and GloVe matrix built...')
        else:
            self._init_tokenizer(questions, answers)
            print('Tokenizer trained...')
        
        max_len = 0  # To compute the maximum question length
        # Tokenize and encode questions and answers
        example = 0
        for _, question in questions.items():
            if example < 5:
                print('Sample question string to tokenize: ', question.question_str)
                print('- corresponding token sequence: ', question.tokenize(self.tokenizer))
            else:
                question.tokenize(self.tokenizer)
            example += 1
            # Get the maximum question length
            if question.get_tokens_length() > max_len:
                max_len = question.get_tokens_length()
        print('Questions tokenized...')

        # If the question max len has not been set in options file, assign to the
        # maximum question length in the dataset
        if not self.max_sentence_len:
            self.max_sentence_len = max_len
        print('Actual max sentence length: {}'.format(max_len))
        print('Model uses max sentence length: {}'.format(self.max_sentence_len))
        
        for _, answer in answers.items():
            answer.tokenize(self.tokenizer)
        print('Answers tokenized...')

        self._create_samples(images, questions, answers)

        print('\nSample Questions -> Answers')
        print('---------------------------')
        _, ques_strings, _, _, _, ans_strings, _, _, _ = self.get_qa_lists()
        for q, a in zip(ques_strings[:20], ans_strings[:20]):
            print('{} -> {}'.format(q, a))
        
    def _encode_answers(self,answers,answer_one_hot_mapping):
        """
           keep top [n_answer_classes] most common answers to reduce the number of answer classes
        """

        if answer_one_hot_mapping == None:
            # build the one-hot-encoding
            answer_counts = Counter() 
            for answer in answers.values():
                answer_counts[answer.answer_str] += 1

            sorted_answers = answer_counts.most_common(self.n_answer_classes) 

            print("Top 100 answers are ", sorted_answers[:100])
            
            # one slot is reserved for out of vocabulary words
            top_k = self.n_answer_classes - 1

            print('Top {} answers by word count:'.format(top_k))
            top_k_num_words = Counter()
            for s in sorted_answers[:top_k]:
                top_k_num_words[len(s[0].split())] += 1
            for word_count, num_answers in top_k_num_words.items():
                print('- {} of the Top {} answers have {} words'.format(num_answers, top_k, word_count))

            # Note that sorted_answers are ("answer text", count) tuples
            # we store the index as that would be used for the one hot encoding
            # index starts at 1 for valid training words. index 0 is reserved for out of vocab words seen in validation/test
            self.answer_one_hot_mapping = {answer_tuple[0]:one_hot_idx+1 
                                           for one_hot_idx, answer_tuple in enumerate(sorted_answers[:top_k])}
            self.answer_one_hot_mapping['<unk>'] = 0

        else:
            self.answer_one_hot_mapping = answer_one_hot_mapping

        print("Length of one hot mapping vector",len(self.answer_one_hot_mapping))
    
        # apply the one-hot encoding and save into answer objects
        for answer_id, answer in answers.items():
            one_hot_index = self.answer_one_hot_mapping.get(answer.answer_str, 0)  # 0 = OOV answer string
            answers[answer_id].one_hot_index = one_hot_index

        return answers    

    def _pad_samples(self,num_samples,batch_size):
        """
        If the number of samples are not an exact multiple of batch_size, then replicate
        some randomly picked elements so as to make it an exact multiple.
        We need to re-sort the sample array by image index so that the memory management code works.
        """

        num_extra = (num_samples // batch_size +1) * batch_size - num_samples
        print("Padding samples with {} elements so that it is an exact multiple of batch size {}".format(num_extra,batch_size))

        # the first argument is treated as a range. so picks num_extra integers from range [0..num_samples-1]
        indices = np.random.choice(num_samples,num_extra,replace=False)

        # add extra samples
        for idx in indices:
            sample = self.samples[idx]
            if (sample.sample_type == DatasetType.TEST):
                self.samples.append(VQASample(sample.question, sample.image, dataset_type=sample.sample_type))
            else:
                self.samples.append(VQASample(sample.question, sample.image, sample.answer, sample.sample_type))
     
        num_samples = len(self.samples)
        assert(len(self.samples) % batch_size == 0)

        ## re-sort the samples by image index

        self.samples = sorted(self.samples, key=lambda sample: sample.image.features_idx) 
        
        return num_samples 
        
    def _create_sample_chunks(self,num_samples,batch_size):

        """
           We assume num_samples is an exact multiple of batch_size as they should have been padded earlier
           Chunks are created so the memory can be managed in chunk units
           The chunk_dict has a tuple of (batch_start, batch_end) indices for each batch

           creates the self.chunk_dict
        """

        batches_per_chunk = 200 

        # num_samples should be exactly divisible by batch_size
        self.num_chunks = math.ceil((num_samples // batch_size) /  batches_per_chunk)

        assert(self.num_chunks >= 1)
        
        self.chunk_dict = {}
        end_sample_idx = 0

        print("Num Chunks -> ",self.num_chunks)
        print("Batch_size -> ",batch_size)
        

        for chunk_num in range(self.num_chunks):
            start_sample_idx = end_sample_idx
            end_sample_idx =  min(start_sample_idx + batch_size*batches_per_chunk, num_samples)
            self.chunk_dict[chunk_num] = (start_sample_idx,end_sample_idx - 1)
            print(" {} Chunk {}, start-index : {}, end index : {}, Memory idx {} to {}"
                  .format(self.dataset_type,chunk_num,start_sample_idx,end_sample_idx - 1,
                          self.samples[start_sample_idx].image.features_idx, self.samples[end_sample_idx-1].image.features_idx))


    def _load_batch_images(self,current_chunk_idx):

        """
           Make sure that images are loaded in chunk sizes
           Each time a chunk is loaded we can also free up memory from the last chunk.
           The sample list is treated as a circular array as the batches are generated in a loop.
           Possible values of current_chunk_indx is [0..(num_chunks -1 )]

           Each chunk_dict stores tuples of sample indices (begin_sample_indx, end_sample_indx) for samples in the chunk

           returns the next chunk index to the caller.

        """
        load_mem = False
        free_mem = False

        next_chunk_idx = (current_chunk_idx + 1 ) % self.num_chunks
        prev_chunk_idx = (current_chunk_idx - 1 ) % self.num_chunks
 
        # find the indices for the start and end samples in this chunk
        start_sample_idx = self.chunk_dict[current_chunk_idx][0]
        end_sample_idx   = self.chunk_dict[current_chunk_idx][1]

        # find corresponding image indices
        start_image_idx = self.samples[start_sample_idx].image.features_idx
        end_image_idx   = self.samples[end_sample_idx].image.features_idx


        ## go through all chunks and free memory for chunks whose image_index for last sample is 
        ## less than the start_image_idx for current chunk

        chunks_to_clean = []
        for chunk_idx, sample_indices in self.chunk_dict.items():

            # Samples from these chunks might still be in the job queue
            if (chunk_idx == current_chunk_idx): continue
            if (chunk_idx == next_chunk_idx): continue
            if (chunk_idx == prev_chunk_idx): continue

            #  if this chunks highest image index is lower than chunk being worked on
            if (self.samples[sample_indices[1]].image.features_idx < start_image_idx):
                chunks_to_clean.append(chunk_idx)
            # if this chunks lowest index is higher than current chunks highest index    
            if (self.samples[sample_indices[0]].image.features_idx > end_image_idx ):
                chunks_to_clean.append(chunk_idx)

        for chunk_idx in chunks_to_clean:
            #print("Freeing image indices from {} to {} for samples from {} to {}"
            #     .format(self.samples[self.chunk_dict[chunk_idx][0]].image.features_idx,self.samples[self.chunk_dict[chunk_idx][1]].image.features_idx,
            #             self.chunk_dict[chunk_idx][0],self.chunk_dict[chunk_idx][1]))
            for sample_idx in range(self.chunk_dict[chunk_idx][0],self.chunk_dict[chunk_idx][1] + 1):
                self.samples[sample_idx].image.reset() 

        ## read images from disk and allocate memory for images in this chunk

        if ((np.shape(self.samples[start_sample_idx].image.features)[0] == 0) or
             (np.shape(self.samples[end_sample_idx].image.features)[0] == 0)):

            print("\t loading {} images from index {} to {} for samples {} to {}"
                          .format(end_image_idx - start_image_idx, start_image_idx, 
                                  end_image_idx,start_sample_idx,end_sample_idx))

            with h5py.File(self.image_features_path,"r") as f:
                image_cache = f['embeddings'][start_image_idx:end_image_idx+1]

            for sample_idx in range(start_sample_idx,end_sample_idx + 1):
                self.samples[sample_idx].image.load(image_cache,offset = start_image_idx)

        # samples inside the chunks can be shuffled as long as the 1st and last elements are preserved for index comparison
        #randomize indices excluding the first and last
        if self.dataset_type != DatasetType.TEST: 
           # Above code prevents shuffle for the test mode as they get mis-aligned with the true labels
           shuffle_list = self.samples[start_sample_idx + 1: end_sample_idx]
           np.random.shuffle(shuffle_list)
           self.samples[start_sample_idx + 1 : end_sample_idx] =  shuffle_list
            
        return next_chunk_idx

    def batch_generator(self, text_only=False, img_only=False):
        """
          Yields a batch of data of size batch_size
          Assumes the samples are sorted by their image indices , see the prepare() function
          We step through the same sequence as images stored in the hdf5 file
          In doing so we can prevent the large memory footprint needed to load all the images in memory
          
          NOTE: text_only and img_only implementations are inefficient, mainly used for debugging.  All logic for
          processing images into batches still happens (same code path), except the images are not yielded
        """

        if text_only and img_only:
            raise ValueError('A batch cannot be both text-only and image-only')
            
        batch_size = self.options['batch_size']
        assert(self.max_sample_size != None and self.max_sample_size <= len(self.samples))

        num_samples = self.max_sample_size

        #Pad Samples if needed to be an exact multiple of batch_size
        if (num_samples % batch_size) != 0:
            if (num_samples // batch_size + 1)*batch_size <= len(self.samples):
                # increase num_samples to be an exact multiple
                num_samples = (num_samples // batch_size + 1) * batch_size
            else:
                num_samples = self._pad_samples(num_samples,batch_size)

        self._create_sample_chunks(num_samples,batch_size)

        batch_start = 0
        batch_end = batch_size
        current_chunk_idx = 0

        n_image_regions = self.options['n_image_regions']
        n_image_embed   = self.options['n_image_embed']


        print("Total Sample size -> ", num_samples)

        while True:

            # if we have reached the end of the current chunk, load the images for the next chunk
            # while freeing memory of prev to previous chunk
            if (batch_start ==  self.chunk_dict[current_chunk_idx][0]):
                #print("Batch Start, {}, Current chunk {}, chunk_start {}".format(batch_start,current_chunk_idx, self.chunk_dict[current_chunk_idx]))
                current_chunk_idx = self._load_batch_images(current_chunk_idx)
            
            # Initialize matrix
            I = np.zeros((batch_size,n_image_regions,n_image_embed), dtype=np.float32)
            Q = np.zeros((batch_size, self.max_sentence_len), dtype=np.int32)


            if self.dataset_type != DatasetType.TEST:
                A = np.zeros((batch_size, self.n_answer_classes), dtype=np.bool_)

                # randomize order of samples within a batch
                batch_indices = [i for i in range(batch_start,batch_end)]
                randomized_indices = np.random.choice(batch_indices,len(batch_indices),replace=False)
                for idx,sample_idx in enumerate(randomized_indices):
                    I[idx], Q[idx] = self.samples[sample_idx].get_input(self.max_sentence_len)
                    A[idx] = self.samples[sample_idx].get_output()  # the answer's one_hot_index

                # yield (output) appropriate batches of data
                if text_only:
                    yield([Q], A)
                elif img_only:
                    yield([I], A)
                else:
                    yield([I, Q], A)
            else:
                # in test mode, we should not randomize within the batch
                for idx in range(batch_size):
                    I[idx], Q[idx] = self.samples[batch_start + idx].get_input(self.max_sentence_len)

                if text_only:
                    yield (Q) 
                elif img_only:
                    yield (I)
                else:
                    yield ([I, Q])
                    
            # Update interval
            batch_start += batch_size
            # An epoch has finished
            if batch_start >= num_samples:
                batch_start = 0
                
            batch_end = batch_start + batch_size
            if batch_end > num_samples:
                batch_end = num_samples

    def size(self):
        """Returns the size (number of examples) of the dataset"""

        return len(self.samples)

        # Add complementary_question_id

    def _create_questions_dict(self, questions_json_path):
        """Create a dictionary of Question objects containing the information of the questions from the .json file.

        Args:
            questions_json_path (str): path to the JSON file with the questions

        Returns:
            A dictionary of Question instances with their id as a key
        """

        print('Loading VQA question data from ->', questions_json_path)
        questions_json = json.load(open(questions_json_path))
        questions = {question['question_id']: Question(question['question_id'],
                                                       process_sentence(question['question']),
                                                       question['image_id'])
                     for question in questions_json['questions']}
        return questions

    def _set_question_complements(self, questions, pairs_json_path):
        """
            Add `complementary_questions` attribute to quesetions that have one
            NOTE: this is only relevant for the VQA v2 dataset

        Args:
            questions (dict of Question instances)
            pairs_json_path (str): path to the JSON file defining the pairs
            
        Returns:
            queestions (updated dict of Question instances)
        """

        print('Loading VQA complementary pair data from ->', pairs_json_path)
        
        # load json into a dict
        pairs = dict(json.load(open(pairs_json_path)))
        # mapping is symmetrical, but *most* pairs are not listed both ways
        inverted_pairs = dict([[v,k] for k,v in pairs.items()])
        pairs.update(inverted_pairs)

        # iterate through questions dict and update complements
        for question in questions:
            if question in pairs:
                questions[question].complement_id = pairs[question]
        print('Complements added to questions')
        
        return questions
        
    def _create_answers_dict(self, answers_json_path):
        """
        Create a dictionary of Answer objects containing the information of the answers from the .json file.

        Args:
            answers_json_path (str): path to the JSON file with the answers

        Returns:
            A dictionary of Answer instances with a composed unique id as key
        """

        # There are no answers in the test dataset
        if self.dataset_type == DatasetType.TEST and not self.val_test_split:
            return {}

        print('Loading VQA answers data from ->', answers_json_path)
        answers_json = json.load(open(answers_json_path))

        # Please note that answer_id's for the 10 human rater answers are not unique across all the answers.
        # They are only unique for the particular question. However, we are only using these values for 
        # post-analysis, so we only record the corresponding list of 10 strings.  Note that the dataset also includes
        # a self-reported "confidence" (yes/maybe/no) for each one; we are not currently using that information.

        # keep the official label (`multiple_choice_answer`) and also all 10 human ratings
        answers = {}
        example = 0
        for annotation in answers_json['annotations']:
            rater_annotations = []
            for rater_responses in annotation['answers']:
                rater_annotations.append(process_answer(rater_responses.get('answer', None)))
            if example < 5:
                print('Sample rater annotations:', rater_annotations)
                example += 1 
            next_answer = Answer(answer_id=annotation['question_id'] * 10,
                                 answer_str=process_answer(annotation['multiple_choice_answer']),
                                 question_id=annotation['question_id'],
                                 image_id=annotation['image_id'],
                                 question_type=annotation['question_type'],
                                 answer_type=annotation['answer_type'],
                                 annotations=rater_annotations,
                                 n_answer_classes=self.n_answer_classes)
            answers[annotation['question_id'] * 10] = next_answer

        return answers

    def _create_images_dict(self, image_ids):
        """
            Creates and returns a dict allowing for the lookup of an Image object given an image_id.
        """
        
        images = {image_id: Image(image_id, features_idx)
                  for image_id, features_idx in image_ids.items()}

        return images

    def _create_samples(self, images, questions, answers ):
        """
        Fills the list of samples with VQASample instances given questions and answers dictionary.

        If dataset_type is DatasetType.TEST, answers will be ignored.
        """

        # Check for DatasetType
        answers_built = True
        if self.dataset_type == DatasetType.TEST and not self.val_test_split:
            answers_built = False

        if answers_built:
            print("Creating Samples with Images, Questions and Answers")
            for answer_id, answer in answers.items():
                question = questions[answer.question_id]
                image_id = question.image_id
                image = images[image_id]
                self.samples.append(VQASample(question, image, answer, \
                                              self.dataset_type, \
                                              val_test_split=self.options['val_test_split']))
        else:
            print("Creating Samples with Image and Questions ")
            for question_id, question in questions.items():
                image_id = question.image_id
                image = images[image_id]
                self.samples.append(VQASample(question, image, dataset_type=self.dataset_type))

        # samples are sorted by image index to enable sequential disk IO
        print("Sorting samples by Image indices...")
        self.samples = sorted(self.samples, key=lambda sample: sample.image.features_idx) 

        # If we are in val_test split mode, the samples need to be divided among the
        # validation and test sets

        print("Size of dataSet {} before split: {}".format(self.dataset_type, len(self.samples)))

        if(self.val_test_split):
            if (self.dataset_type == DatasetType.VALIDATION):
                print('Using first half of validation set (for validation)')
                self.samples = self.samples[:len(self.samples)//2]
            elif (self.dataset_type == DatasetType.TEST):
                print('Using reserved second half of validation set (for test)')
                self.samples = self.samples[len(self.samples)//2:]

        if (self.max_sample_size == None):
            self.max_sample_size = len(self.samples)
            print('Used max_sample_size for dataset {} = {}'.format(self.dataset_type, self.max_sample_size))
      
        # build optional, alternate list-based representation
        self.qa_lists = defaultdict(list)
        for s in self.samples:
            # populate attributes of Question objects
            self.qa_lists['question_ids'].append(s.question.id)
            self.qa_lists['question_strings'].append(s.question.question_str)
            self.qa_lists['image_ids'].append(s.question.image_id)
            if self.dataset_type != DatasetType.TEST or self.options['val_test_split']:
                # populate attributes of Answer objects
                self.qa_lists['answer_ids'].append(s.answer.id)
                self.qa_lists['answer_strings'].append(s.answer.answer_str)
                self.qa_lists['question_types'].append(s.answer.question_type)
                self.qa_lists['answer_types'].append(s.answer.answer_type)
                self.qa_lists['answer_annotations'].append(s.answer.annotations) 
                self.qa_lists['one_hot_index'].append(s.answer.one_hot_index)
        
    def get_qa_lists(self):
        """
            Convenience method to return question and answer data as lists.  This is useful
            during post-processing when true answers (labels) need to be compared to predictions
        """

        ques_ids = self.qa_lists['question_ids']
        ques_strings = self.qa_lists['question_strings']
        ques_types = self.qa_lists['question_types']
        image_ids = self.qa_lists['image_ids']
        ans_ids = self.qa_lists['answer_ids']
        ans_strings = self.qa_lists['answer_strings']
        ans_types = self.qa_lists['answer_types']
        ans_annotations = self.qa_lists['answer_annotations']
        ans_ohe = self.qa_lists['one_hot_index']

        return ques_ids, ques_strings, ques_types, image_ids, ans_ids, ans_strings, ans_types, ans_annotations, ans_ohe
      
    def get_question_lists(self):
        """
            Return the subset of data lists corresponding to Question attributes
        """
        
        ques_ids, ques_strings, ques_types, image_ids, _, _, _, _, _ = self.get_qa_lists()
        return ques_ids, ques_strings, ques_types, image_ids
    
    def get_answer_lists(self, with_annotations=False):
        """
            Return the subset of data lists corresponding to Answer attributes
        """
        
        _, _, _, _, ans_ids, ans_strings, ans_types, ans_annotations, ans_ohe = self.get_qa_lists()
        return ans_ids, ans_strings, ans_types, ans_annotations
        
    def _init_tokenizer(self, questions, answers, build_glove_matrix=False):
        """Fits the tokenizer with the questions and answers and saves this tokenizer into a file for later use"""

        # contrary to the docs, `word_index` exists before training, so can't use `hasattr` check
        if len(self.tokenizer.word_index) == 0:
            # only have to train it once.
            print('Tokenizer is not yet trained.  Training now...')
            questions_list = [question.question_str for _, question in questions.items()]
            answers_list = [answer.answer_str for _, answer in answers.items()]
            self.tokenizer.fit_on_texts(questions_list + answers_list)

            # Save tokenizer object
            pickle.dump(self.tokenizer, open(self.tokenizer_path, 'wb'))  
            
        else:
            print('Trained Tokenizer is available. Using it...')

        # Calculate vocab size. NOTE: this is different than Yang's number
        self.word_index = self.tokenizer.word_index           # Keras word_index starts indexing at 1
        self.vocab_size = len(self.tokenizer.word_index) + 1  # +1 to account for <unk>
        print('Words in tokenizer index (incl. 1 for <unk>):', self.vocab_size)

        # it's possible that Tokenizer was originally created without GloVe embeddings,
        # so check if the file needs to be created
        if build_glove_matrix:
            self._init_glove()

    def _init_glove(self):
        """
            Builds a lookup matrix of GloVe embeddings and save to disk as a separate pickle file.
            It can't just be an attribute of the dataset because in test mode, it will need to be loaded
            separately.
        """
        
        glove_path = self.options['glove_path']  # input
        glove_matrix_path = self.options['glove_matrix_path']  # output
        
        print('Loading glove embeddings from ->', glove_path)
        print('Saving glove matrix to ->', glove_matrix_path)
        glove_index = pickle.load(open(glove_path, 'rb')) # dictionary, keyed by word (string)
        # Confirm embedding dimension by looking up "the"; if "the" isn't present the embeddings aren't valid
        embed_dim = self.options['n_sent_embed'] = len(glove_index['the'])
        print('Using {} dimensional GloVe embeddings'.format(embed_dim))
                
        # build glove_matrix containing embeddings, keyed by word id (from self.word_index)
        print('Buiding GloVe embedding matrix')
        glove_matrix = np.zeros((self.vocab_size, embed_dim)) # accounts for <unk>
        for word, i in self.word_index.items():               # loops (vocab_size - 1) times because <unk> isn't in index
            glove_vector = glove_index.get(word)
            if glove_vector is not None:
                # words not found in embedding index (incl. <unk>) will be all-zeros.
                glove_matrix[i] = glove_vector
        # Save glove_matrix pickle file (which will be loaded by the model)
        pickle.dump(glove_matrix, open(glove_matrix_path, 'wb'))  

        print('Generated and saved GloVe embedding lookup matrix.  Shape: {}'.format(glove_matrix.shape))
            
    def _get_image_ids(self, image_features_path):
        """
            Load image IDs from the hdf5 file containing the VGGNet embeddings.  Create and return
            a dict that allows us to return the index within the embeddings matrix for a given
            image ID key (where image ID is defined in the questions.json file).
        """
        
        print("Loading image ids from file ->", image_features_path)
        with h5py.File(image_features_path,"r") as f:
            image_ids = f['imageIds'][:]

        image_ids_dict = {}
        for idx, image_id in enumerate(image_ids):
            image_ids_dict[image_id] = idx

        return image_ids_dict
