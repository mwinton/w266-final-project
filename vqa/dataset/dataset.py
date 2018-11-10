## File to load the dataset and prepare batches for training 
## Adapted from https://github.com/imatge-upc/vqa-2016-cvprw, Issey Masuda Mora 
## 
## Added Memory management related code for Image Embeddings

import h5py
import json
import math
import numpy as np
import os
import pickle
import random
import scipy.io
import time

from collections import Counter

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

        # Questions file
        questions_path = ModelOptions.get_questions_path(options,dataset_type)
        if os.path.isfile(questions_path):
            self.questions_path = questions_path
        else:
            raise ValueError('The file ' + questions_path + ' does not exists')

        # Images path
        images_path = ModelOptions.get_images_path(options,dataset_type)
        if os.path.isdir(images_path):
            self.images_path = images_path
        else:
            raise ValueError('The directory ' + images_path + ' does not exists')

        # Features path
        features_path = ModelOptions.get_images_embed_path(options,dataset_type)
        if os.path.isfile(features_path):
            self.features_path = features_path
        else:
            raise ValueError('The file ' + features_path + ' does not exists')


        # Answers file
        answers_path = ModelOptions.get_annotations_path(options,dataset_type)
        if answers_path and (not os.path.isfile(answers_path)):
            raise ValueError('The directory ' + answers_path + ' does not exists')
        elif (not answers_path) and (dataset_type != DatasetType.TEST and dataset_type != DatasetType.EVAL):
            raise ValueError('Answers path needed to proceed further')
        self.answers_path = answers_path

        # Vocabulary size
        self.vocab_size = self.options['n_vocab']

        # Number of answer classes
        self.n_answer_classes = self.options['n_answer_classes']

        # Tokenizer path
        self.tokenizer_path = self.options['tokenizer_path']
        print("Tokenizer path -> ", self.tokenizer_path)
        tokenizer_dir = os.path.dirname(os.path.abspath(self.tokenizer_path))
        print("Tokenizer directory -> ", tokenizer_dir)
        if not os.path.isdir(tokenizer_dir):
            os.mkdir(tokenizer_dir)

        # Load pre-trained Tokenizer if one exists
        if os.path.isfile(self.tokenizer_path):
            self.tokenizer = pickle.load(open(self.tokenizer_path, 'rb'))
            print("Opening existing Tokenizer...")
        # Create new Tokenizer, but it can't be used until training
        else:
            print("Building tokenizer...")
            # TODO: determine if we need to set the oov_token param for the Tokenizer
            # NOTE: 0 is a reserved index that won't be assigned to any word.
            # NOTE: Tokenizer removes all punctuation, so contraction preprocessing isn't needed
            self.tokenizer = Tokenizer(num_words=self.vocab_size, lower=True)

        # Check if max sentence length has been specified
        self.max_sentence_len = self.options.get('max_sentence_len', None)

        self.answer_one_hot_mapping = None

        if (dataset_type == DatasetType.TRAIN):
            self.max_sample_size = self.options['max_train_size']
        elif (dataset_type == DatasetType.VALIDATION):
            self.max_sample_size = self.options['max_val_size']

        # List with samples
        self.samples = []

    def prepare(self,answer_one_hot_mapping):
        """Prepares the dataset to be used.

        It will load all the questions and answers in memory and references to the images. It will also train a
        Tokenizer holding the word dictionary and both answers and questions will be tokenized and encoded using that
        Tokenizer.

        As a side result, this function also creates a one hot encoding index for all the top (n_answer_classes) answer choices
        Any answer outside the scope of this is tagged with a special out of vocab index - index 0
        """

        # if this isn't a training dataset, the answer one hot indices are expected to be available
        if (self.dataset_type != DatasetType.TRAIN):
            assert(answer_one_hot_mapping != None)

        # Load Questions and Answers
        questions = self._create_questions_dict(self.questions_path)
        print('Questions dict created. Num entries: {}'.format(len(questions)))        
        answers = self._create_answers_dict(self.answers_path)
        print('Answers dict created. Num entries: {}'.format(len(answers)))
        
        # Load Image IDs
        image_ids = self._get_image_ids(self.features_path)
        images = self._create_images_dict(image_ids)
        print('Images dict created')

        # We only keep the n_answer_classes choices for answers as this
        # is formulated as a classification problem
        answers = self.encode_answers(answers, answer_one_hot_mapping)

        # Ensure we have a trained tokenizer with a dictionary
        self._init_tokenizer(questions, answers)

        aux_len = 0  # To compute the maximum question length
        # Tokenize and encode questions and answers
        debug = 0
        for _, question in questions.items():
            if debug < 5:
                print('Sample question string to tokenize: ', question.question_str)
                print('- corresponding token sequence: ', question.tokenize(self.tokenizer))
            else:
                question.tokenize(self.tokenizer)
            debug += 1
            # Get the maximum question length
            if question.get_tokens_length() > aux_len:
                aux_len = question.get_tokens_length()

        # If the question max len has not been set in options file, assign to the
        # maximum question length in the dataset
        if not self.max_sentence_len:
            self.max_sentence_len = aux_len
        print('Actual max sentence length: {}'.format(aux_len))
        print('Model uses max sentence length: {}'.format(self.max_sentence_len))
        
        for _, answer in answers.items():
            answer.tokenize(self.tokenizer)

        print('Tokenizer created')

        self._create_samples(images, questions, answers)

        print("Sorting Samples by Image indices ")
       
        self.samples = sorted(self.samples, key=lambda sample: sample.image.features_idx) 

        if (self.max_sample_size == None):
            self.max_sample_size = len(self.samples)


    def encode_answers(self,answers,answer_one_hot_mapping):
        """
           keep top [n_answer_classes]  most common answers to reduce the number of answer classes
        """

        if answer_one_hot_mapping == None:

            answer_counts = Counter() 
            for answer in answers.values():
                answer_counts[answer.answer_str] += 1

            sorted_answers = answer_counts.most_common(self.n_answer_classes) 

            print("Top 100 answers are ", sorted_answers[:100])
            
            # TODO: change to Top k instead of hard-coded top 1000
            print('Top 1000 answers by word count')
            top_k_num_words = Counter()
            for s in sorted_answers[:1000]:
                top_k_num_words[len(s[0].split())] += 1
            for word_count, num_answers in top_k_num_words.items():
                print('- {} of the Top 1000 answers have {} words'.format(num_answers, word_count))

            # one slot is reserved for out of vocabulary words
            top_k = self.n_answer_classes - 1

            # Note that sorted_answers are ("answer text",count) tuples
            # we store the index as that would be used for the one hot encoding
            # index starts at 1 for valid training words. index 0 is reserved for out of vocab words seen in validation/test
            self.answer_one_hot_mapping = {answer_tuple[0]:one_hot_idx+1 
                                           for one_hot_idx,answer_tuple in enumerate(sorted_answers[:top_k])}
            self.answer_one_hot_mapping['<unk>'] = 0

        else:
            self.answer_one_hot_mapping = answer_one_hot_mapping

        print("Length of one hot mapping vector",len(self.answer_one_hot_mapping))


        for answer_id, answer in answers.items():
            one_hot_index = self.answer_one_hot_mapping.get(answer.answer_str,0)
            answers[answer_id].one_hot_index = one_hot_index

        return answers    

    def pad_samples(self,num_samples,batch_size):
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
            self.samples.append(VQASample(sample.question, sample.image, sample.answer, sample.sample_type))
     
        num_samples = len(self.samples)
        assert(len(self.samples) % batch_size == 0)

        ## re-sort the samples by image index

        self.samples = sorted(self.samples, key=lambda sample: sample.image.features_idx) 
        
        return num_samples 
        
    def create_sample_chunks(self,num_samples,batch_size):

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


    def load_batch_images(self,current_chunk_idx):

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

            with h5py.File(self.features_path,"r") as f:
                image_cache = f['embeddings'][start_image_idx:end_image_idx+1]

            for sample_idx in range(start_sample_idx,end_sample_idx + 1):
                self.samples[sample_idx].image.load(image_cache,offset = start_image_idx)

        # samples inside the chunks can be shuffled as long as the 1st and last elements are preserved for index comparison
        #randomize indices exclusing the first and last
        shuffle_list = self.samples[start_sample_idx + 1: end_sample_idx]
        np.random.shuffle(shuffle_list)
        self.samples[start_sample_idx + 1 : end_sample_idx] =  shuffle_list
            
        return next_chunk_idx

    def batch_generator(self):
        """
          Yields a batch of data of size batch_size
          Assumes the samples are sorted by their image indices , see the prepare() function
          We step through the same sequence as images stored in the hdf5 file
          In doing so we can prevent the large memory footprint needed to load all the images in memory
        """

        batch_size = self.options['batch_size']
        assert(self.max_sample_size != None and self.max_sample_size <= len(self.samples))

        num_samples = self.max_sample_size

        #Pad Samples if needed to be an exact multiple of batch_size
        if (num_samples % batch_size) != 0:
            if (num_samples // batch_size + 1)*batch_size <= len(self.samples):
                # increase num_samples to be an exact multiple
                num_samples = (num_samples // batch_size + 1) * batch_size
            else:
                num_samples = self.pad_samples(num_samples,batch_size)

        self.create_sample_chunks(num_samples,batch_size)

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
                current_chunk_idx = self.load_batch_images(current_chunk_idx)
            
            # Initialize matrix
            I = np.zeros((batch_size,n_image_regions,n_image_embed), dtype=np.float32)
            Q = np.zeros((batch_size, self.max_sentence_len), dtype=np.int32)
            A = np.zeros((batch_size, self.n_answer_classes), dtype=np.bool_)

            # randomize order of samples within a batch
            batch_indices = [i for i in range(batch_start,batch_end)]
            randomized_indices = np.random.choice(batch_indices,len(batch_indices),replace=False)
            for idx,sample_idx in enumerate(randomized_indices):
                #if (np.shape(self.samples[sample_idx].image.features)[0] == 0):
                    #print("Error for batch ({}-{}), sample-idx {}, image {} is missing".format(
                    #       batch_start,batch_end,sample_idx,self.samples[sample_idx].image.features_idx))

                I[idx], Q[idx] = self.samples[sample_idx].get_input(self.max_sentence_len)
                A[idx] = self.samples[sample_idx].get_output()

            yield ([I, Q], A)

            # Update interval
            batch_start += batch_size
            # An epoch has finished
            if batch_start >= num_samples:
                batch_start = 0
                
            batch_end = batch_start + batch_size
            if batch_end > num_samples:
                batch_end = num_samples

    def get_dataset_input(self):
        # this is called in test model
        # Load all the images in memory in this mode. TODO, need to chunk this

        with h5py.File(self.features_path,"r") as f:
            features = f['embeddings'][:]

        for sample in self.samples:
            sample.image.load(features, True)
        images_list = []
        questions_list = []

        for sample in self.samples:
            images_list.append(sample.get_input(self.max_sentence_len)[0])
            questions_list.append(sample.get_input(self.max_sentence_len)[1])

        return np.array(images_list), np.array(questions_list)

    def get_dataset_output(self):
        output_array = [sample.get_output() for sample in self.samples]

        print('output_array list created')

        return np.array(output_array).astype(np.bool_)

    def size(self):
        """Returns the size (number of examples) of the dataset"""

        return len(self.samples)

    def _create_questions_dict(self, questions_json_path):
        """Create a dictionary of Question objects containing the information of the questions from the .json file.

        Args:
            questions_json_path (str): path to the JSON file with the questions

        Returns:
            A dictionary of Question instances with their id as a key
        """

        questions_json = json.load(open(questions_json_path))
        questions = {question['question_id']:
                         Question(question['question_id'], process_sentence(question['question']), question['image_id'],
                                  self.vocab_size)
                     for question in questions_json['questions']}
        return questions

    def _create_answers_dict(self, answers_json_path):
        """Create a dictionary of Answer objects containing the information of the answers from the .json file.

        Args:
            answers_json_path (str): path to the JSON file with the answers

        Returns:
            A dictionary of Answer instances with a composed unique id as key
        """

        # There are no answers in the test dataset
        if self.dataset_type == DatasetType.TEST or self.dataset_type == DatasetType.EVAL:
            return {}

        answers_json = json.load(open(answers_json_path))

        # Please note that answer_id's are not unique across all the answers. They are only unique
        # for the particular question. We generate unique id's for answers as described below:
        # (annotation['question_id'] * 10 + (answer['answer_id'] - 1): creates a unique answer id
        # of that question.
        # As question_id is composed by appending the question number (0-2) to the image_id (which is unique)
        # we've composed the answer id the same way. The substraction of 1 is due to the fact that the
        # answer['answer_id'] ranges from 1 to 10 instead of 0 to 9

       
        if (self.options["keep_single_answer"] == False): 
            answers = {(annotation['question_id'] * 10 + (answer['answer_id'] - 1)):
                       Answer(answer['answer_id'], process_answer(answer['answer']), annotation['question_id'],
                              annotation['image_id'], self.vocab_size, self.n_answer_classes)
                       for annotation in answers_json['annotations'] for answer in annotation['answers']}
        else:
            answers  = answers = {annotation['question_id'] * 10 :
                                  Answer(annotation['question_id'] * 10, process_answer(annotation['multiple_choice_answer']), 
                                  annotation['question_id'], annotation['image_id'], self.vocab_size, self.n_answer_classes)
                                  for annotation in answers_json['annotations'] }
        return answers

    def _create_images_dict(self, image_ids):
        images = {image_id: Image(image_id, features_idx) for image_id, features_idx in image_ids.items()}

        return images

    def _create_samples(self, images, questions, answers ):
        """Fills the list of samples with VQASample instances given questions and answers dictionary.

        If dataset_type is DatasetType.TEST, answers will be ignored.
        """

        # Check for DatasetType
        if self.dataset_type != DatasetType.TEST and self.dataset_type != DatasetType.EVAL:
            for answer_id, answer in answers.items():
                question = questions[answer.question_id]
                image_id = question.image_id
                image = images[image_id]
                self.samples.append(VQASample(question, image, answer, self.dataset_type))
        else:
            for question_id, question in questions.items():
                image_id = question.image_id
                image = images[image_id]
                self.samples.append(VQASample(question, image, dataset_type=self.dataset_type))

    def _init_tokenizer(self, questions, answers):
        """Fits the tokenizer with the questions and answers and saves this tokenizer into a file for later use"""

        print('Training tokenizer...')
        questions_list = [question.question_str for _, question in questions.items()]
        answers_list = [answer.answer_str for _, answer in answers.items()]

        print("Sample Questions : \n {}".format(questions_list[:10]))
        print ("\n*************\n")
        print("Sample Answers : \n {}".format(answers_list[:10]))

        self.tokenizer.fit_on_texts(questions_list + answers_list)
        print('Words in tokenizer index: ', len(self.tokenizer.word_index))
        # Save tokenizer object
        pickle.dump(self.tokenizer, open(self.tokenizer_path, 'wb'))

    def _get_image_ids(self, features_path):

        print("Accessing file =>",features_path)
        with h5py.File(features_path,"r") as f:
            image_ids = f['imageIds'][:]

        image_ids_dict = {}
        for idx, image_id in enumerate(image_ids):
            image_ids_dict[image_id] = idx

        return image_ids_dict


class MergeDataset:
    def __init__(self, train_dataset, val_dataset, percentage=0.7):
        if not isinstance(train_dataset, VQADataset):
            raise TypeError('train_dataset has to be an instance of VQADataset')

        if not isinstance(val_dataset, VQADataset):
            raise TypeError('val_dataset has to be an instance of VQADataset')

        self.percentage = percentage

        # Get parameters
        self.features_path = train_dataset.features_path
        self.max_sentence_len = train_dataset.max_sentence_len
        self.vocab_size = train_dataset.vocab_size

        # Split validation dataset to use some of it for training
        self.train_samples = train_dataset.samples
        self.val_samples = val_dataset.samples
        random.shuffle(self.val_samples)
        threshold = int(self.percentage * len(self.val_samples))
        self.train_samples = self.train_samples + self.val_samples[:threshold]
        self.val_samples = self.val_samples[threshold:]
        print('Training samples: {}'.format(len(self.train_samples)))
        print('Validation samples: {}'.format(len(self.val_samples)))

    def batch_generator(self, batch_size, split='train'):
        """Yields a batch of data of size batch_size"""

        # Load all the images in memory
        print('Loading visual features...')
        features = scipy.io.loadmat(self.features_path)['features']

        if split == 'train':
            samples = self.train_samples
        else:
            samples = self.val_samples

        for sample in samples:
            sample.image.load(features, 0)
        print('Visual features loaded')

        num_samples = len(samples)
        batch_start = 0
        batch_end = batch_size

        while True:
            # Initialize matrix
            I = np.zeros((batch_size, 1024), dtype=np.float16)
            Q = np.zeros((batch_size, self.max_sentence_len), dtype=np.int32)
            A = np.zeros((batch_size, self.vocab_size), dtype=np.bool_)
            # Assign each sample in the batch
            for idx, sample in enumerate(samples[batch_start:batch_end]):
                I[idx], Q[idx] = sample.get_input(self.max_sentence_len)
                A[idx] = sample.get_output()

            yield ([I, Q], A)

            # Update interval
            batch_start += batch_size
            # An epoch has finished
            if batch_start >= num_samples:
                batch_start = 0
                # Change the order so the model won't see the samples in the same order in the next epoch
                random.shuffle(samples)
            batch_end = batch_start + batch_size
            if batch_end > num_samples:
                batch_end = num_samples

    def train_size(self):
        return len(self.train_samples)

    def val_size(self):
        return len(self.val_samples)
