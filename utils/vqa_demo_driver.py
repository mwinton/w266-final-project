# Interface for accessing the VQA dataset.
#
# Adapted slightly from the official VQA Python API and Evaluation code at:
# (https://github.com/GT-Vision-Lab/VQA)
#
# Per their notes, that code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link: 
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA              - VQA class that loads VQA annotation file and prepares data structures.
#  get_question_ids - Get question ids that satisfy given filter conditions.
#  get_image_ids    - Get image ids that satisfy given filter conditions.
#  load_qa          - Load questions and answers with the specified question ids.
#  show_qa          - Display the specified questions and answers.
#  load_results     - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime
import copy

class VQA:
    def __init__(self, annotation_file=None, question_file=None, pairs_file=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.

        Args:
            annotation_file: a string specifying the path to the VQA annotation file
        """
        
        # load dataset
        self.dataset = {}
        self.questions = {}      
        self.qa = {}
        self.qqa = {}
        self.image_to_qa = {}
        self.pairs = {}          # pairs of complementary question ids
        
        if not annotation_file == None and not question_file == None:
            print ('loading VQA annotations and questions into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            questions = json.load(open(question_file, 'r'))
            print (datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.questions = questions
            self.create_index()
            
            # if we have balanced pairs data, load it
            if not pairs_file == None:
                pairs = json.load(open(pairs_file, 'r'))
                for pair in pairs:
                    self.pairs[pair[0]] = pair[1]
                    # TODO: implement better way to deal with reverse pairs so dict isn't double length
#                     self.pairs[pair[1]] = pair[0]
                print('pairs data loaded!')

    def create_index(self):
        """
        Method to build index from an already-loaded dataset.
        """
        
        # create index
        print ('creating index...')
        image_to_qa = {ann['image_id']: [] for ann in self.dataset['annotations']}
        qa =  {ann['question_id']: [] for ann in self.dataset['annotations']}
        qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            image_to_qa[ann['image_id']] += [ann]
            qa[ann['question_id']] = ann
        for ques in self.questions['questions']:
              qqa[ques['question_id']] = ques
        print ('index created!')

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.image_to_qa = image_to_qa

    def info(self):
        """
        Print information about the VQA annotation file.
        """
        
        for key, value in self.datset['info'].items():
            print ('%s: %s'%(key, value))

    def get_question_ids(self, image_ids=[], question_types=[], answer_types=[]):
        """
        Get question ids that satisfy given filter conditions; default skips that filter.
        
        Args:
            image_ids: integer list of image ids to return question ids for
            question_types: string list of question types to return question ids for
            answer_types: string list of answer types to return question ids for
        
        Returns:
            An integer list of question ids
        """
        
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        question_types = question_types if type(question_types) == list else [question_types]
        answer_types = answer_types if type(answer_types) == list else [answer_types]

        if len(image_ids) == len(question_types) == len(answer_types) == 0:
            annotations = self.dataset['annotations']
        else:
            if not len(image_ids) == 0:
                annotations = sum([self.image_to_qa[iid] for iid in image_ids if iid in self.image_to_qa],[])
            else:
                annotations = self.dataset['annotations']
            annotations = annotations if len(question_types) == 0 \
                else [ann for ann in annotations if ann['question_type'] in question_types]
            annotations = annotations if len(answer_types)  == 0 \
                else [ann for ann in annotations if ann['answer_type'] in answer_types]
        ids = [ann['question_id'] for ann in annotations]
        return ids

    def get_complementary_pairs(self, question_ids=[], question_types=[], answer_types=[]):
        """
        Get the question ids that satisfy given filter conditions; default skips that filter.
        
        Args:
            question_ids: integer list of question ids to return image ids for
            question_types: string list of question types to return image ids for
            answer_types: string list of answer types to return image ids for
            
        Returns:
            A dict containing appropriate pairs of question IDs as key-value pairs. Callers
            will also need to look at the reverse mapping value-key.
        """
        
        question_ids = question_ids if type(question_ids) == list else [question_ids]
        question_types = question_types if type(question_types) == list else [question_types]
        answer_types = answer_types if type(answer_types) == list else [answer_types]

        if len(question_ids) == len(question_types) == len(answer_types) == 0:
            pairs = self.pairs
#         else:
            # TODO: implement filtering by question_ids, question_types, answer_types
        return pairs

    def get_image_ids(self, question_ids=[], question_types=[], answer_types=[]):
        """
        Get image ids that satisfy given filter conditions; default skips that filter.
        
        Args:
            question_ids: integer list of question ids to return image ids for
            question_types: string list of question types to return image ids for
            answer_types: string list of answer types to return image ids for
            
        Returns:
            An integer list of image ids
        """
        
        question_ids = question_ids if type(question_ids) == list else [question_ids]
        question_types = question_types if type(question_types) == list else [question_types]
        answer_types = answer_types if type(answer_types) == list else [answer_types]

        if len(question_ids) == len(question_types) == len(answer_types) == 0:
            annotations = self.dataset['annotations']
        else:
            if not len(question_ids) == 0:
                annotations = sum([self.qa[qid] for qid in question_ids if qid in self.qa],[])
            else:
                annotations = self.dataset['annotations']
            annotations = annotations if len(question_types) == 0 \
                else [ann for ann in annotations if ann['question_type'] in question_types]
            annotations = annotations if len(answer_types)  == 0 \
                else [ann for ann in annotations if ann['answer_type'] in answer_types]
        ids = [ann['image_id'] for ann in annotations]
        return ids

    def load_qa(self, question_ids=[]):
        """
        Load questions and answers with the specified question ids.
        
        Args:
            question_ids: integer list of question ids
            
        Returns:
            A list of QA dicts
        """

        if type(question_ids) == list:
            return [self.qa[qid] for qid in question_ids]
        elif type(question_ids) == int:
            return [self.qa[question_ids]]

    def show_qa(self, annotations):
        """
        Display the specified annotations.
        
        Args:
            annotations: list of dicts containing annotations
        """
        
        if len(annotations) == 0:
            return 0
        for ann in annotations:
            qid = ann['question_id']
            label = ann['multiple_choice_answer']
            print ("Question ({}): {}".format(qid, self.qqa[qid]['question']))
            print ('Ground truth label: {}'.format(label))
            for ans in ann['answers']:
                print ("Answer %d: %s" % (ans['answer_id'], ans['answer']))
        
    def load_results(self, result_file, question_file):
        """
        Load result file and return a result object.
        
        Args:
            result_file: string containing result file name
            question_file: string containing question file name
        
        Returns:
            VQA object containing results
        """
        
        res = VQA()
        res.questions = json.load(open(question_file))
        res.dataset['info'] = copy.deepcopy(self.questions['info'])
        res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
        res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
        res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
        res.dataset['license'] = copy.deepcopy(self.questions['license'])

        print ('Loading and preparing results...')
        time_t = datetime.datetime.utcnow()
        annotations    = json.load(open(result_file))
        assert type(annotations) == list, 'results is not an list of objects'
        
        ann_qids = [ann['question_id'] for ann in annotations]
        assert set(ann_qids) == set(self.get_question_ids()), \
        'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is at least one question id that does not belong to the question ids in the annotation file.'

        for ann in annotations:
            qid = ann['question_id']
            if res.dataset['task_type'] == 'Multiple Choice':
                assert ann['answer'] in self.qqa[qid]['multiple_choices'], 'predicted answer is not one of the multiple choices'
            qa_ann = self.qa[qid]
            ann['image_id'] = qa_ann['image_id'] 
            ann['question_type'] = qa_ann['question_type']
            ann['answer_type'] = qa_ann['answer_type']
        print ('DONE (t=%0.2fs)'% ((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = annotations
        res.create_index()
        return res
