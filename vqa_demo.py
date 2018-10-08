#
# Adapted slightly from the official VQA Python API and Evaluation code at:
# (https://github.com/GT-Vision-Lab/VQA)
#

from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

# configuration constants
DATA_PATH     ='../../VQA'
VERSION_PREFIX ='v2_' # this should be '' when using VQA v2.0 dataset
TASK_TYPE ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
DATA_TYPE ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
DATA_SUBTYPE ='train2014'
ANNOTATION_FILE ='%s/annotations/%s%s_%s_annotations.json'%(DATA_PATH, VERSION_PREFIX, DATA_TYPE, DATA_SUBTYPE)
QUESTION_FILE ='%s/questions/%s%s_%s_%s_questions.json'%(DATA_PATH, VERSION_PREFIX, TASK_TYPE, DATA_TYPE, DATA_SUBTYPE)
IMAGE_PATH = '%s/images/%s/%s/' %(DATA_PATH, DATA_TYPE, DATA_SUBTYPE)

# initialize VQA api for QA annotations
vqa=VQA(ANNOTATION_FILE, QUESTION_FILE)

# load and display QA annotations for given question types
"""
All possible question_types for abstract and mscoco has been provided in respective text files in ../QuestionTypes/ folder.
"""
annotation_ids = vqa.get_question_ids(question_types='how many');   
annotations = vqa.load_qa(annotation_ids)
random_annotation = random.choice(annotations)
vqa.show_qa([random_annotation])
iid = random_annotation['image_id']
image_filename = 'COCO_' + DATA_SUBTYPE + '_'+ str(iid).zfill(12) + '.jpg'
if os.path.isfile(IMAGE_PATH + image_filename):
    image = io.imread(IMAGE_PATH + image_filename)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# load and display QA annotations for given answer types
"""
answer_types can be one of the following
yes/no
number
other
"""
annotation_ids = vqa.get_question_ids(answer_types='yes/no');   
annotations = vqa.load_qa(annotation_ids)
random_annotation = random.choice(annotations)
vqa.show_qa([random_annotation])
iid = random_annotation['image_id']
image_filename = 'COCO_' + DATA_SUBTYPE + '_'+ str(iid).zfill(12) + '.jpg'
if os.path.isfile(IMAGE_PATH + image_filename):
    image = io.imread(IMAGE_PATH + image_filename)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# load and display QA annotations for given images
"""
Usage: vqa.get_image_ids(question_ids=[], question_types=[], answer_types=[])
Above method can be used to retrieve image ids for given question ids or given question types or given answer types.
"""
ids = vqa.get_image_ids()
annotation_ids = vqa.get_question_ids(iids=random.sample(ids,5));  
annotations = vqa.load_qa(annotation_ids)
random_annotation = random.choice(annotations)
vqa.show_qa([random_annotation])  
iid = random_annotation['image_id']
image_filename = 'COCO_' + DATA_SUBTYPE + '_'+ str(iid).zfill(12) + '.jpg'
if os.path.isfile(IMAGE_PATH + image_filename):
    I = io.imread(IMAGE_PATH + image_filename)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

