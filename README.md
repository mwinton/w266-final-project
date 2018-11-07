# w266-final-project

Final VQA project for UC Berkeley NLP class (Fall 2018)

> Note that datasets are not included in this repo.


## VQA demo notebook

The [`vqa_demo.ipynb`](utils/vqa_demo.ipynb) notebook demonstrates the format of the dataset by previewing a few examples of image, question, and the answers provided by humans. It uses the [`vqa_demo_driver.py`](utils/vqa_demo_driver.py) utility function to load and parse the data.

In order to run this demo notebook, you must first download and extract the various components of the [VQA 2.0 "balanced" dataset](http://visualqa.org/download.html) and the corresponding COCO images linked from that page.  Downloading the images took me ~1.5 hours from a Google Cloud Compute Engine instance.

## Stacked Attention Network model

First, download the appropriate datasets, and extract components into the path structure described below.

* [VQA v1 dataset](http://visualqa.org/vqa_v1_download.html)
* [VQA v2 "balanced" dataset](http://visualqa.org/download.html)

In the /home/<userName>/vqa_data the following directory structure is expected:
.
├── annotations
│   ├── mscoco_train2014_annotations.json
│   ├── mscoco_val2014_annotations.json
│   ├── v2_mscoco_train2014_annotations.json
│   ├── v2_mscoco_val2014_annotations.json
├── images
│   └── mscoco
    	├── embeddings
        ├── test2015
    	├── train2014
    	└── val2014
├── pairs
│   ├── v2_mscoco_train2014_complementary_pairs.json
│   └── v2_mscoco_val2014_complementary_pairs.json
├── questions
│   ├── MultipleChoice_mscoco_test2015_questions.json
│   ├── MultipleChoice_mscoco_test-dev2015_questions.json
│   ├── MultipleChoice_mscoco_train2014_questions.json
│   ├── MultipleChoice_mscoco_val2014_questions.json
│   ├── OpenEnded_mscoco_test2015_questions.json
│   ├── OpenEnded_mscoco_test-dev2015_questions.json
│   ├── OpenEnded_mscoco_train2014_questions.json
│   ├── OpenEnded_mscoco_val2014_questions.json
│   ├── v2_OpenEnded_mscoco_test2015_questions.json
│   ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
├── questiontypes
    ├── abstract_v002_question_types.txt
    └── mscoco_question_types.txt


### Training the model:

All runs need to be launched from the ./bin directory

1. Train the SAN model on all the train/val samples. runtime will be quite high (~3hrs per epoch on Nvidia K80)

   > python3 ./visualqa.py --verbose --model SAN  

2. (Optional) Train the SAN model with smaller train/val set

   > python3 ./visualqa.py --verbose --model SAN  --max_train_size 20000 -max_val_size 10000

