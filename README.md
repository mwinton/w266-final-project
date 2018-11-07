# w266-final-project

Final VQA project for UC Berkeley NLP class (Fall 2018)

> Note that datasets are not included in this repo.


## VQA demo notebook

The [`vqa_demo.ipynb`](vqa_demo.ipynb) notebook demonstrates the format of the dataset by previewing a few examples of image, question, and the answers provided by humans. It uses the [`vqa.py`](vqa.py) utility function to load and parse the data.

In order to run this demo notebook, you must first download and extract the various components of the [VQA 2.0 "balanced" dataset](http://visualqa.org/download.html) and the corresponding COCO images linked from that page.  Downloading the images took me ~1.5 hours from a Google Cloud Compute instance.

- The "annotations" files need to be extracted into `./annotations`.
- The "questions" files need to be extracted into `./questions`.
- The "question types" files need to be extracted into `./questiontypes`.
- The COCO images need to be extracted into `./images`.
- The "complementary pairs" files need to be extracted into `./pairs`. _NOTE: these aren't used in the demo notebook yet_


Sample runs:

All runs need to be launched from the ./bin directory

1. Train the SAN model on all the train/val samples. runtime will be quite high (~3hrs per epoch on Nvidia K80)

   > python3 ./visualqa.py --verbose --model SAN  

2. Train the SAN model with smaller train/val set

   > python3 ./visualqa.py --verbose --model SAN  --max_train_size 20000 -max_val_size 10000

