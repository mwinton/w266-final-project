# w266-final-project

Final VQA project for UC Berkeley NLP class (Fall 2018)

> Note that datasets are not included in this repo.


## VQA demo notebook

The [`vqa_demo.ipynb`](utils/vqa_demo.ipynb) notebook demonstrates the format of the dataset by previewing a few examples of image, question, and the answers provided by humans. It uses the [`vqa_demo_driver.py`](utils/vqa_demo_driver.py) utility function to load and parse the data.

In order to run this demo notebook, you must first download and extract the various components of the [VQA 2.0 "balanced" dataset](http://visualqa.org/download.html) and the corresponding COCO images linked from that page.  Downloading the images took me ~1.5 hours from a Google Cloud Compute Engine instance.

## Visual question answering project

### Overview of project structure:

These are some of the most noteworthy paths in the project:

- `/bin/visualqa.py`: main entry point to the code
- `/data/preprocessed/`: stored tokenizer and datasets (as pickle files)
- `/results/acc_curves/`: training/validation accuracy plots (as png files)
- `/results/loss_curves/`: training/validation loss plots (as png files)
- `/saved_models/json/`: Keras model descriptions (as json files)
- `/saved_models/weights/`: trained model weights (as hdf5 files)
- `/utils/vqa_demo.ipynb/`: demo notebook for exploring dataset structure
- `/utils/create_embeddings.py`: program used to generate image embeddings (e.g. from VGGNet)
- `/utils/create_glove_embeddings.py`: program used to generate GloVe embeddings "pickle" file from the original text
- `/vqa/dataset/dataset.py`: processes the original data (images and json files) into data structures
- `/vqa/dataset/sample.py`: class representation of a single data sample (used by dataset.py)
- `/vqa/experiments/`: location for json files that define experiments (overriding default parameters)
- `/vqa/model/`: location for python files representing various Keras models
- `/vqa/model/model_select.py`: file that loads appropriate Keras model from model library
- `/vqa/model/options.py`: default values for hyperparameters (can be overridden by json experiment files)
- `/vqa/model/*_model.py`: individual model files (e.g. `san_model.py` is the Stacked Attention Network)


### Downloading and extracting VQA dataset

First, download the appropriate datasets, and extract components into the path structure described below.

* [VQA v1 dataset](http://visualqa.org/vqa_v1_download.html)
* [VQA v2 "balanced" dataset](http://visualqa.org/download.html)

In the /home/&lt;username&gt;/vqa_data the following directory structure is expected:
```
.
├── annotations
│   ├── mscoco_train2014_annotations.json
│   ├── mscoco_val2014_annotations.json
│   ├── v2_mscoco_train2014_annotations.json
│   ├── v2_mscoco_val2014_annotations.json
├── images
│   └── mscoco
    	├── embeddings
            ├── vgg16
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
```

### Downloading and extracting GloVe embeddings

Copy the "pickled" GloVe embeddings to /home/&lt;username&gt;/glove/glove.p

### Enabling logging of hyperparameters and plots:

To enable [MLFlow](https://www.mlflow.org) logging, set the `MLFLOW_TRACKING_URI` environment variable in your `~/.bashrc` file:

```
export MLFLOW_TRACKING_URI="http://xxx.xxx.xxx.xxx:5000"
```

If the environment variable isn't set, then logging will be automatically disabled.

### Enabling TensorBoard:

TensorBoard looks for its logs in the /home/&lt;username&gt;/logs directory.

Assuming you have set up your SSH port mapping, you can view TensorBoard at [http://localhost:6006](http://localhost:6006).

### Training and testing the model:

All runs need to be launched from the ./bin directory.  Here are some examples of how to use the command line arguments:

1. The primary way to confirm a run is to load an experiment from a json file in `vqa/experiments/`

```
python3 visualqa.py --verbose --experiment 2 --epochs 10
```

2. Run the test set on the latest weights from a trained model:

```
python3 visualqa.py --verbose --model san  --action test
```

3. (Optional) Train the SAN model with a smaller train/val set:

```
python3 visualqa.py --verbose --model san  --max_train_size 20000 --max_val_size 10000 --epochs 2
```

4. (Optional) Train a (faster) text-only CNN model for debugging:

```
python3 visualqa.py --verbose --model text_cnn  --max_train_size 2000 --max_val_size 1000 --epochs 2
```

5. (Optional) Run an experiment with the VQA v2 dataset:

```
python3 visualqa.py --verbose --experiment 2 --epochs 10 --dataset v2
```

6. (Optional) Run default SAN model with `adam` optimizer instead of default `sgd`:

```
python3 visualqa.py --verbose --model san --optimizer adam --max_train_size 20000 --max_val_size 10000 -e 3
```

7. (Optional) Run prediction on validation set used during training, after training finishes:

```
python3 visualqa.py --verbose --model san --max_train_size 20000 --max_val_size 10000 -e 2 --predict_on_validation_set --max_test_size=1000
```

To see all command line arguments:

```
python3 visualqa.py --help
```
