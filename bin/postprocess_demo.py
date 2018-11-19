from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
import sys

# need to append parent directory to path to import from peer directories
sys.path.append('..')
from vqa.dataset.dataset import VQADataset

# load results from model.predict()
dataset_path = '/home/mwinton/w266-final-project/results/y_pred/y_proba_san_expt0_2018-11-18-23:50:13.p'
with open(dataset_path, 'rb') as f:
    y_pred = pickle.load(f)
print('\nShape of retrieved y_pred:', y_pred.shape)
# print('\nSample y_pred softmax probabilities:', y_pred[:20])

# determine predicted label
y_argmax = np.argmax(y_pred, axis=1)
# print('First 20 predicted label indices (argmax):', y_argmax[:20])

# load tokenizer to find corresponding words
tokenizer_path = '/home/mwinton/w266-final-project/data/preprocessed/tokenizer.p'
tokenizer = pickle.load(open(tokenizer_path, 'rb'))
word_to_index = tokenizer.word_index
index_to_word = dict([[v,k] for k, v in word_to_index.items()])
# manually add <unk>
index_to_word[0] = '<unk>'

# determine corresponding text labels for predictions
y_pred_labels = [index_to_word[id] for id in y_argmax]
# print('\First 20 corresponding predicted words:', y_pred_labels[:20])

# load validation dataset
val_path =  '/home/mwinton/w266-final-project/data/preprocessed/validate_dataset.p'
val_dataset = pickle.load(open(val_path, 'rb'))

# NOTE: these don't seem to be aligned with the y_pred results
samples = val_dataset.samples
print('dataset length (should match):', len(samples))

# get lists of the various data elements for convenience
_, ques_strings, ques_types, image_ids = val_dataset.get_question_lists()
_, ans_strings, ans_types, _ = val_dataset.get_answer_lists()

print('\nSamples from the dataset')
print('Prediction | Question | Answer | Q Type | A Type')
example = 0
for pred, qst, ast, qty, aty in zip(y_pred_labels, ques_strings, ans_strings, ques_types, ans_types):
    if example == 20: break
    print('{} | {} | {} | {} | {}'.format(pred, qst, ast, qty, aty))
    example += 1

