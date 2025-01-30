from datasets import load_dataset
from pathlib import Path
from datasets import Dataset
import torch
import argparse
from random import shuffle
import os

from language_modeling.model_args import LanguageModelingArgs
from language_modeling.language_modeling_model import LanguageModelingModel

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--vocab_size', required=False, help='vocab size', default="32000")
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
VOCAB_SIZE = int(arguments.vocab_size)

# dataset = load_dataset("sinhala-nlp/sinhala-7m-corpus", column_names=['text'])
dataset = Dataset.to_pandas(load_dataset('sinhala-nlp/sinhala-7m-corpus', split='train'))
lines = dataset['text'].tolist()

lines = lines[:5000]
shuffle(lines)
train_lines = lines[:int(len(lines)*.8)]
test_lines = lines[int(len(lines)*.8):len(lines)]

lines = None
del lines

# Path(os.path.join("outputs", MODEL_TYPE)).mkdir(parents=True, exist_ok=True)

with open('train.txt', 'w', encoding='utf-8') as f:
    # write each integer to the file on a new line
    for line in train_lines:
        f.write(str(line) + '\n')

with open('test.txt', 'w', encoding='utf-8') as f:
    # write each integer to the file on a new line
    for line in test_lines:
        f.write(str(line) + '\n')



model_args = LanguageModelingArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 10
model_args.dataset_type = "simple"
model_args.train_batch_size = 32
model_args.eval_batch_size = 64
model_args.learning_rate = 1e-4
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 30000
model_args.save_eval_checkpoints = True
model_args.save_best_model = True
model_args.save_recent_only = True
model_args.wandb_project = "Sinhala Transformers"
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.vocab_size = VOCAB_SIZE

# model_args.output_dir = os.path.join("outputs", MODEL_TYPE)
# model_args.best_model_dir = os.path.join("outputs", MODEL_TYPE, "best_model")
# model_args.cache_dir = os.path.join("cache_dir", MODEL_TYPE)

train_file = "train.txt"
test_file = "test.txt"

model = LanguageModelingModel(
    MODEL_TYPE, None, args=model_args, train_files=train_file, use_cuda=torch.cuda.is_available()
)

# Train the model
model.train_model(train_file, eval_file=test_file, args=model_args)

# Evaluate the model
result = model.eval_model(test_file)