from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from transformers import RobertaConfig
import os

os.environ['HF_HOME'] = '/mnt/data/ranasint/'

# load dataset
dataset = load_dataset('sinhala-nlp/sinhala-7m-corpus', split='train')

# Instantiate tokenizer
tokenizer = ByteLevelBPETokenizer()

num_valid_texts = sum(1 for text in dataset["text"] if text is not None)
print(f"Number of valid text entries: {num_valid_texts}")


def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        batch_texts = dataset[i: i + batch_size]["text"]
        batch_texts = [text for text in batch_texts if text is not None]
        if batch_texts:
            yield batch_texts

# Customized training
tokenizer.train_from_iterator(batch_iterator(), vocab_size=64000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save("sinhala-roberta-base/tokenizer.json")

config = RobertaConfig.from_pretrained("FacebookAI/roberta-base", vocab_size=64000)
config.save_pretrained("sinhala-roberta-base")