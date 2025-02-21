from datasets import load_dataset
from tokenizers import Tokenizer, normalizers, trainers, models, pre_tokenizers

# Load dataset
dataset = load_dataset('sinhala-nlp/sinhala-7m-corpus', split='train')

# Instantiate a BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Set normalizer (optional)
tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])

# Pre-tokenizer (splits text into words before BPE)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Training function using dataset iterator
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]

# Define trainer
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Train tokenizer
tokenizer.train_from_iterator(batch_iterator(), trainer)

# Save tokenizer
tokenizer.save("sinhala-roberta-base/tokenizer.json")
