from datasets import load_dataset
from tokenizers import Tokenizer, normalizers, trainers, models, pre_tokenizers

# Load dataset (first 10,000 samples)
dataset = load_dataset('sinhala-nlp/sinhala-7m-corpus', split='train[:10000]')

# Initialize a WordPiece tokenizer
tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))

# Set a normalizer (ensures correct Unicode handling)
tokenizer.normalizer = normalizers.NFKC()

# Use a whitespace pre-tokenizer to avoid byte-level issues
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Define a function to yield text batches
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]

# Define a trainer for WordPiece
trainer = trainers.WordPieceTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Train the tokenizer
tokenizer.train_from_iterator(batch_iterator(), trainer)

# Save tokenizer
tokenizer.save("sinhala-roberta-base/tokenizer.json")
