from datasets import load_dataset
from tokenizers import Tokenizer, normalizers, trainers, models, pre_tokenizers

# Load dataset (first 10,000 samples)
dataset = load_dataset('sinhala-nlp/sinhala-7m-corpus', split='train')

# Initialize a WordPiece tokenizer
tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))

# Set a normalizer (ensures correct Unicode handling)
tokenizer.normalizer = normalizers.NFKC()

# Use a whitespace pre-tokenizer to avoid byte-level issues
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Define a function to yield text batches
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        batch_texts = dataset[i: i + batch_size]["text"]
        # Filter out None values
        batch_texts = [text for text in batch_texts if text is not None]
        if batch_texts:  # Only yield non-empty batches
            yield batch_texts


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
