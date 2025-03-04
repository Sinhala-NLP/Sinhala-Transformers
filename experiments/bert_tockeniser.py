from datasets import load_dataset
from tokenizers import Tokenizer, normalizers, trainers, models, pre_tokenizers
from transformers import BertConfig

dataset = load_dataset('sinhala-nlp/sinhala-7m-corpus', split='train')

tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))

tokenizer.normalizer = normalizers.NFKC()

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

num_valid_texts = sum(1 for text in dataset["text"] if text is not None)
print(f"Number of valid text entries: {num_valid_texts}")


def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        batch_texts = dataset[i: i + batch_size]["text"]
        batch_texts = [text for text in batch_texts if text is not None]
        if batch_texts: 
            yield batch_texts


trainer = trainers.WordPieceTrainer(
    vocab_size=64000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

tokenizer.train_from_iterator(batch_iterator(), trainer)

tokenizer.save("sinhala-bert-base/tokenizer.json")

config = BertConfig.from_pretrained("google-bert/bert-base-cased", vocab_size=64000)
config.save_pretrained("sinhala-bert-base")
