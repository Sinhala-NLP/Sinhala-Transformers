from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

# model_path = "/mnt/data/ranasint/Projects/Sinhala-Transformers/sinhala-roberta-base/"
model_path = "NLPC-UOM/SinBERT-large"

# Load tokenizer (it will detect whether to use BertTokenizer or RobertaTokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (automatically detects architecture)
# model = AutoModelForMaskedLM.from_pretrained(model_path, from_flax=True)
model = AutoModelForMaskedLM.from_pretrained(model_path)

# Create the fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Test sentence
sentence = "ශ්‍රී දළදා මාළිගාව යනු බුදුරජාණන් වහන්සේගේ <mask> දන්තධාතූන් වහන්සේ වර්තමානයේ තැන්පත් කර ඇති මාළිගාවයි. "

# Get predictions
predictions = fill_mask(sentence)

# Print top predictions
for idx, pred in enumerate(predictions):
    print(f"Rank {idx+1}: {pred['sequence']} (Score: {pred['score']:.4f})")