from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

model_path = "/mnt/data/ranasint/Projects/Sinhala-Transformers/sinhala-roberta-base/"

# Load tokenizer (it will detect whether to use BertTokenizer or RobertaTokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (automatically detects architecture)
model = AutoModelForMaskedLM.from_pretrained(model_path, from_flax=True)

# Create the fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Test sentence
sentence = "නිවේදනයක් නිකුත් කරමින් <mask> මාධ්‍ය කෙට්ඨාසය මේ බව දැනුම් දී තිබේ."

# Get predictions
predictions = fill_mask(sentence)

# Print top predictions
for idx, pred in enumerate(predictions):
    print(f"Rank {idx+1}: {pred['sequence']} (Score: {pred['score']:.4f})")