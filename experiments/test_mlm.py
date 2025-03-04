from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

model_path = "NLPC-UOM/SinBERT-small"

# Load tokenizer (it will detect whether to use BertTokenizer or RobertaTokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (automatically detects architecture)
model = AutoModelForMaskedLM.from_pretrained(model_path)

# Create the fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Test sentence
sentence = "එම ඝාතනය සම්බන්ධයෙන් කොළඹ අපරාධ කොට්ඨාසය විසින් <mask> සිදුකරමින් පවතී."

# Get predictions
predictions = fill_mask(sentence)

# Print top predictions
for idx, pred in enumerate(predictions):
    print(f"Rank {idx+1}: {pred['sequence']} (Score: {pred['score']:.4f})")