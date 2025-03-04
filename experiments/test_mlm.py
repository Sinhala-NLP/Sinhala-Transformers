from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

model_path = "NLPC-UOM/SinBERT-small"

# Load tokenizer (it will detect whether to use BertTokenizer or RobertaTokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (automatically detects architecture)
model = AutoModelForMaskedLM.from_pretrained(model_path)

# Create the fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Test sentence
sentence = "මෙම නියෝජිතයින් රජය අපහසුතාවට පත් කිරීමේ අරමුණින් ඉන්ධන බෙදා හැරීමේ කටයුතු අඩපණ කිරීමේ <mask> නියැලෙස බව සඳහන් කරමින් අපරාධ පරීක්ෂණ දෙපාර්තමේන්තුවට පැමිණිල්ලක් ලැබී ඇත."

# Get predictions
predictions = fill_mask(sentence)

# Print top predictions
for idx, pred in enumerate(predictions):
    print(f"Rank {idx+1}: {pred['sequence']} (Score: {pred['score']:.4f})")