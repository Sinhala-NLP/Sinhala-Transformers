from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

model_path = "/mnt/data/ranasint/Projects/Sinhala-Transformers/sinhala-roberta-base/"

# Load tokenizer (it will detect whether to use BertTokenizer or RobertaTokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (automatically detects architecture)
model = AutoModelForMaskedLM.from_pretrained(model_path, from_flax=True)

# Create the fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Test sentence
sentence = "සංවිධානාත්මක අපරාධකරුවෙකු වන සංජීව කුමාර සමරරත්න නොහොත් ගණේමුල්ල සංජීව වෙඩි තබා ඝාතනය කිරීමේ අපරාධයේ සැඟව සිටින ප්‍රධාන සැකකාරිය සම්බන්ධයෙන් <mask> තොරතුරක් ලබා දෙන අයෙකුට රුපියල් ලක්ෂ 12ක මුදල් ත්‍යාගයක් ලබාදෙන බව පොලිස් මූලස්ථානය පවසයි."

# Get predictions
predictions = fill_mask(sentence)

# Print top predictions
for idx, pred in enumerate(predictions):
    print(f"Rank {idx+1}: {pred['sequence']} (Score: {pred['score']:.4f})")