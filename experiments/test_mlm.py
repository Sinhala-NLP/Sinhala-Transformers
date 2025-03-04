from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

model_path = "/mnt/data/ranasint/Projects/Sinhala-Transformers/sinhala-roberta-base/"

# Load tokenizer (it will detect whether to use BertTokenizer or RobertaTokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (automatically detects architecture)
model = AutoModelForMaskedLM.from_pretrained(model_path, from_flax=True)

# Create the fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Test sentence
sentence = "<mask> තැබීම සිදුකළ දිනයේ සිට මේ වන තෙක් ඇය සැඟව සිටින ස්ථානය සම්බන්ධයෙන් නිසි තොරතුරක් නොමැති බැවින් සැකකාරිය අත්අඩංගුවට ගැනීම සඳහා සාර්ථක තොරතුරක් ලබාදෙන අයෙකුට රුපියල් ලක්ෂ 12 ක මුදල් ත්‍යාගයක් පිරිනැමීමට පොලිස් මූලස්ථානය තීරණය කර ඇත"

# Get predictions
predictions = fill_mask(sentence)

# Print top predictions
for idx, pred in enumerate(predictions):
    print(f"Rank {idx+1}: {pred['sequence']} (Score: {pred['score']:.4f})")