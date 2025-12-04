import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Create directory
os.makedirs("models/embed_model", exist_ok=True)
os.makedirs("models/llm_model", exist_ok=True)

print("1. Downloading Embedding Model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embed_model.save("models/embed_model")
print("Embedding model saved locally.")

print("2. Downloading LLM (TinyLlama)...")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# We load model and tokenizer separately to ensure clean saving
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.save_pretrained("models/llm_model")
model.save_pretrained("models/llm_model")
print("LLM saved locally.")