import os
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 1. HARDWARE CONFIGURATION
# ==========================================
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    print(f" GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    dtype = torch.float32
    print(" GPU NOT FOUND. Running on CPU (slower).")

base_path = "offline_models"
embed_path = os.path.join(base_path, "embed_model")
llm_path = os.path.join(base_path, "llm_model")

if not os.path.exists(embed_path) or not os.path.exists(llm_path):
    raise FileNotFoundError("Run 'download_models.py' first to download models.")

# ==========================================
# 2. GENERATE & CLEAN DATA
# ==========================================
print("\nGenerating Synthetic Claims Data...")
np.random.seed(42)

def generate_claims(n=3000):
    specialties = ["Cardiology", "Orthopedics", "Neurology", "Dermatology", "Pediatrics"]
    diag = ["D1", "D2", "D3", "D4", "D5"]
    proc = ["P1", "P2", "P3", "P4", "P5"]
    states = ["CA", "TX", "NY", "FL", "WA"]

    df = pd.DataFrame({
        "claim_id": range(1, n + 1),
        "patient_id": np.random.randint(1000, 9000, n),
        "provider_id": np.random.randint(10, 200, n),
        "specialty": np.random.choice(specialties, n),
        "diagnosis_code": np.random.choice(diag, n),
        "procedure_code": np.random.choice(proc, n),
        "claim_amount": np.abs(np.random.normal(3000, 800, n)),
        "claim_date": pd.to_datetime(np.random.choice(pd.date_range("2025-01-01", "2025-10-31"), n)),
        "state": np.random.choice(states, n),
    })

    # Inject Fraud
    dup_rows = df.sample(30, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    high_idx = np.random.choice(df.index, 50, replace=False)
    df.loc[high_idx, "claim_amount"] = df.loc[high_idx, "claim_amount"] * 5
    
    bad_idx = np.random.choice(df.index, 50, replace=False)
    for i in bad_idx:
        spec = df.at[i, "specialty"]
        if spec == "Cardiology": df.at[i, "procedure_code"] = "P5"
        elif spec == "Dermatology": df.at[i, "procedure_code"] = "P3"
    
    null_idx = np.random.choice(df.index, 50, replace=False)
    df.loc[null_idx, "state"] = None
    df.loc[null_idx, "diagnosis_code"] = None

    return df

df = generate_claims()

# Clean & Flag Logic
df.columns = df.columns.str.lower()
df = df.drop_duplicates(subset=["claim_id"])
df["diagnosis_code"] = df["diagnosis_code"].fillna("UNKNOWN")
df["procedure_code"] = df["procedure_code"].fillna("UNKNOWN")
df["state"] = df["state"].fillna("UNKNOWN")

df["dup_flag"] = df.duplicated(["patient_id", "claim_amount", "claim_date"], keep=False)
stats = df.groupby("specialty")["claim_amount"].agg(["mean", "std"]).reset_index()
df = df.merge(stats, on="specialty", how="left")
df["zscore"] = (df["claim_amount"] - df["mean"]) / df["std"]
df["abnormal_flag"] = df["zscore"].abs() > 3

valid_pairs = {("Cardiology", "P1"), ("Cardiology", "P2"), ("Orthopedics", "P3"), ("Neurology", "P4"), ("Dermatology", "P1"), ("Pediatrics", "P2")}
def is_mismatch(row):
    if row["procedure_code"] == "UNKNOWN": return False
    return (row["specialty"], row["procedure_code"]) not in valid_pairs

df["mismatch_flag"] = df.apply(is_mismatch, axis=1)
df["missing_flag"] = (df["state"] == "UNKNOWN") | (df["diagnosis_code"] == "UNKNOWN")
df["fraud_flag"] = df[["dup_flag", "abnormal_flag", "missing_flag", "mismatch_flag"]].any(axis=1)

def fraud_reasons(row):
    out = []
    if row["dup_flag"]: out.append("Duplicate Billing")
    if row["abnormal_flag"]: out.append("Abnormal Amount")
    if row["missing_flag"]: out.append("Missing Data")
    if row["mismatch_flag"]: out.append("Invalid Mapping")
    return ", ".join(out) if out else "Valid"

df["fraud_reason"] = df.apply(fraud_reasons, axis=1)
print(f"Data Processing Complete. Total Claims: {len(df)}")

# ==========================================
# 3. LOAD LOCAL AI MODELS
# ==========================================
print("\nLoading AI Models from 'offline_models' folder...")
embed_model = SentenceTransformer(embed_path)

docs = []
for _, r in df.iterrows():
    status = "FRAUD" if r["fraud_flag"] else "VALID"
    text = f"Claim ID: {r['claim_id']}, Specialty: {r['specialty']}, Amount: ${r['claim_amount']:.2f}, Status: {status}, Reason: {r['fraud_reason']}"
    docs.append(text)

embeddings = embed_model.encode(docs, show_progress_bar=True)
nn = NearestNeighbors(n_neighbors=5, metric="cosine")
nn.fit(embeddings)

print(f"Loading TinyLlama to {device}...")
pipe = pipeline(
    "text-generation",
    model=llm_path,
    tokenizer=llm_path,
    model_kwargs={"torch_dtype": dtype},
    device_map="auto",
    max_new_tokens=512
)
print(" AI System Ready.")

# ==========================================
# 4. QUERY FUNCTION (UPDATED)
# ==========================================

def ask_fraud_analyst(user_query):
    print("\n" + "="*80)
    print(f" QUESTION: {user_query}")
    print("-" * 80)

    # 1. Retrieve Data
    q_vec = embed_model.encode([user_query])
    distances, indices = nn.kneighbors(q_vec, n_neighbors=5) # Fetch top 5
    
    # --- NEW: RETRIEVE RAW DATAPOINTS ---
    # We grab the actual rows from the pandas dataframe using the indices found
    retrieved_rows = df.iloc[indices[0]].copy()
    
    # Print the datapoints clearly
    print(" RETRIEVED DATAPOINTS (EVIDENCE):")
    cols_to_show = ["claim_id", "specialty", "claim_amount", "fraud_reason", "fraud_flag"]
    print(retrieved_rows[cols_to_show].to_string(index=False))
    print("-" * 80)
    # ------------------------------------

    # 2. Build Prompt for LLM
    # We convert the retrieved rows into a text block for the LLM to read
    context_str = ""
    for idx in indices[0]:
        context_str += "- " + docs[idx] + "\n"

    prompt = f"""<|system|>
You are a Fraud Analyst Assistant. 
1. Use the Retrieved Datapoints provided below to answer.
2. If the claims are valid, say so. If they are fraud, explain why.
3. Be professional and concise.

RETRIEVED DATAPOINTS:
{context_str}
</s>
<|user|>
{user_query}
</s>
<|assistant|>
"""

    # 3. Generate Answer
    outputs = pipe(
        prompt, 
        max_new_tokens=512,
        do_sample=True, 
        temperature=0.4,
        top_k=50, 
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    response = outputs[0]['generated_text'].split("<|assistant|>")[-1].strip()
    
    return(f" ANALYST REPORT:\n{response}")
    # print("="*80 + "\n")


# ==========================================
# 5. RUN TEST QUERIES
# ==========================================
if __name__ == "__main__":
    ask_fraud_analyst("Identify valid claims.")
    ask_fraud_analyst("Show me some valid Pediatrics claims.")
    ask_fraud_analyst("Do we have any duplicate bills?")