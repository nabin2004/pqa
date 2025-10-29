import pandas as pd
import json
import ast
import os

# === INPUT / OUTPUT ===
input_path = "data/combined_with_image_paths.csv"  # can also be .parquet
output_path = "data/scienceqa_nepali_train.jsonl"
os.makedirs("data", exist_ok=True)

# === LOAD DATA ===
if input_path.endswith(".parquet"):
    df = pd.read_parquet(input_path)
else:
    df = pd.read_csv(input_path)

print(f"Loaded {len(df)} samples")

# === CONVERT ===
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        image = str(row.get("image", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()

        # Skip if core fields missing
        if not image or not question or not answer:
            continue

        # Parse choices safely
        choices_text = ""
        raw_choices = row.get("choices", None)
        if isinstance(raw_choices, str):
            try:
                parsed = ast.literal_eval(raw_choices)
                if isinstance(parsed, (list, tuple)):
                    choices_text = "\n" + "\n".join([f"{i+1}. {c}" for i, c in enumerate(parsed)])
            except Exception:
                pass

        # Build full question prompt
        question_full = question + choices_text

        # Combine context fields
        context_parts = []
        for field_name, label in [
            ("hint", "Hint"),
            ("lecture", "Lecture"),
            ("solution", "Solution"),
            ("skill", "Skill"),
        ]:
            val = row.get(field_name)
            if isinstance(val, str) and len(val.strip()) > 0:
                context_parts.append(f"{label}: {val.strip()}")

        # Metadata fields
        meta_parts = []
        for field_name, label in [
            ("grade", "Grade"),
            ("subject", "Subject"),
            ("topic", "Topic"),
            ("category", "Category"),
            ("task", "Task"),
        ]:
            val = row.get(field_name)
            if isinstance(val, str) and len(val.strip()) > 0:
                meta_parts.append(f"{label}: {val.strip()}")

        # Combine context
        context = ""
        if context_parts or meta_parts:
            context = "\n".join(meta_parts + [""] + context_parts)

        # Construct final JSONL entry
        item = {
            "image": image,
            "question": question_full.strip(),
            "context": context.strip(),
            "answer": answer.strip(),
        }

        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(df)} formatted examples to {output_path}")
