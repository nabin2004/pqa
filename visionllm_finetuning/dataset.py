# dataset.py
import os
import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset

def load_examples(jsonl_path: str, imageid_to_idx: dict):
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            # Make sure all required keys exist
            if not all(k in obj for k in ("image", "question", "answer", "context")):
                continue

            img_name = os.path.basename(obj["image"])
            if img_name not in imageid_to_idx:
                continue

            examples.append({
                "image_id": img_name,
                "question": obj["question"],
                "answer": obj["answer"],
                "context": obj["context"]
            })
    return examples

class VisionPrefixDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict],
        embeddings: torch.Tensor,
        imageid_to_idx: dict,
        tokenizer,
        max_text_len: int = 256,
        max_label_len: int = 128,
    ):
        self.examples = examples
        self.emb = embeddings  # CPU tensor
        self.imageid_to_idx = imageid_to_idx
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_label_len = max_label_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        emb_idx = self.imageid_to_idx[ex["image_id"]]
        vision_emb = self.emb[emb_idx]
        
        system_prompt = (
            "You are a teaching assistant. Given a student's question and an image, "
            "always respond in Nepali in a friendly and explanatory tone. "
            "Provide the answer and context in the following JSON format, without adding any extra text:\n"
            '{\n  "answer": "<the answer in Nepali>",\n'
            '  "context": "<the explanation or context in Nepali>"\n}\n\n'
        )

        # system_prompt = (
        #     "तिमी एक शिक्षक सहायक हौ। विद्यार्थीको प्रश्न र छवि दिइएको छ भने, "
        #     "कृपया सधैं निम्न JSON ढाँचामा उत्तर दिनुहोस्:\n"
        #     '{\n  "answer": "<तपाईंको उत्तर>",\n  "context": "<सन्दर्भ/व्याख्या>"\n}\n\n'
        # )


    
        # Input prompt: only question
        prompt = system_prompt + f"Question: {ex['question']}"
    
        # Target text: answer + context
        target = f"Answer: {ex['answer']}\nContext: {ex['context']}"
    
        # Tokenize input
        enc_input = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt"
        )
    
        # Tokenize target
        enc_target = self.tokenizer(
            target,
            truncation=True,
            padding="max_length",
            max_length=self.max_label_len,
            return_tensors="pt"
        )
    
        labels = enc_target["input_ids"].clone()
        labels[enc_target["attention_mask"] == 0] = -100  # mask padding
    
        return {
            "vision_emb": vision_emb,
            "input_ids": enc_input["input_ids"].squeeze(0),
            "attention_mask": enc_input["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
