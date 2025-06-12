from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
from transformers import TextStreamer

# Load environment variables
load_dotenv()

# MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI"))


db = client["healthcare_dashboard"]
collection = db["sentiment"]

# Define request body
class InferenceRequest(BaseModel):
    input_text: str

# Initialize FastAPI app
app = FastAPI()

max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

train_model, train_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/content/drive/MyDrive/shin_colab/mistral_lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(train_model) # Enable native 2x faster inference


text_streamer = TextStreamer(train_tokenizer)

def input_prompt(input_text):
  prompt = f"""[INST]
You are an expert in analyzing sentiment from drug reviews. Your task is to classify a given drug review as **Positive, Negative, or Neutral** based on the user's experience.

### **Criteria for Sentiment Classification:**
- **Positive**: The review expresses satisfaction with the drug, mentioning effectiveness, relief from symptoms, or minimal side effects.
- **Negative**: The review expresses dissatisfaction, describing strong side effects, lack of effectiveness, or worsening of the condition.
- **Neutral**: The review is mixed or inconclusive, mentioning both pros and cons without strong emotions, or if the experience is unclear.

### **Only return one of the following labels: Positive, Negative, or Neutral. Do not provide explanations.

### **Examples:**

#### **Example 1:**
**Review:** "This medication completely stopped my migraines within a week. I haven't had a single headache since!"
**Sentiment:** Positive

#### **Example 2:**
**Review:** "I had high hopes, but this drug didn’t help my pain at all. On top of that, I had severe nausea."
**Sentiment:** Negative

#### **Example 3:**
**Review:** "The drug helped somewhat with my anxiety, but I still experience occasional panic attacks. No major side effects so far."
**Sentiment:** Neutral

---

### **Now classify the following review:**
**Review:** "{input_text}"
**Sentiment:**
[/INST]"""
  return prompt

def inference_model(input_text, model, tokenizer):
  prompt = input_prompt(input_text)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  model_inputs = encodeds.to(device)

  generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=False, temperature=0.5)

  decoded = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
  # print(decoded[0])

  # Get the output only instead of whole prompt
  output_text = decoded[0][len(prompt)-40:].strip().split("\n")[-1]


  return output_text.strip()

# Define API endpoint
@app.post("/predict")
def predict(request: InferenceRequest):
    result = inference_model(input_text=request.input_text, model=train_model, tokenizer=train_tokenizer)
    # Save request info to MongoDB
    record = {
        "input_text": request.input_text,
        "result": result,
        "timestamp": datetime.now()
    }
    collection.insert_one(record)
    return {"result": result}