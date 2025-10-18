#from transformers import AutoModelForCausalLM, AutoTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration
import torch


# Load the model and tokenizer from Hugging Face
model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

#tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer = MistralTokenizer.from_hf_hub(model_id)

#model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto") # device_map="auto" uses your GPU if available

model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)


# Format your prompt in the correct chat template
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
