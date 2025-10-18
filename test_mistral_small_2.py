#from transformers import AutoModelForCausalLM, AutoTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration
import torch


# Load the model and tokenizer from Hugging Face
model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

tokenizer = MistralTokenizer.from_hf_hub(model_id)

model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Manual prompt formatting for Mistral instruct models
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

# Manually create the prompt in Mistral format
prompt = f"<s>[INST] {messages[0]['content']} [/INST]"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)