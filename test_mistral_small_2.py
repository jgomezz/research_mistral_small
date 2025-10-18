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

tokenized = tokenizer.encode_chat_completion(messages)     
input_ids = torch.tensor([tokenized.tokens])
attention_mask = torch.ones_like(input_ids)
# Generate a response from the model
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,              
    max_new_tokens=100,
)[0]
# Decode and print the model's response
decoded_output = tokenizer.decode(output[len(tokenized.tokens) :])
print(decoded_output)
# Example output: "The capital of France is Paris."
#from transformers import AutoModelForCausalLM, AutoTokenizer   
#from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
#from transformers import Mistral3ForConditionalGeneration
#import torch   