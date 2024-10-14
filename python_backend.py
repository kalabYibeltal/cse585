import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# llama imports
#import llama_cpp import Llama

# Load model and tokenizer
model_name = "gpt2"  # Using a smaller model for demonstration


# llama model
#llama_model = llama_cpp.Llama(model_path="path/to/your/llama/model.bin", n_ctx=2048)



tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure we're using CPU
device = torch.device("cpu")
model = model.to(device)

def generate_response(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def process_sequential(prompts):
    start_time = time.time()
    responses = [generate_response(prompt) for prompt in prompts]
    end_time = time.time()
    return responses, end_time - start_time

def process_batched(prompts, batch_size=4):
    start_time = time.time()
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50, pad_token_id=tokenizer.eos_token_id)
        responses.extend([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
    end_time = time.time()
    return responses, end_time - start_time


# Llama processing
# def process_llama_cpp(prompts):
#     start_time = time.time()
#     responses = [generate_llama_response(prompt) for prompt in prompts]
#     end_time = time.time()
#     return responses, end_time - start_time

# def generate_llama_response(prompt, max_length=50):
#     output = model(prompt, max_tokens=max_length)
#     return output['choices'][0]['text']

# Example usage
prompts = [
    "Answer directly. Who is better, Lionel Messi or Cristiano Ronaldo?",
    "In a galaxy far, far away",
    "Once upon a time in a land",
    "The greatest challenge of our time",
    "As the sun set on the horizon",
    "Deep in the heart of the forest",
    "On a dark and stormy night",
    "In the bustling streets of New York"
]

print("Sequential processing:")
seq_responses, seq_time = process_sequential(prompts)
print(f"Time taken: {seq_time:.2f} seconds")

# print("*************************")
# for response in seq_responses:
#     print(response)
# print("*************************")


print("\nBatched processing:")
batch_responses, batch_time = process_batched(prompts)
print(f"Time taken: {batch_time:.2f} seconds")

# print("..................")
# for response in batch_responses:
#     print(response)
# print("..................")

# Llama processing
# print("\nllama.cpp processing:")
# llama_responses, llama_time = process_llama_cpp(prompts)
# print(f"Time taken: {llama_time:.2f} seconds")

print("\nComparison:")
print(f"Sequential time: {seq_time:.2f} seconds")
print(f"Batched time: {batch_time:.2f} seconds")
print(f"Speedup: {seq_time / batch_time:.2f}x")
# print(f"llama.cpp time: {llama_time:.2f} seconds")
# print(f"Speedup (llama.cpp vs Sequential): {seq_time / llama_time:.2f}x")
# print(f"Speedup (llama.cpp vs Batched): {batch_time / llama_time:.2f}x")