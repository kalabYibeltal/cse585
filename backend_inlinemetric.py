import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import psutil
import statistics
import sys
import os
from llama_cpp import Llama

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MAX_LENGTH = 50

if len(sys.argv) > 1:
    NUM_CORES = int(sys.argv[1])
else:
    NUM_CORES = psutil.cpu_count(logical=False)  # Default to physical cores
torch.set_num_threads(NUM_CORES)
torch.set_num_interop_threads(NUM_CORES)
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)

if len(sys.argv) > 2:
    BATCH_SIZE = int(sys.argv[2])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
# model = Llama(model_path="./llama3/Meta-Llama-3-8B/consolidated.00.pth", n_ctx=2048)

cpu_usage_overall = []
cpu_usage_per_core = []
memory_usage = []
latencies = []
start_time = 0
total_prompts_processed = 0

def log_resources():
    cpu_usage_individual = psutil.cpu_percent(interval=None, percpu=True)  # Get CPU usage per core
    overall_cpu_usage = psutil.cpu_percent(interval=None)  # Overall CPU usage

    print("Overall CPU", overall_cpu_usage, "CPU", cpu_usage_individual)
    
    # print(f"Overall CPU Usage: {overall_cpu_usage:.2f}%")
    # print(f"CPU Usage per core: {cpu_usage_individual}")
    return overall_cpu_usage, cpu_usage_individual

def generate_response(prompt, max_length=MAX_LENGTH):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def process_prompts(prompts):
    global total_prompts_processed, peak_memory_usage
    global latencies, cpu_usage_overall, cpu_usage_per_core
    
    start_time = time.time()
    responses = []
    overall_cpu, cpu = log_resources()
    
    
    for i in range(0, len(prompts), BATCH_SIZE):
        request_start_time = time.time()
        batch = prompts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=MAX_LENGTH, pad_token_id=tokenizer.eos_token_id)
        responses.extend([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
        latencies.append(time.time()-request_start_time)
    overall_cpu, cpu = log_resources()
    cpu_usage_overall.append(overall_cpu)
    cpu_usage_per_core.append(cpu)
        

    end_time = time.time()
    
    total_prompts_processed += len(prompts)
    return responses, end_time - start_time

# Example prompts
prompts = [
    "Answer directly. Who is better, Lionel Messi or Cristiano Ronaldo?",
    "In a galaxy far, far away",
    "Once upon a time in a land",
    "The greatest challenge of our time"
    # "As the sun set on the horizon",
    # "Deep in the heart of the forest",
    # "On a dark and stormy night",
    # "In the bustling streets of New York",
    # "The future of artificial intelligence lies in",
    # "Beneath the surface of the ocean"
]

def calculate_avg_cpu_utilization_over_time(cpu_data_array, threshold=5):
    total_utilization = 0
    valid_core_count = 0

    # Loop through each time snapshot (each list in the array)
    for snapshot in cpu_data_array:
        # Filter cores with at least the threshold utilization
        filtered_cores = [usage for usage in snapshot if usage >= threshold]

        # If cores meet the threshold, accumulate their utilization
        if filtered_cores:
            total_utilization += sum(filtered_cores)
            valid_core_count += len(filtered_cores)

    # Avoid division by zero if no cores meet the threshold
    if valid_core_count == 0:
        return 0.0, 0

    # Calculate the average utilization across all filtered cores
    avg_utilization = total_utilization / valid_core_count

    return avg_utilization

if __name__ == "__main__":
    print(f"Using {NUM_CORES} cores")


    # Process prompts
    print("\nBatched processing:")
    batch_responses, batch_time = process_prompts(prompts)
    print(f"Time taken: {batch_time:.2f} seconds")
    

    # Calculate metrics
    average_latency = sum(latencies) / len(latencies)
    throughput = total_prompts_processed / batch_time
    
    process = psutil.Process()  
    memory_info = process.memory_info()
    peak_memory_usage = memory_info.vms / (1024 * 1024)  # Peak virtual memory usage in MB (Virtual Memory Size)
    avg_overall_cpu_usage = sum(cpu_usage_overall) / len(cpu_usage_overall)

    # Print results
    print("\nResource Usage:")
    print(f"Average CPU Usage for each Core: {[sum(core)/len(core) for core in zip(*cpu_usage_per_core)]}")
    print(f"Average CPU Usage per Core: {calculate_avg_cpu_utilization_over_time(cpu_usage_per_core)}")
    print(f"Average overall CPU Usage: {avg_overall_cpu_usage:.2f}%")
    
    print(f"Peak Memory Usage: {peak_memory_usage:.2f}Mbs")

    print("\nPerformance Metrics:")
    print(f"Total Time: {batch_time:.2f} seconds")
    print(f"Average Latency: {average_latency:.4f} seconds/prompt")
    print(f"Throughput: {throughput:.2f} prompts/second")

    # print("\nComparison:")
    # print(f"Batched time: {batch_time:.2f} seconds")
