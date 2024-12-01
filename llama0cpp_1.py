import time
import threading
import psutil
import statistics
import sys
import os
from llama_cpp import Llama
import requests
import json

import asyncio
import aiohttp

# Configuration
MODEL_PATH = "../llama.cpp/Llama-3.2-3B-Instruct-uncensored-Q2_K.gguf"  # Adjust path as needed
MAX_LENGTH = 50
BATCH_SIZE = 2
NUM_CORES = 14

# if len(sys.argv) > 1:
#     NUM_CORES = int(sys.argv[1])
# else:
#     NUM_CORES = psutil.cpu_count(logical=False)

# if len(sys.argv) > 2:
#     BATCH_SIZE = int(sys.argv[2])

# Initialize model

import requests

# prompts = [
#     "What is AI?",
#     "Explain quantum computing.",
#     "what is 1 + 1",
#     "Tell me about space exploration.",
#     "who is obama",
# ]

    
    
    
model = Llama(
    model_path=MODEL_PATH,
    n_threads=NUM_CORES,
    n_ctx=2048,
    n_batch=BATCH_SIZE
)

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
    return overall_cpu_usage, cpu_usage_individual

def generate_response(prompt, max_length=MAX_LENGTH):
    
    output = model(
        prompt,
        max_tokens=max_length,
        echo=False,
        temperature=0.7
    )
    # print(output)
    return output['choices'][0]['text']


def process_prompts(prompts):
    global total_prompts_processed
    global latencies, cpu_usage_overall, cpu_usage_per_core
    
    start_time = time.time()
    responses = []
    overall_cpu, cpu = log_resources()
    
    for i in range(0, len(prompts), BATCH_SIZE):
        request_start_time = time.time()
        batch = prompts[i:i+BATCH_SIZE]
        
        # Process each prompt in the batch
        batch_responses = []
        for prompt in batch:
            response = generate_response(prompt)
            batch_responses.append(response)
            
        responses.extend(batch_responses)
        latencies.append(time.time() - request_start_time)
    
    overall_cpu, cpu = log_resources()
    cpu_usage_overall.append(overall_cpu)
    cpu_usage_per_core.append(cpu)
    
    end_time = time.time()
    total_prompts_processed += len(prompts)
    return responses, end_time - start_time

def calculate_avg_cpu_utilization_over_time(cpu_data_array, threshold=5):
    total_utilization = 0
    valid_core_count = 0

    for snapshot in cpu_data_array:
        filtered_cores = [usage for usage in snapshot if usage >= threshold]
        if filtered_cores:
            total_utilization += sum(filtered_cores)
            valid_core_count += len(filtered_cores)

    if valid_core_count == 0:
        return 0.0, 0

    avg_utilization = total_utilization / valid_core_count
    return avg_utilization

# Example prompts
with open('questions.txt', 'r') as file:
    line_counter = 0
    for line in file:
        if line_counter == 10:
            break
        prompts = (line.strip())
        line_counter += 1

print(len(prompts))
prompts[5]

def count_words(input_string):
    # Split the string by whitespace and count the resulting parts
    words = input_string.split()
    return len(words)

# # # Define the async function to send a POST request for each prompt
async def send_prompt(session, prompt):
    async with session.post("http://127.0.0.1:8080/completion", json={"prompt": prompt, "n_predict": 10}, timeout=60000  ) as response:
    #     data =   await response.json()
    #     return data["body"]
    
        # if response.headers.get('Content-Type', None) == 'application/json':
        data = await response.json()
        # else:
        # Handle non-JSON response (e.g., print raw text)
            # data = response
        return data

# Main async function to handle all requests concurrently
async def gen():
    async with aiohttp.ClientSession() as session:
        # Send all prompts concurrently using asyncio.gather
        tasks = [send_prompt(session, prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        # Print responses in order of completion
        for response in responses:
            # print(response['content'])
            data = response
            # print(data)
    return responses

# Run the async event loop


if __name__ == "__main__":
    
  
    
    print(f"Using {NUM_CORES} cores")

    # Process prompts
    print("\nBatched processing:")
    # batch_responses, batch_time = process_prompts(prompts)
    
    start_time = time.time()
    
    overall_cpu, cpu = log_resources()
        
    responses = asyncio.run(gen())
    
    
    
    overall_cpu, cpu = log_resources()
        
    
    end_time = time.time()
    
    overall_cpu, cpu = log_resources()
    
    batch_time = end_time - start_time
    
    token_count = 0
    for response in responses:
        answer = response['context']
        token_count = token_count + count_words(answer)
    
    
    
    print(f"Time taken: {batch_time:.2f} seconds")

    # Calculate metrics
    average_latency = batch_time / len(prompts)
    throughput = token_count / batch_time
    
    process = psutil.Process()  
    memory_info = process.memory_info()
    peak_memory_usage = memory_info.vms / (1024 * 1024)

    # # Print results
    # print("\nResource Usage:")
    print(f"CPU Usage for each Core: {cpu}")
    print(f"Average CPU Usage per Core: {calculate_avg_cpu_utilization_over_time(cpu_usage_per_core)}")
    print(f"Average overall CPU Usage: {overall_cpu:.2f}%")
    print(f"Peak Memory Usage: {peak_memory_usage:.2f}Mbs")

    # print("\nPerformance Metrics:")
    # print(f"Total Time: {batch_time:.2f} seconds")
    # print(f"Average Latency: {average_latency:.4f} seconds/prompt")
    # print(f"Throughput: {throughput:.2f} prompts/second")