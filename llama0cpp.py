import time
import threading
import psutil
import statistics
import sys
import os
from llama_cpp import Llama
import requests
import json
import math
import subprocess


import asyncio
import aiohttp

# Configuration
# MODEL_PATH = "../llama.cpp/Llama-3.2-3B-Instruct-uncensored-Q2_K-1.49-GB.gguf"  # Adjust path as needed
# MODEL_PATH = "../llama.cpp/Llama-3.2-3B-Instruct-uncensored-Q5_K_S-2.54-GB.gguf"  # Adjust path as needed
# MODEL_PATH = "../llama.cpp/Llama-3.2-3B-Instruct-uncensored-Q2_K-1.49-GB.gguf"  # Adjust path as needed
# MAX_LENGTH = 50
# BATCH_SIZE = 2
# NUM_CORES = 14

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

    
    
    
# model = Llama(
#     model_path=MODEL_PATH,
#     n_threads=NUM_CORES,
#     n_ctx=2048,
#     n_batch=BATCH_SIZE
# )

# cpu_usage_overall = []
# cpu_usage_per_core = []
# memory_usage = []
# latencies = []
# start_time = 0
# total_prompts_processed = 0


# Example prompts
prompts = []
with open('special_prompts.txt', 'r') as file:
    line_counter = 0
    
    # print the size here
    for line in file:
        # if line_counter > 109 and line_counter < 1000:
        #     line_counter += 1
        #     continue
        # if line_counter > (9):
        #     break
        prompts.append((line.strip()))
        
        # line_counter += 1
        # if line_counter == 5:
        #     break
        
 

# prompts.sort(key=len)


# temp = []
# for i in range(0, len(prompts) // 2):
#     temp.append(prompts[i])
#     temp.append(prompts[ len(prompts) - i -1 ])
    
# prompts =  [
#     "Who let the dogs out ",
#     "water is",
#     "who is mandela",
#     "why do I have to debug",
#     "Finaly"
# ]

print(len(prompts))


# # # Define the async function to send a POST request for each prompt
async def send_prompt(session, prompt):
    async with session.post("http://127.0.0.1:8080/completion", json={"prompt": prompt, "n_predict": 10}, timeout=6000000  ) as response:
        # data =   await response.json()
        # return data["body"]
    
        # if response.headers.get('Content-Type', None) == 'application/json':
            # data = await response.json()
        
        try:
            data = await response.json()
        except Exception as e:
            data = "error"
        
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
        minDuration = math.inf
        maxDuration = 0
        avgDuration = 0
        for response in responses:
            if isinstance(response, dict):
                # print(response['content'])
                minDuration = min(minDuration, int(response['duration']))
                maxDuration = max(maxDuration, int(response['duration']))
                avgDuration += int(response['duration'])
                
        # #         print(response['duration'])
        #     # else:
        #         # print(response)
        # #     # print(data)
        
        avgDuration = avgDuration / len(responses)
        
        print("minDuration: ", minDuration)
        print("maxDuration: ", maxDuration)
        print("avgDuration: ", avgDuration)

# Run the async event loop

def calculate_metrics(filename):
    cpu_utilization = []
    memory_utilization = []
    
    with open(filename, 'r') as file:
        # Skip the first line (start time)
        next(file)
        
        # Process each line
        for line in file:
            try:
                data = json.loads(line)
                cpu_utilization.append(data['llama-server']['cpu'])
                memory_utilization.append(data['llama-server']['mem'])
            except json.JSONDecodeError:
                continue
            
    # print(cpu_utilization[0])
    average_total_cpu = sum(cpu_utilization)
    # / (len(cpu_utilization) - 1)
    average_memory = sum(memory_utilization) / len(memory_utilization)
    
    return average_total_cpu, average_memory


if __name__ == "__main__":
    
  
    
    # print(f"Using {NUM_CORES} cores")

    # Process prompts
    print("\nBatched processing:")
    monitor_log_file =  "./monitor.jsonl"
    

    try :
        start_time = time.time_ns() + 5 * int(1e9)
        
        
        boot_command = [
            "python3",
            "./monitor.py",
            monitor_log_file,
            str(start_time),
        ]
        monitor_process = subprocess.Popen(boot_command)
        
        # for prompt in prompts:
        
        # response = requests.post("http://localhost:8080/completions", json={"prompt": prompts,"n_predict": 100})
        # print("----------------------------------------------------")
        # result =  response.json()
        # print(result)
            
        time_delta = start_time / 1e9 - time.time()
        time.sleep(time_delta if time_delta > 0 else 0)
        
        asyncio.run(gen())
        

        
        end_time = time.time_ns()
        
    except Exception as e:
        print("error", e)
        # monitor_process.terminate()
    
    finally:
        monitor_process.terminate()
        
        batch_time = (end_time - start_time) / 1000000000
    
        average_total_cpu, avg_memory = calculate_metrics(monitor_log_file)
    
    
        print(f"Time taken: {batch_time:.3f} seconds")
        print(f"Total CPU utilization: {average_total_cpu}")
        print(f"Average memory utilization: {avg_memory}")
