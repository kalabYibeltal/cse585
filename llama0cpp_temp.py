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

import asyncio
import aiohttp

# # Configuration
# # MODEL_PATH = "../llama.cpp/Llama-3.2-3B-Instruct-uncensored-Q2_K-1.49-GB.gguf"  # Adjust path as needed
# MODEL_PATH = "../llama.cpp/Llama-3.2-3B-Instruct-uncensored-Q5_K_S-2.54-GB.gguf"  # Adjust path as needed
# # MODEL_PATH = "../llama.cpp/Llama-3.2-3B-Instruct-uncensored-Q2_K-1.49-GB.gguf"  # Adjust path as needed
# MAX_LENGTH = 50
# BATCH_SIZE = 2
# NUM_CORES = 14

# # if len(sys.argv) > 1:
# #     NUM_CORES = int(sys.argv[1])
# # else:
# #     NUM_CORES = psutil.cpu_count(logical=False)

# # if len(sys.argv) > 2:
# #     BATCH_SIZE = int(sys.argv[2])

# # Initialize model

# import requests

# # prompts = [
# #     "What is AI?",
# #     "Explain quantum computing.",
# #     "what is 1 + 1",
# #     "Tell me about space exploration.",
# #     "who is obama",
# # ]

    
    
    
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

# def log_resources():
#     cpu_usage_individual = psutil.cpu_percent(interval=None, percpu=True)
#     overall_cpu_usage = psutil.cpu_percent(interval=None)
#     return overall_cpu_usage, cpu_usage_individual

# def generate_response(prompt, max_length=MAX_LENGTH):
    
#     output = model(
#         prompt,
#         max_tokens=max_length,
#         echo=False,
#         temperature=0.7
#     )
#     # print(output)
#     return output['choices'][0]['text']


# def process_prompts(prompts):
#     global total_prompts_processed
#     global latencies, cpu_usage_overall, cpu_usage_per_core
    
#     start_time = time.time()
#     responses = []
#     overall_cpu, cpu = log_resources()
    
#     for i in range(0, len(prompts), BATCH_SIZE):
#         request_start_time = time.time()
#         batch = prompts[i:i+BATCH_SIZE]
        
#         # Process each prompt in the batch
#         batch_responses = []
#         for prompt in batch:
#             response = generate_response(prompt)
#             batch_responses.append(response)
            
#         responses.extend(batch_responses)
#         latencies.append(time.time() - request_start_time)
    
#     overall_cpu, cpu = log_resources()
#     cpu_usage_overall.append(overall_cpu)
#     cpu_usage_per_core.append(cpu)
    
#     end_time = time.time()
#     total_prompts_processed += len(prompts)
#     return responses, end_time - start_time

# def calculate_avg_cpu_utilization_over_time(cpu_data_array, threshold=5):
#     total_utilization = 0
#     valid_core_count = 0

#     for snapshot in cpu_data_array:
#         filtered_cores = [usage for usage in snapshot if usage >= threshold]
#         if filtered_cores:
#             total_utilization += sum(filtered_cores)
#             valid_core_count += len(filtered_cores)

#     if valid_core_count == 0:
#         return 0.0, 0

#     avg_utilization = total_utilization / valid_core_count
#     return avg_utilization

# Example prompts
# prompts = ["What are the biggest buildings in north america?", "How important is climate change in the future?", "What is the best strategy for studying in school?", "Name the top 5 most famous people in the world right now?"]
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
print(len(prompts))


    
# prompts = temp

# [
#     "Who let the dogs out ",
#     "water is",
#     "who is mandela",
#     "why do I have to debug",
#     "Finaly"
# ]




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
                # print("--------------------------")
                # print(response['content'])
                minDuration = min(minDuration, int(response['duration']))
                maxDuration = max(maxDuration, int(response['duration']))
                avgDuration += int(response['duration'])
                
                # print(response['duration'])
           
        
        avgDuration = avgDuration / len(responses)
        
        print("minDuration: ", minDuration)
        print("maxDuration: ", maxDuration)
        print("avgDuration: ", avgDuration)

# Run the async event loop


if __name__ == "__main__":
    
  
    

    # Process prompts
    print("\nBatched processing:")
    # batch_responses, batch_time = process_prompts(prompts)

    
    start_time = time.time()

    # for prompt in prompts:
    
    # response = requests.post("http://localhost:8080/completions", json={"prompt": prompts,"n_predict": 100})
    # print("----------------------------------------------------")
    # result =  response.json()
    # print(len(result))
        
    # time_delta = start_time / 1e9 - time.time()
    # time.sleep(time_delta if time_delta > 0 else 0)
    
    asyncio.run(gen())
    
    # response = requests.post(
    #     "http://127.0.0.1:8080/completion",
    #     json={
    #         "prompt": "who am i ",
    #         "n_predict": 128
    #     }
    # )
    # Check if the request was successful
    # if response.status_code == 200:
    # # Parse the JSON response and print the content
    #     data = response.json()
    #     print(data.get('content')) 
    # print(response)
    
    end_time = time.time()
    
    batch_time = end_time - start_time
    
    
    
    
    print(f"Time taken: {batch_time:.2f} seconds")

    # Calculate metrics
    # average_latency = sum(latencies) / len(latencies)
    # throughput = total_prompts_processed / batch_time
    
    # process = psutil.Process()  
    # memory_info = process.memory_info()
    # peak_memory_usage = memory_info.vms / (1024 * 1024)
    # avg_overall_cpu_usage = sum(cpu_usage_overall) / len(cpu_usage_overall)

    # # Print results
    # print("\nResource Usage:")
    # print(f"Average CPU Usage for each Core: {[sum(core)/NUM_CORES for core in zip(*cpu_usage_per_core)]}")
    # print(f"Average CPU Usage per Core: {calculate_avg_cpu_utilization_over_time(cpu_usage_per_core)}")
    # print(f"Average overall CPU Usage: {avg_overall_cpu_usage:.2f}%")
    # print(f"Peak Memory Usage: {peak_memory_usage:.2f}Mbs")

    # print("\nPerformance Metrics:")
    # print(f"Total Time: {batch_time:.2f} seconds")
    # print(f"Average Latency: {average_latency:.4f} seconds/prompt")
    # print(f"Throughput: {throughput:.2f} prompts/second")