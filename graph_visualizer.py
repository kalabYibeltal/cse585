import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel('data_for_cse_585_oct_20.xlsx')

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')


# Function to create and save plots
def create_plot(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 8))
    for core in df['Number of Cores'].unique():
        data = df[df['Number of Cores'] == core]
        plt.plot(data[x], data[y], marker='o', label=f'{core} cores')
    
    # plt.title(title, fontsize=16)
    # plt.xlabel(xlabel, fontsize=12)
    # plt.ylabel(ylabel, fontsize=12)
    plt.legend(title='Number of Cores', title_fontsize='12', fontsize='10')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. Average Latency vs Batch Size
create_plot('Batch Size', 'Average Latency (seconds/batch)', 
            'Average Latency vs Batch Size', 
            'Batch Size', 'Average Latency (seconds/batch)',
            'average_latency_vs_batch_size.png')

# 2. Throughput vs Batch Size
create_plot('Batch Size', 'Throughput (prompts/second)', 
            'Throughput vs Batch Size', 
            'Batch Size', 'Throughput (prompts/second)',
            'throughput_vs_batch_size.png')

# 3. Average Overall CPU Utilization vs Batch Size
create_plot('Batch Size', 'Average Overall CPU Usage (%)', 
            'Average Overall CPU Utilization vs Batch Size', 
            'Batch Size', 'Average Overall CPU Usage (%)',
            'cpu_utilization_vs_batch_size.png')

# 4. Peak Memory Usage vs Batch Size
create_plot('Batch Size', 'Peak Memory Usage (MB)', 
            'Peak Memory Usage vs Batch Size', 
            'Batch Size', 'Peak Memory Usage (MB)',
            'peak_memory_usage_vs_batch_size.png')

print("All graphs have been created and saved.")