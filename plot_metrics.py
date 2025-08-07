# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "matplotlib",
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
flex_nano_df = pd.read_csv('flex_nano_vllm_metrics.csv')
vllm_df = pd.read_csv('vllm_metrics.csv')

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Running requests comparison
ax1.plot(flex_nano_df['step'], flex_nano_df['requests_running'], 
         label='Flex Nano VLLM', color='blue', linewidth=1.5)
ax1.plot(vllm_df['steps'], vllm_df['requests_running'], 
         label='VLLM', color='red', linewidth=1.5)
ax1.set_title('Running Requests Over Time')
ax1.set_xlabel('Step')
ax1.set_ylabel('Running Requests')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Waiting requests comparison
ax2.plot(flex_nano_df['step'], flex_nano_df['requests_waiting'], 
         label='Flex Nano VLLM', color='blue', linewidth=1.5)
ax2.plot(vllm_df['steps'], vllm_df['requests_waiting'], 
         label='VLLM', color='red', linewidth=1.5)
ax2.set_title('Waiting Requests Over Time')
ax2.set_xlabel('Step')
ax2.set_ylabel('Waiting Requests')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Flex Nano VLLM step types
prefill_steps = flex_nano_df[flex_nano_df['step_type'] == 'prefill']
decode_steps = flex_nano_df[flex_nano_df['step_type'] == 'decode']

ax3.scatter(prefill_steps['step'], prefill_steps['requests_running'], 
           label='Prefill', alpha=0.6, s=10, color='green')
ax3.scatter(decode_steps['step'], decode_steps['requests_running'], 
           label='Decode', alpha=0.6, s=10, color='orange')
ax3.set_title('Flex Nano VLLM: Running Requests by Step Type')
ax3.set_xlabel('Step')
ax3.set_ylabel('Running Requests')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Total requests (running + waiting)
flex_nano_total = flex_nano_df['requests_running'] + flex_nano_df['requests_waiting']
vllm_total = vllm_df['requests_running'] + vllm_df['requests_waiting']

ax4.plot(flex_nano_df['step'], flex_nano_total, 
         label='Flex Nano VLLM Total', color='blue', linewidth=1.5)
ax4.plot(vllm_df['steps'], vllm_total, 
         label='VLLM Total', color='red', linewidth=1.5)
ax4.set_title('Total Requests (Running + Waiting)')
ax4.set_xlabel('Step')
ax4.set_ylabel('Total Requests')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("Metrics comparison saved as 'metrics_comparison.png'")
plt.show()

# Print some summary statistics
print("\n=== Summary Statistics ===")
print(f"Flex Nano VLLM - Max running: {flex_nano_df['requests_running'].max()}")
print(f"VLLM - Max running: {vllm_df['requests_running'].max()}")
print(f"Flex Nano VLLM - Max waiting: {flex_nano_df['requests_waiting'].max()}")
print(f"VLLM - Max waiting: {vllm_df['requests_waiting'].max()}")
print(f"Flex Nano VLLM - Total steps: {len(flex_nano_df)}")
print(f"VLLM - Total steps: {len(vllm_df)}")