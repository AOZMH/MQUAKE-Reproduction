import os
#os.chdir('../MQuAKE/')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="EleutherAI/gpt-j-6B", tensor_parallel_size = torch.cuda.device_count(), gpu_memory_utilization=0.999)

