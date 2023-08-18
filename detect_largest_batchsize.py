"""
detect largest batch_size fit single gpu with device_map={"":0}
detect largest batch_size fit multiple gpus with device_map='auto'
"auto" and "balanced" evenly split the model on all available GPUs, making it possible for you to use a larger batch size.
"""

# import necessary packages
from transformers import AutoModelForCausalLM
import torch
import time

# model
hf_model_id = "gpt2"
device_map={"":0}
# device_map="auto"

def detect_largest_batchsize(model, length=None):
    batch_size = 1
    if length is None:
        length = model.config.n_positions #the max_length of model input

    device = model.device if model.hf_device_map is None else list(set(model.hf_device_map.values()))
    while True:
        try:
            # dummy input
            test_batch = torch.ones((batch_size, length), device=model.device).long() # long() is int64 takes 4 bytes the same as float32
            model(test_batch)
            print(f"inferece {hf_model_id} with input batch_size: {batch_size} on {device}")
            batch_size *= 2
        except RuntimeError: # OOM
            batch_size //= 2
            print(f"OOM on {device}")
            break
    return batch_size

print("detected_largest_batchsize: ", detect_largest_batchsize(AutoModelForCausalLM.from_pretrained(hf_model_id, device_map=device_map), length=200))
"""
test on 2 gpus(10GB vRAM each)

1. using single gpu
inferece gpt2-xl with input batch_size: 1 on [0]
OOM on [0]
detected_largest_batchsize: 1

2. using 2 gpus
inferece gpt2-xl with input batch_size: 1 on [0, 1]
inferece gpt2-xl with input batch_size: 2 on [0, 1]
inferece gpt2-xl with input batch_size: 4 on [0, 1]
OOM on [0, 1]
detected_largest_batchsize: 4
""" 


def batchsize_timer(model):
    batch_size = 1

    max_length = 200 #model.config.n_positions the max_length of model input
    try:
        while True:
            # dummy input
            test_batch = torch.ones((batch_size, max_length), device=model.device).long() # long() is int64 takes 4 bytes the same as float32
            n_repeat = 50
            time_start = time.perf_counter()
            for i in range(0, n_repeat):
                model(test_batch)
            time_stop = time.perf_counter()
    
            batch_size *= 2
            print(f"inferece {hf_model_id} batch_size: {batch_size} on {model.device}")
            print(f"Inference a sentence in {(time_stop - time_start)/n_repeat:0.4f} seconds")
            torch.cuda.empty_cache()
    except RuntimeError:# torch.cuda.OutOfMemoryError:
        print(f"OOM on {model.device}")

# batchsize_timer(AutoModelForCausalLM.from_pretrained(hf_model_id, device_map={"":0}))
# model = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map={"":0})
# print(model.hf_device_map)
# batchsize_timer(model)
"""
inferece gpt2-xl batch_size: 2 on cuda:0
Inference a sentence in 1.1464 seconds
inferece gpt2-xl batch_size: 4 on cuda:0
Inference a sentence in 0.1289 seconds
inferece gpt2-xl batch_size: 8 on cuda:0
Inference a sentence in 0.1237 seconds
inferece gpt2-xl batch_size: 16 on cuda:0
Inference a sentence in 0.1817 seconds
inferece gpt2-xl batch_size: 32 on cuda:0
Inference a sentence in 0.2920 seconds
OOM on cuda:0
"""
# generate_with_timer(model, test_batch)