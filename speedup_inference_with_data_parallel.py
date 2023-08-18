"""
Usage:
accelerate launch --num_processes=2 speedup_inference_with_data_parallel.py 

"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import PartialState
from accelerate.utils import gather
import torch.nn.functional as F
import time
import os


hf_model_id = 'gpt2'

batch_size=32
length=200


current_env = os.environ.copy()
use_data_parallel = "MASTER_PORT" in current_env
# use_data_parallel = True

if use_data_parallel:
    test_batch = torch.ones((batch_size, length)).long() # long() is int64 takes 4 bytes the same as float32
    model = AutoModelForCausalLM.from_pretrained(hf_model_id)
    distributed_state = PartialState()
    distributed_model = model.to(distributed_state.device)

    n_repeat=50

    time_start = time.perf_counter()
    for _ in range(0, n_repeat):
        with distributed_state.split_between_processes(test_batch) as splited_test_batch:
            # print(distributed_state.device)
            splited_test_batch = splited_test_batch.to(distributed_state.device)
            # print(f'{splited_test_batch.size()} on {distributed_state.process_index}')
            distributed_output = distributed_model(splited_test_batch)
            # print(distributed_state.device)
            # print(distributed_state.process_index)
            # logits = F.log_softmax(distributed_output.logits, dim=-1)
            outputs = gather(distributed_output.logits)
            # print(outputs)
    time_stop = time.perf_counter()            
    print(f"inferece {hf_model_id} batch_size: {batch_size} on process: {distributed_state.process_index} in {(time_stop - time_start)/n_repeat:0.4f} seconds")
    """
    inferece gpt2 batch_size: 16 on process: 1 in 0.1418 seconds
    inferece gpt2 batch_size: 16 on process: 2 in 0.1423 seconds
    """
else:
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map={"":0})
    test_batch = torch.ones((batch_size//2, length)).long() # long() is int64 takes 4 bytes the same as float32
    test_batch = test_batch.to(model.device)
    n_repeat=50

    time_start = time.perf_counter()
    for _ in range(0, n_repeat):
        model(test_batch)
    time_stop = time.perf_counter()  
    print(f"inferece {hf_model_id} batch_size: {batch_size//2} on {model.device} in {(time_stop - time_start)/n_repeat:0.4f} seconds")
    """
    inferece gpt2 batch_size: 16 on cuda:0 in 0.1030 seconds
    """