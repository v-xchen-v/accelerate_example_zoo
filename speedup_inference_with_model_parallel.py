"""
setting device_map = auto means that you want to set Model Parallel(MP), meaning putting the model into different GPU layers and one GPU at a time will be used

detect largest batch_size fit multiple gpus with device_map='auto'
"auto" and "balanced" evenly split the model on all available GPUs, making it possible for you to use a larger batch size.

model parallel can benefit inference speed in two situations:
1. with model parallel, huge model(cannot fit in single gpu) could splits and inferences with multiple gpus with high speed
Notice: If there are some submodules on cpu or disk, the inference speed could be very slow, close to inference speed on GPU)
2. larger batch size could bring shorter inference time per token
"""

# import necessary packages
from transformers import AutoModelForCausalLM, AutoTokenizer
# from accelerate import infer_auto_device_map, dispatch_model
# import torch
import time

# generate with gpus
def generate_with_timer(model, inputs):
    device = model.device if not hasattr(model, "hf_device_map") or model.hf_device_map is None else list(set(model.hf_device_map.values()))
    if hasattr(model, "model_parallel") and model.model_parallel:
        device = list(set(model.device_map.keys()))
    else:
        device = model.device if not hasattr(model, "hf_device_map") or model.hf_device_map is None else list(set(model.hf_device_map.values()))
    time_start = time.perf_counter()

    # Generate
    outputs = model.generate(inputs["input_ids"])
    time_stop = time.perf_counter()
    print(f"Inference a sentence in {time_stop - time_start:0.4f} seconds on {device}")

    # decode ids to sentence
    decoded_sentence = tokenizer.decode(outputs[0].tolist())
    print(decoded_sentence)

hf_model_id = "gpt2-xl"
# initialize tokenizer and tokenize input sentense
tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
inputs = tokenizer("Hello world! My name is", return_tensors="pt")

flag = 5
# """"""""""""inference on cpu""""""""""""
if flag==1:
    model = AutoModelForCausalLM.from_pretrained(hf_model_id)
    generate_with_timer(model, inputs)
"""
Inference a sentence in 16.8693 seconds on cpu
"""

# """"""""""""inference on gpus with model parallel""""""""""""
if flag==2:
    # device_map = "auto"
    # model.parallelize()
    # print(model.device_map)
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map="auto")
    generate_with_timer(model, inputs.to(model.device))
"""
Inference a sentence in 1.6531 seconds on [0, 1](gpus)
"""

# model.deparallelize()
# print(model.device) # cpu
# """"""""""""inference on single gpu""""""""""""
if flag==3:
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map={"":0})
    print(model.hf_device_map)
    generate_with_timer(model, inputs.to(model.device))
"""
Inference a sentence in 0.9578 seconds on [0]
"""

if flag==4:
    hf_model_id = 'huggyllama/llama-7b'
    model = AutoModelForCausalLM.from_pretrained(hf_model_id)
    generate_with_timer(model, inputs)
"""
Inference a sentence in 25.6326 seconds on cpu
"""

if flag==5:
    hf_model_id = 'huggyllama/llama-7b'
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map="auto")
    print(model.hf_device_map)
    generate_with_timer(model, inputs.to(model.device))
"""
Inference a sentence in 34.6187 seconds on [0, 1, 'cpu']
"""
 