"""
load and inference model:

- on `CPU` is **ok but slow**
- device_map=”auto”(2 10GB vRAM GPU and CPU used for llama 7B) is **ok but still slow** evenly to the time cost on CPU
- device_map=”auto”(2 10GB vRAM GPU and CPU used for gpt2-xl) **is ok and speed up** to 2s while 25s on CPU
- device_map on single GPU throw OOM error
"""

# import necessary packages
from transformers import AutoTokenizer, AutoModelForCausalLM
# test on 2*10GB vRAM GPU
import time
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils.modeling import get_balanced_memory

hf_model_id="huggyllama/llama-7b"
# hf_model_id="gpt2-xl"

def generate_with_timer(model, inputs):
    time_start = time.perf_counter()
    # Generate
    outputs = model.generate(inputs["input_ids"])
    decoded_sentence = tokenizer.decode(outputs[0].tolist())
    print(decoded_sentence)
    time_stop = time.perf_counter()
    print(f"Inference a sentence in {time_stop - time_start:0.4f} seconds")

# initialize tokenizer and tokenize input sentense
tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
inputs = tokenizer("Hello world! My name is", return_tensors="pt")
"""""""""""""""""inference large model on cpu, very slow!"""""""""""""""""
# initialize model on cpu
llama7b = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
print(llama7b.device)
"""cpu"""

# generate with cpu
generate_with_timer(llama7b, inputs)
"""
Inference a sentence in 25.2321 seconds
"""

"""""""inference large model on gpus(device_map=auto), very slow! too"""""""
# max_memory = get_balanced_memory(llama7b)
# device_map = infer_auto_device_map(llama7b, max_memory=max_memory)
# dispatched_llama7b = dispatch_model(llama7b, device_map)
dispatched_llama7b = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map="auto")
# You can inspect how the model was split across devices by looking at its hf_device_map attribute
# put the tokenized ids to GPU, the tokenized id should in the same device with model, otherwise cuda error triggered
inputs=inputs.to(0)
print(dispatched_llama7b.hf_device_map)
# need 24GB vRAM, 2x10GB vRAM is not enough, so gpu is still needed. That's why it's slow. The gpt-xl(1.5 biilion parameters) will take 25.8s on cpu and 2s on 2 gpus(no cpu need).
"""
{'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu', 'model.norm': 'cpu', 'lm_head': 'cpu'}
"""
# generate with gpus
generate_with_timer(dispatched_llama7b, inputs)

"""""""inference large model on single gpu, very slow! too"""""""
# throw OutOfMemoryError
llama7b = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map={"": 0})