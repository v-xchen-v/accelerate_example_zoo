"""dispatch model to different devices"""

# import necessary packages
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model

# initialize tokenizer and tokenize input sentense
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello world! My name is", return_tensors="pt").to(0)

# initialize model
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
# print weight_names of checkpoint
print([gpt2.state_dict().keys()])
"""
odict_keys(['transformer.wte.weight', 'transformer.wpe.weight', 'transformer.h.0.ln_1.weight', 'transformer.h.0.ln_1.bias', 'transformer.h.0.attn.c_attn.weight', 'transformer.h.0.attn.c_attn.bias', 'transformer.h.0.attn.c_proj.weight', 'transformer.h.0.attn.c_proj.bias', 'transformer.h.0.ln_2.weight', 'transformer.h.0.ln_2.bias', 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_fc.bias', 'transformer.h.0.mlp.c_proj.weight', 'transformer.h.0.mlp.c_proj.bias', 'transformer.h.1.ln_1.weight', 'transformer.h.1.ln_1.bias', 'transformer.h.1.attn.c_attn.weight', 'transformer.h.1.attn.c_attn.bias', 'transformer.h.1.attn.c_proj.weight', 'transformer.h.1.attn.c_proj.bias', 'transformer.h.1.ln_2.weight', 'transformer.h.1.ln_2.bias', 'transformer.h.1.mlp.c_fc.weight', 'transformer.h.1.mlp.c_fc.bias', 'transformer.h.1.mlp.c_proj.weight', 'transformer.h.1.mlp.c_proj.bias', 'transformer.h.2.ln_1.weight', 'transformer.h.2.ln_1.bias', 'transformer.h.2.attn.c_attn.weight', 'transformer.h.2.attn.c_attn.bias', 'transformer.h.2.attn.c_proj.weight', 'transformer.h.2.attn.c_proj.bias', 'transformer.h.2.ln_2.weight', 'transformer.h.2.ln_2.bias', 'transformer.h.2.mlp.c_fc.weight', 'transformer.h.2.mlp.c_fc.bias', 'transformer.h.2.mlp.c_proj.weight', 'transformer.h.2.mlp.c_proj.bias', 'transformer.h.3.ln_1.weight', 'transformer.h.3.ln_1.bias', 'transformer.h.3.attn.c_attn.weight', 'transformer.h.3.attn.c_attn.bias', 'transformer.h.3.attn.c_proj.weight', 'transformer.h.3.attn.c_proj.bias', 'transformer.h.3.ln_2.weight', 'transformer.h.3.ln_2.bias', 'transformer.h.3.mlp.c_fc.weight', 'transformer.h.3.mlp.c_fc.bias', 'transformer.h.3.mlp.c_proj.weight', 'transformer.h.3.mlp.c_proj.bias', 'transformer.h.4.ln_1.weight', 'transformer.h.4.ln_1.bias', 'transformer.h.4.attn.c_attn.weight', 'transformer.h.4.attn.c_attn.bias', 'transformer.h.4.attn.c_proj.weight', 'transformer.h.4.attn.c_proj.bias', 'transformer.h.4.ln_2.weight', 'transformer.h.4.ln_2.bias', 'transformer.h.4.mlp.c_fc.weight', 'transformer.h.4.mlp.c_fc.bias', 'transformer.h.4.mlp.c_proj.weight', 'transformer.h.4.mlp.c_proj.bias', 'transformer.h.5.ln_1.weight', 'transformer.h.5.ln_1.bias', 'transformer.h.5.attn.c_attn.weight', 'transformer.h.5.attn.c_attn.bias', 'transformer.h.5.attn.c_proj.weight', 'transformer.h.5.attn.c_proj.bias', 'transformer.h.5.ln_2.weight', 'transformer.h.5.ln_2.bias', 'transformer.h.5.mlp.c_fc.weight', 'transformer.h.5.mlp.c_fc.bias', 'transformer.h.5.mlp.c_proj.weight', 'transformer.h.5.mlp.c_proj.bias', 'transformer.h.6.ln_1.weight', 'transformer.h.6.ln_1.bias', 'transformer.h.6.attn.c_attn.weight', 'transformer.h.6.attn.c_attn.bias', 'transformer.h.6.attn.c_proj.weight', 'transformer.h.6.attn.c_proj.bias', 'transformer.h.6.ln_2.weight', 'transformer.h.6.ln_2.bias', 'transformer.h.6.mlp.c_fc.weight', 'transformer.h.6.mlp.c_fc.bias', 'transformer.h.6.mlp.c_proj.weight', 'transformer.h.6.mlp.c_proj.bias', 'transformer.h.7.ln_1.weight', 'transformer.h.7.ln_1.bias', 'transformer.h.7.attn.c_attn.weight', 'transformer.h.7.attn.c_attn.bias', 'transformer.h.7.attn.c_proj.weight', 'transformer.h.7.attn.c_proj.bias', 'transformer.h.7.ln_2.weight', 'transformer.h.7.ln_2.bias', 'transformer.h.7.mlp.c_fc.weight', 'transformer.h.7.mlp.c_fc.bias', 'transformer.h.7.mlp.c_proj.weight', 'transformer.h.7.mlp.c_proj.bias', 'transformer.h.8.ln_1.weight', 'transformer.h.8.ln_1.bias', 'transformer.h.8.attn.c_attn.weight', 'transformer.h.8.attn.c_attn.bias', 'transformer.h.8.attn.c_proj.weight', 'transformer.h.8.attn.c_proj.bias', 'transformer.h.8.ln_2.weight', 'transformer.h.8.ln_2.bias', 'transformer.h.8.mlp.c_fc.weight', 'transformer.h.8.mlp.c_fc.bias', 'transformer.h.8.mlp.c_proj.weight', 'transformer.h.8.mlp.c_proj.bias', 'transformer.h.9.ln_1.weight', 'transformer.h.9.ln_1.bias', 'transformer.h.9.attn.c_attn.weight', 'transformer.h.9.attn.c_attn.bias', 'transformer.h.9.attn.c_proj.weight', 'transformer.h.9.attn.c_proj.bias', 'transformer.h.9.ln_2.weight', 'transformer.h.9.ln_2.bias', 'transformer.h.9.mlp.c_fc.weight', 'transformer.h.9.mlp.c_fc.bias', 'transformer.h.9.mlp.c_proj.weight', 'transformer.h.9.mlp.c_proj.bias', 'transformer.h.10.ln_1.weight', 'transformer.h.10.ln_1.bias', 'transformer.h.10.attn.c_attn.weight', 'transformer.h.10.attn.c_attn.bias', 'transformer.h.10.attn.c_proj.weight', 'transformer.h.10.attn.c_proj.bias', 'transformer.h.10.ln_2.weight', 'transformer.h.10.ln_2.bias', 'transformer.h.10.mlp.c_fc.weight', 'transformer.h.10.mlp.c_fc.bias', 'transformer.h.10.mlp.c_proj.weight', 'transformer.h.10.mlp.c_proj.bias', 'transformer.h.11.ln_1.weight', 'transformer.h.11.ln_1.bias', 'transformer.h.11.attn.c_attn.weight', 'transformer.h.11.attn.c_attn.bias', 'transformer.h.11.attn.c_proj.weight', 'transformer.h.11.attn.c_proj.bias', 'transformer.h.11.ln_2.weight', 'transformer.h.11.ln_2.bias', 'transformer.h.11.mlp.c_fc.weight', 'transformer.h.11.mlp.c_fc.bias', 'transformer.h.11.mlp.c_proj.weight', 'transformer.h.11.mlp.c_proj.bias', 'transformer.ln_f.weight', 'transformer.ln_f.bias', 'lm_head.weight'])
"""

# Dispatch on GPUs 0 and 1
# device_map: A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the same device.
# {module_name: device} and make sure each weight_name startswith modulename
device_map = {
    "transformer.wte": 0,
    "transformer.wpe": 0,
    "transformer.ln_f": 1,
    "lm_head": 0,
}
for i in range(12):
    device_map[f"transformer.h.{i}"] = 0 if i <= 5 else 1
gpt2 = dispatch_model(gpt2, device_map)
# You can inspect how the model was split across devices by looking at its hf_device_map attribute
print(gpt2.hf_device_map)
"""
{'transformer.wte': 0, 'transformer.wpe': 0, 'transformer.ln_f': 1, 'lm_head': 0, 'transformer.h.0': 0, 'transformer.h.1': 0, 'transformer.h.2': 0, 'transformer.h.3': 0, 'transformer.h.4': 0, 'transformer.h.5': 0, 'transformer.h.6': 1, 'transformer.h.7': 1, 'transformer.h.8': 1, 'transformer.h.9': 1, 'transformer.h.10': 1, 'transformer.h.11': 1}
"""

# Generate
outputs = gpt2.generate(inputs["input_ids"])
decoded_sentence = tokenizer.decode(outputs[0].tolist())
print(decoded_sentence)
"""Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo"""