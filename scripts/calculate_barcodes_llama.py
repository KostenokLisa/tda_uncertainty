import os
import sys

script = str(sys.argv[1])

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
import torch
from transformers import AdamW
import subprocess

PRETRAINED_WEIGHTS="/mnt/disk2-part1/pretrained-models-pytorch/llm/llama-7b-hf"

config = AutoConfig.from_pretrained(PRETRAINED_WEIGHTS, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_WEIGHTS, use_fast=False, trust_remote_code=True)
kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True, "device_map": "balanced"}
model = AutoModelForCausalLM.from_pretrained(PRETRAINED_WEIGHTS, config=config, trust_remote_code=True, **kwargs)

import coqa
seed=42

get_dataset = coqa.get_dataset
dataset = get_dataset(tokenizer)
dataset = dataset.train_test_split(test_size=(1 - 0.7), seed=seed)['test']
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

import numpy as np

def grab_attention_weights(model, inputs):
    inputs['input_ids'] = inputs['input_ids'].to("cuda:0")
    inputs['attention_mask'] = inputs['attention_mask'].to("cuda:0")

    with torch.no_grad():
        attention = model(
            inputs["input_ids"], output_hidden_states=True, output_attentions=True
        )['attentions']
    # layer X sample X head X n_token X n_token
    attention = np.asarray([layer.cpu().detach().numpy() for layer in attention], dtype=np.float16)
    
    return attention

from math import ceil
import numpy as np

def calculate_barcodes(filenames, i_start, i_finish):
    # topological_features.py
    subprocess.run(f"python subproccesing.py {script} {i_start} {i_finish}", shell=True, check=True)
    # subprocess.run(f"python run_barcodes_calc.py {i_start} {i_finish}", shell=True, check=True)
    print(i , "out of", len(dataloader))
    for file_path in filenames:
        os.remove(file_path)


output_dir = "./llama_coqa_atten_test/"
filenames = []

BATCH = 100

i = 0
for batch in dataloader:
    attention_w = grab_attention_weights(model, batch)
    filename = output_dir + "part" + str(i) + '.npy'
    np.save(filename, attention_w)
    filenames.append(filename)

    if i % BATCH == BATCH - 1:
        assert len(filenames) == BATCH
        calculate_barcodes(filenames, i_start=i - BATCH + 1, i_finish=i + 1)
        filenames = []

    i += 1

if len(filenames) > 0:
    i_start = i - len(filenames)
    assert i_start % BATCH == 0
    calculate_barcodes(filenames, i_start=i_start, i_finish=i)

print("Results saved.")