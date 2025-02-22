import copy 
import torch                      # added for deepcopy copy
from torch.utils.data import DataLoader
import re
from datasets import load_dataset, Dataset
from transformers import GenerationConfig  # updated model import
from peft import get_peft_model, LoraConfig   # added for PEFT
import wandb
from tqdm import tqdm
from grpo import GRPOTrain
from utils import extract_xml_answer, extract_hash_answer


SYSTEM_PROMPT = """
You are an helpful Assistant with excellent reasoning ability. When the user asks the question and the assistant solves the problem by reasoning in a step by step process and then provides the user with the answer. Always respond in the following format:
<reasoning> {your step by step reasoning process here} </reasoning>
<answer> {answer here} </answer>
"""

batch_size = 16
evaluation_steps = 5
ema_decay = 0.99         # EMA decay factor for reference model
kl_lambda = 0.1          # weight for KL divergence loss
group_size = 16      # number of generations per batch
dtype = torch.bfloat16
device = "cuda"
grpo_iter = 2
epochs = 5
max_len = 1024

gen_config = GenerationConfig(
    num_return_sequences=1,
    max_length=max_len,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

mini_batch_size = 4
eta = 0.1
beta = 0.1
max_iter = 100000

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
    'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
    #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
    #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
    #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
    #    answer="7"
    #)},
                {'role': 'user', 'content': x['question']}
            ],
    'answer': extract_hash_answer(x['answer'])
        }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda x:
     {k: [b[k] for b in x] for k in x[0].keys()},
)
print("batches:", len(dataloader))

trainer = GRPOTrain(batch_size=batch_size
                    , evaluation_steps=evaluation_steps
                    , ema_decay=ema_decay
                    , kl_lambda=kl_lambda
                    , group_size=group_size
                    , dtype=dtype
                    , device=device
                    , grpo_iter=grpo_iter
                    , epochs=epochs
                    , max_len=max_len
                    , gen_config=gen_config
                    , mini_batch_size=mini_batch_size
                    , eta=eta
                    , beta=beta
                    , max_iter=max_iter
                    , use_wandb = True
                    , system_prompt=SYSTEM_PROMPT
                    )

peft_config = {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"}

trainer.set_peft_config(peft_config)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
trainer.init_model(model_name, get_peft=True)
trainer.train(dataloader, save_directory="models/")
