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

BATCH_SIZE = 16
evaluation_steps = 5
ema_decay = 0.99         # EMA decay factor for reference model
kl_lambda = 0.1          # weight for KL divergence loss
GROUP_SIZE = 16      # number of generations per batch
DTYPE = torch.bfloat16
DEVICE = "cuda"
GRPO_ITER = 2
EPOCHS = 5
MAX_LEN = 1024

GEN_CONFIG = GenerationConfig(
    num_return_sequences=1,
    max_length=MAX_LEN,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

MINI_BATCH_SIZE = 4
ETA = 0.1
BETA = 0.1
MAX_ITER = 100000

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
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x:
     {k: [b[k] for b in x] for k in x[0].keys()},
)
print("batches:", len(dataloader))

trainer = GRPOTrain(BATCH_SIZE=BATCH_SIZE
                    , evaluation_steps=evaluation_steps
                    , ema_decay=ema_decay
                    , kl_lambda=kl_lambda
                    , GROUP_SIZE=GROUP_SIZE
                    , DTYPE=DTYPE
                    , DEVICE=DEVICE
                    , GRPO_ITER=GRPO_ITER
                    , EPOCHS=EPOCHS
                    , MAX_LEN=MAX_LEN
                    , GEN_CONFIG=GEN_CONFIG
                    , MINI_BATCH_SIZE=MINI_BATCH_SIZE
                    , ETA=ETA
                    , BETA=BETA
                    , MAX_ITER=MAX_ITER
                    , use_wandb = True
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
