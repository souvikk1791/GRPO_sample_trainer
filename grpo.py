import torch
import torch.nn as nn
import torch.nn.functional as F   # added for KL divergence
import copy                       # added for deepcopy copy
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # updated model import
from peft import get_peft_model, LoraConfig   # added for PEFT
from tqdm import tqdm
from utils import *
import wandb
# Global variables
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


class GRPOTrain():
    def __init__(self, 
                 BATCH_SIZE: int = 16, 
                 evaluation_steps: int = 5,
                 ema_decay: float = 0.99, 
                 kl_lambda: float = 0.1, 
                 GROUP_SIZE: int = 16, 
                 DTYPE = torch.float16, 
                 DEVICE: str = "cuda", 
                 GRPO_ITER: int = 2, 
                 EPOCHS: int = 5, 
                 MAX_LEN: int = 1024,
                 wandb = None,
                 MINI_BATCH_SIZE = 4, 
                 ETA = 0.1,
                 BETA = 0.1,
                 GEN_CONFIG: GenerationConfig = None):
        """
        Initialize GRPOTrain with training configuration.

        Args:
            BATCH_SIZE (int): Batch size for training.
            evaluation_steps (int): Frequency of evaluation steps.
            ema_decay (float): EMA decay for the reference model.
            kl_lambda (float): Weight for KL divergence loss.
            GROUP_SIZE (int): Number of generations per batch.
            DTYPE: Data type for computation.
            DEVICE (str): Device for model training (e.g., 'cuda').
            GRPO_ITER (int): Number of GRPO iterations per batch.
            EPOCHS (int): Total number of training epochs.
            MAX_LEN (int): Maximum length for model inputs.
            GEN_CONFIG (GenerationConfig): Generation configuration; defaults to preset if None.
        """
        if GEN_CONFIG is None:
            GEN_CONFIG = GenerationConfig(
                num_return_sequences=1,
                max_length=1024,
                do_sample=True,
                temperature=0.75,
                top_p=0.75,
            )
        self.BATCH_SIZE = BATCH_SIZE
        self.evaluation_steps = evaluation_steps
        self.ema_decay = ema_decay         # EMA decay factor for reference model
        self.kl_lambda = kl_lambda          # weight for KL divergence loss
        self.GROUP_SIZE = GROUP_SIZE        # number of generations per batch
        self.DTYPE = DTYPE
        self.DEVICE = DEVICE
        self.GRPO_ITER = GRPO_ITER
        self.EPOCHS = EPOCHS
        self.MAX_LEN = MAX_LEN
        self.MINI_BATCH_SIZE = MINI_BATCH_SIZE
        self.ETA = ETA
        self.BETA = BETA
        self.GEN_CONFIG = GEN_CONFIG
        self.MAX_ITER = 100000

        if wandb:
            wandb.init(project="grpo")

    def compute_step_loss(self, logp_proxy, logp_refs, logp_old, advantages, current_inputs, final_inputs_len, output_len):
    # Shift logits and labels for causal LM (predict next token)
        kl_div = (
                torch.exp(logp_refs - logp_proxy)
                - (logp_refs - logp_proxy)
                - 1
                    )

        adv_scale = torch.exp(logp_proxy - logp_old)
        clipped_scale = torch.clamp(adv_scale, 1 - self.ETA, 1 + self.ETA)
        scaled_advantages = (
            advantages * adv_scale
        )
        clipped_advantages = (
            advantages * clipped_scale
        )

        mask = scaled_advantages < clipped_advantages
        final_advantages = torch.where(
            mask, scaled_advantages, clipped_advantages
        )

        per_token_loss = -final_advantages + self.BETA * kl_div
        loss_mask = current_inputs["attention_mask"][:, :-1]

        for idx, k in enumerate(final_inputs_len):
            loss_mask[idx, :k] = 0

        loss = torch.sum(
                torch.where(
                    loss_mask.bool(),
                    per_token_loss,
                    torch.zeros_like(per_token_loss),
                ), dim=1
            )/output_len
        
        step_kl_div = torch.sum(kl_div * loss_mask, dim=1)/output_len
        step_loss = loss
        step_objective = torch.sum((final_advantages - kl_div * self.BETA) * loss_mask, dim=1)/output_len

        return step_objective, step_loss, step_kl_div
    
    def set_peft_config(self, peft_config = {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"}):
        self.peft_config = LoraConfig(**peft_config)

    def init_model(self, model_name, get_peft=True):

        # Load the model
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name,device_map=self.DEVICE, )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if get_peft:
            self.model = get_peft_model(self.model, self.peft_config)

        # Create reference model as a deep copy and set to eval
        self.reference_model = copy.deepcopy(self.model)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        optimizer = Adam(self.model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.MAX_ITER)

    def selective_log_softmax(self, logits, index):

        """
        A memory-efficient implementation of the common `log_softmax -> gather` operation.

        This function is equivalent to the following naive implementation:
        ```python
        logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        ```

        Args:
            logits (`torch.Tensor`):
                Logits tensor of shape `(..., num_classes)`.
            index (`torch.Tensor`):
                Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

        Returns:
            `torch.Tensor`:
                Gathered log probabilities with the same shape as `index`.
        """
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            # loop to reduce peak mem consumption
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def logp_per_token(self, model, inputs):

        with torch.autocast(self.DEVICE, self.DTYPE):
            outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        input_ids = inputs["input_ids"][:, 1:]
        return self.selective_log_softmax(logits.float(), input_ids)
    

    def run_inference(self, model, inputs):
        with torch.inference_mode():
            outputs =model.generate(
                            **inputs,
                            generation_config=self.GEN_CONFIG
                        )
            torch.cuda.empty_cache()
            return outputs

            
    
    def train(self, train_dataloader, val_dataloader, save_directory):

        pbar = tqdm(
        range(len(train_dataloader) * self.EPOCHS * self.GRPO_ITER), 
        desc="GRPO Training", smoothing=0.01
            )

        for epoch in range(self.EPOCHS):

            for batch in train_dataloader:
                self.model.train()

                self.optimizer.zero_grad()
                total_loss_sum = 0.0
                # Run multiple generations per batch and average the loss
                output_groups = []

                questions = [x for x in [q]*self.GROUP_SIZE  for q in batch["question"]]
                answers = [x for x in [q]*self.GROUP_SIZE  for q in batch["answer"]]

                batch_inputs = self.tokenizer.apply_chat_template(batch["prompt"])
                inputs = self.tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                ).to(self.DEVICE)

                refs = self.tokenizer(
                    batch_inputs, return_tensors="pt", padding=True, padding_side="left"
                )["input_ids"]                

                    # use sequential generation for now
                for generation in range(self.GROUP_SIZE):
                    outputs = self.run_inference(self.model, inputs, group_size=self.GROUP_SIZE)
                    output_groups.append(outputs)

            
                ref_stacked = [refs]*self.GROUP_SIZE
                ref_stacked = torch.stack(ref_stacked)
                # logits_stacked = torch.stack([output.logits for output in output_groups], dim=1)
                # logits_stacked = logits_stacked.view(-1, logits_stacked.size(-1))
                completions = [
                    self.tokenizer.decode(g[len(l) :], skip_special_tokens=True)
                    for (g, l) in zip(torch.stack(output_groups), ref_stacked)
                ]

                completions = completions.view(self.GROUP_SIZE, -1).T
                advantages, rewards, rewards_list = self.get_advantages(
                    completions,
                    group_size=self.GROUP_SIZE,
                    answers=answers,
                    questions=questions,
                )
                if self.wandb:
                    table = self.wandb.Table(columns=["question", "answer", "model output", "reward"])
                    for idx, (q, a, c, r) in enumerate(
                        zip(questions, answers, completions, rewards)
                    ):
                        if idx % self.GROUP_SIZE == 0:
                            table.add_data(q, a, c, r.item())
                    self.wandb.log({"completions": table}, commit=False)

                ## Compute reference log-prob
                logp_old = [None] * len(inputs["input_ids"])

                final_inputs_len = [
                    len(
                        self.tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": q},
                            ]
                        )
                    )
                    for q in questions
                ]

                final_model_input = [
                            self.tokenizer.apply_chat_template(
                                [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": q},
                                    {"role": "assistant", "content": c},
                                ],
                                tokenize=False,
                            )
                            for q, c in zip(questions, completions)
                        ]
                inputs = self.tokenizer(
                    final_model_input,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.MAX_LEN,
                    padding_side="left",
                    truncation=True,
                )

                final_total_len = (inputs["attention_mask"].sum(dim=1) - 1).tolist()
                output_len = [
                    (total - inp_len - 1)
                    if total < self.MAX_LEN
                    else (min(self.MAX_LEN, total - inp_len) - 1)
                    for inp_len, total in zip(final_inputs_len, final_total_len)
                ]

                # check this implementation is required or not
                inputs["input_ids"] = inputs["input_ids"][:, -self.MAX_LEN:]
                inputs["attention_mask"] = inputs["attention_mask"][:, self.MAX_LEN:]
                # token_with_loss = sum(output_len)


                with torch.inference_mode():
                    logp_refs_stacked = []
                    for i in range(0, len(inputs["input_ids"]), self.MINI_BATCH_SIZE):

                        steps += 1
                        current_inputs = {
                            k: v[i : i + self.MINI_BATCH_SIZE].clone().to(self.DEVICE) for k, v in inputs.items()
                        }
                        logp_refs = self.logp_per_token(self.reference_model, current_inputs)

                        current_inputs = {
                            k: v[i : i + self.MINI_BATCH_SIZE].to("cpu") for k, v in inputs.items()
                        }
                        current_inputs = None
                        torch.cuda.empty_cache()
                        logp_refs_stacked.append(logp_refs)


                for _ in range(self.GRPO_ITER):
                    step_loss = 0
                    step_kl_div = 0
                    step_objective = 0
                    steps = 0

                    for i in range(0, len(inputs["input_ids"]), self.MINI_BATCH_SIZE):

                        steps += 1
                        current_inputs = {
                            k: v[i : i + self.MINI_BATCH_SIZE].clone().to(self.DEVICE) for k, v in inputs.items()
                        }
                        with torch.enable_grad():
                            logp_proxy = self.logp_per_token(self.model, current_inputs)

                        current_inputs = {
                            k: v[i : i + self.MINI_BATCH_SIZE].to("cpu") for k, v in inputs.items()
                        }
                        if logp_old[i] is None:
                            logp_old[i] = logp_proxy.detach().clone()                

                        _objective, _loss, _kl_div = self.compute_step_loss(logp_proxy, logp_refs_stacked[i], logp_old[i], advantages[i : i + self.MINI_BATCH_SIZE], 
                                                                    current_inputs, final_inputs_len[i : i + self.MINI_BATCH_SIZE], output_len[i : i + self.MINI_BATCH_SIZE])    

                        step_loss += _loss
                        step_kl_div += _kl_div
                        step_objective += _objective

                        current_inputs = None

                        
                            
                    step_loss = step_loss.sum()/steps   
                    step_loss.backward()
                    step_loss = step_loss.float().item()
                    step_kl_div = step_kl_div.sum().float().item() / steps
                    step_objective = step_objective.sum().float().item() / steps

                    pbar.set_postfix(
                        {
                            "rewards": rewards.float().mean().item(),
                            "loss": step_loss,
                            "kl_div": step_kl_div,
                            "objective": step_objective,
                        }
                    )
                    # torch.nn.utils.clip_grad_norm_(lycoris_net.parameters(), 0.1)
                    self.optimizer.step()
                    self.scheduler.step()  # update cosine scheduler after optimizer step
                    self.optimizer.zero_grad()

                    pbar.update()
                    if self.wandb:
                        self.wandb.log(
                            {
                                "rewards": rewards.float().mean().item(),
                                "correctness_reward": torch.mean(torch.tensor(rewards_list[0])),
                                "format_reward": torch.mean(torch.tensor(rewards_list[1])),
                                "loss": step_loss,
                                "kl_div": step_kl_div,
                                "objective": step_objective,
                            },
                            commit=False,
                        )
                        self.wandb.log({"global_step": pbar.n})

                    if pbar.n > self.MAX_ITER:
                        break

                # Update reference model parameters with EMA
                with torch.no_grad():
                    for ref_param, param in zip(self.reference_model.parameters(), self.model.parameters()):
                        ref_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

                # if training_step % evaluation_steps == 0:
                #     model.eval()
                #     eval_loss = 0.0
                #     eval_steps = 0
                #     with torch.no_grad():
                #         for eval_batch in eval_dataloader:
                #             batch_loss_sum = 0.0
                #             for generation in range(GROUP_SIZE):
                #                 outputs = model(
                #                     input_ids=eval_batch["input_ids"],
                #                     attention_mask=eval_batch["attention_mask"],
                #                     labels=eval_batch["input_ids"])
                #                 batch_loss_sum += compute_loss(outputs.logits, eval_batch["input_ids"], eval_batch["attention_mask"])
                #             batch_avg_loss = batch_loss_sum / GROUP_SIZE
                #             eval_loss += batch_avg_loss.item()
                #             eval_steps += 1
                #             if eval_steps >= 2:  # adjust the number of eval batches as needed
                #                 break
                #     avg_eval_loss = eval_loss / eval_steps if eval_steps > 0 else float('inf')
                #     print(f"Step {training_step}: Evaluation average loss over {eval_steps} steps: {avg_eval_loss}")
                #     model.train()

                if pbar.n > self.MAX_ITER:
                        break
            
            print(f"Epoch {epoch + 1} completed")
            if pbar.n > self.MAX_ITER:
                break

        self.model.save_pretrained(save_directory)
        print("Model save successful")







