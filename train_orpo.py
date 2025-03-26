from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback,BitsAndBytesConfig,pipeline
import torch
import os
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import DPOTrainer,ORPOConfig, ORPOTrainer, setup_chat_format
import bitsandbytes as bnb
import wandb
from dotenv import load_dotenv

load_dotenv()
run= wandb.init(name="gpt2-chat-orpo-spanish-001")

base_model="flax-community/gpt-2-spanish"
dataset_name = "Kukedlc/dpo-orpo-spanish-15k"
#load base model and dataset

dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=434)

def valid_len_pair(row):

    row['len_q_a']=(len(row['chosen']) + len(row['prompt']) + len(row['system']))<1024
    return row


dataset=dataset.map(valid_len_pair)
dataset=dataset.filter(lambda example: example['len_q_a'])


def format_preference_df(row):
    
    row['chosen']=row["chosen"][:2]
    row['rejected']=row["rejected"][:2]
    return row


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="left"

# Load model

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
).half()

model, tokenizer = setup_chat_format(model, tokenizer)

#Process dataset 

dataset = dataset.map(
    format_preference_df,
    remove_columns=['question','prompt','len_q_a'],
    num_proc= os.cpu_count(),
).train_test_split(test_size=0.1)


#make sure we modify the params with new added tokens for chat template
model.resize_token_embeddings(len(tokenizer))

orpo_args = ORPOConfig(
    learning_rate=1e-6,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    beta=0.1,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=30,
    eval_strategy="steps",
    save_steps=300,
    logging_steps=300,
    run_name="gpt2-orpo-chat",
    load_best_model_at_end=True,
    warmup_steps=30,
    report_to="wandb",
    output_dir="./orpo_training_gpt",
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

if __name__=="__main__":

    new_model="gpt2-chat-spanish-orpo"
    trainer.train()
    trainer.save_model(new_model)

    import gc

    del trainer,model
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        new_model,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )
    model.push_to_hub(new_model)
    tokenizer.push_to_hub(new_model)

