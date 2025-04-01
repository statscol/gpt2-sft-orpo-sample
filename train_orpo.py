from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback,BitsAndBytesConfig,pipeline
import torch
import os
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
import bitsandbytes as bnb
import wandb
from dotenv import load_dotenv

load_dotenv()
run= wandb.init(name="gpt2-chat-orpo-spanish-001")


base_model="openai-community/gpt2-large"
dataset_name = "mlabonne/orpo-dpo-mix-40k"
new_model="gpt2-large-chat-eng-orpo"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)


#load preference dataset and base model

dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=434)

def valid_len_pair(row):
    """make sure a preference dataset row has less than 1024 tokens"""

    #a preference dataset can have multi-turn conversations, we'll only use the first pair of user,assistant.
    row['len_q_a']=len(tokenizer.apply_chat_template(row['chosen'][:2])['input_ids'])<1024
    return row


dataset=dataset.map(valid_len_pair)
dataset=dataset.filter(lambda example: example['len_q_a'])


def format_preference_df(row):
    
    row['chosen']=row["chosen"][:2]
    row['rejected']=row["rejected"][:2]
    return row


# Load tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.padding_side="right"
tokenizer.pad_token = tokenizer.eos_token

# Load model & Tokenizer
if tokenizer.chat_template is None:
    model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)
model.config.use_cache=False

#Process dataset 

dataset=dataset.map(valid_len_pair)
dataset = dataset.map(
    format_preference_df,
    remove_columns=['question','prompt','len_q_a'],
    num_proc= os.cpu_count(),
).train_test_split(test_size=0.1)


#double check to make sure we modify the params with new added tokens for chat template, even though this is made by setup_chat_format
model.resize_token_embeddings(len(tokenizer))

orpo_args = ORPOConfig(
    learning_rate=1e-6,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    beta=0.1, #adamw param
    per_device_train_batch_size=12,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=30,
    eval_strategy="steps",
    save_steps=600,
    logging_steps=300,
    run_name="gpt2-orpo-chat",
    load_best_model_at_end=True,
    warmup_steps=50,
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

    
    import gc
    del trainer,model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Merge adapter with base model
    model = PeftModel.from_pretrained(model, new_model)
    model = model.merge_and_unload()

    model.push_to_hub(new_model)
    tokenizer.push_to_hub(new_model)

