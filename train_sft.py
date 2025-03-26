from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
import torch
import os
from datasets import load_dataset
from trl import setup_chat_format
from transformers import Trainer, TrainingArguments,DataCollatorForLanguageModeling
from dotenv import load_dotenv
import wandb

load_dotenv()
run= wandb.init(name="gpt2-chat-spanish-001")


base_model="flax-community/gpt-2-spanish"

#load base model and dataset
dataset_name = "Kukedlc/dpo-orpo-spanish-15k"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=434)


#load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model,device_map="auto")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map='auto')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="left"

#add chatml tokens and modify embeddings
model, tokenizer = setup_chat_format(model, tokenizer)
model.resize_token_embeddings(len(tokenizer))

#data collator for batches of different size
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) ##causal language modeling (mlm=False)

#process data

def tokenize(instance):
    outputs = tokenizer(
        instance['final_text'],
        truncation=True,
        max_length=1024
    )
    return outputs

def valid_len_pair(row):

    row['len_q_a']=(len(row['chosen']) + len(row['prompt']) + len(row['system']))<1024
    return row

def apply_chat_template(instance):
  #make sure this is run in batches
  instance['final_text']=tokenizer.apply_chat_template([{'role':'system','content':instance['system']},{'role':'user','content':instance['prompt']},{'role':'assistant','content':instance['chosen']}],tokenize=False,truncation=True)
  return instance


dataset=dataset.map(valid_len_pair)
dataset=dataset.filter(lambda example: example['len_q_a'])
dataset=dataset.map(apply_chat_template)
original_columns = dataset.column_names

# Applying formatting to the dataset
dataset = dataset.map(
    tokenize,
    remove_columns=original_columns,
    batched=False
).train_test_split(test_size=0.1)


args = TrainingArguments(
    output_dir="orpo_training",
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=500,
    gradient_accumulation_steps=2,
    num_train_epochs=500,
    weight_decay=0.1,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    learning_rate=1e-6,
    save_steps=600,
    fp16=True,
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="gpt2-chat-sft"
)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    callbacks = [EarlyStoppingCallback(early_stopping_patience=20)]
)

if __name__=="__main__":
    new_model="gpt2-chat-spanish-sft"
    trainer.train()
    trainer.model.push_to_hub(new_model)
    tokenizer.push_to_hub(new_model)


