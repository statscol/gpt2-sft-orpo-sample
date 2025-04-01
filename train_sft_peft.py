from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from datasets import load_dataset,Dataset,concatenate_datasets
from trl import SFTTrainer,SFTConfig, setup_chat_format, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
import torch
import wandb
from dotenv import load_dotenv

load_dotenv()

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

base_model="openai-community/gpt2-large"
dataset_name = "mtimur/distill-gpt4-eng-chat "
new_model="gpt2-large-sft-chat-eng"

dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=434)

run= wandb.init(name=f"{new_model}-001")


model = AutoModelForCausalLM.from_pretrained(base_model,quantization_config=bnb_config,torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="flash_attention_2")
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.padding_side="right"
tokenizer.pad_token = tokenizer.eos_token


model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)
model.config.use_cache=False


def valid_len_pair(row):
    """make sure we set True when the converesation has less than 1024 tokens"""
    row['len_q_a']=(len(tokenizer(row['request'])['input_ids']) + len(tokenizer(row['response'])['input_ids']))<1024
    return row

#if the dataset does not have a default system prompt we can add it
sys_prompt="You're a helpful assistant, be kind and answer user questions accordingly"

#alignment with dataset formats https://github.com/huggingface/trl/blob/26d86757a7c7e24e397ea44f57ecce6031dfac01/trl/data_utils.py#L171

def format_conversational(instance):
    return {"messages":[{'role':'system','content':sys_prompt},{'role':'user','content':instance['request']},{'role':'assistant','content':instance['response']}]}

original_columns = dataset.column_names
dataset=dataset.map(format_conversational,remove_columns=original_columns).train_test_split(0.2)

#can take a while in 400k rows
dataset=dataset.map(valid_len_pair)
dataset=dataset.filter(lambda example: example['len_q_a'])
original_columns = dataset.column_names
dataset=dataset.map(format_conversational,remove_columns=original_columns)

print("Conversation structure: \n",dataset['messages'][0])
dataset=dataset.train_test_split(0.2)

print(dataset)

instruction_template = "<|im_start|>user\n"
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False,padding_free=True)

sft_config=SFTConfig(
    learning_rate=1e-6,
    lr_scheduler_type="linear",
    packing=False,
    max_seq_length=1024,
    per_device_train_batch_size=22,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=20,
    eval_strategy="steps",
    save_steps=300,
    bf16=True,
    logging_steps=300,
    run_name="gpt2-large-sft-chat-eng",
    load_best_model_at_end=True,
    warmup_steps=10,
    report_to="wandb", 
    output_dir="./sft_training_gpt",
)

trainer = SFTTrainer(
    model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collator,
    processing_class=tokenizer,
    peft_config=peft_config
)

if __name__=="__main__":


    
    trainer.train()
    trainer.save_model(new_model)
    
    
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