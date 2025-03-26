## Fine-tuning Decoder-models for Chat.


Sample scripts to run ORPO or Next best sentence which will fine-tune a model to allow chat-like interactions. Requires a Preference Dataset (prompt-accepted-rejected) triplet per row.

- Sample data used: [Kukedlc/dpo-orpo-spanish-15k](https://huggingface.co/datasets/Kukedlc/dpo-orpo-spanish-15k)
- Base model fine-tuned: [flax-community/gpt-2-spanish](https://huggingface.co/flax-community/gpt-2-spanish)

# Setup
```bash
pip install uv && uv add -r requirements.txt #or use pip install -r requirements.txt
```

if using uv, add the --env-file=.env args

- Running ORPO Training, 

    `uv run --env-file=.env train_orpo.py`

    If using **DDP** run with accelerate `uv add accelerate` and run with `uv run --env-file=.env accelerate launch train_orpo.py`

- Running SFT Training

    `uv run run --env-file=.env train_sft.py`