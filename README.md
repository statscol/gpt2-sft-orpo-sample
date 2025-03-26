## Fine-tuning Decoder-models for Chat.


Sample scripts to run ORPO or Next best sentence which will fine-tune a model to allow chat-like interactions.


# Setup
```bash
pip install uv && uv add -r requirements.txt #or use pip install -r requirements.txt
```

if using uv, add the --env-file=.env args

- Running ORPO Training, 

    `uv run --env-file=.env train_orpo.py`

- Running SFT Training

    `uv run run --env-file=.env train_sft.py`