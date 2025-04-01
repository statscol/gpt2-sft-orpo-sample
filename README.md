## Fine-tuning Decoder-models for Chat.


Sample scripts to run SFT or ORPO to allow chat-like interactions in a decoder-only model like GPT-2. Requires a Preference Dataset for ORPO (prompt-accepted-rejected) triplet per row, Or a Completion Dataset just for SFT (Prompt+Completion|Response)

- Preference dataset tested: [mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)
- Completion dataset tested: [mtimur/distill-gpt4-eng-chat](https://huggingface.co/datasets/mtimur/distill-gpt4-eng-chat)
- Base model fine-tuned: [openai-community/gpt2-large](https://huggingface.co/openai-community/gpt2-large)

# Setup
```bash
pip install uv && uv add -r requirements.txt #or use pip install -r requirements.txt
```

if using uv, add the --env-file=.env args

- Running ORPO Training, 

    `uv run --env-file=.env train_orpo.py`

    If using **DDP** run with accelerate `uv add accelerate` and run with `uv run --env-file=.env accelerate launch train_orpo.py`

- Running SFT Training

    `uv run --env-file=.env train_sft_peft.py`