## Convert Model to GGUF

- Instructions were based on [this](https://github.com/ggml-org/llama.cpp/discussions/2948) Discussion post in the llama.cpp repo 

- Download your model:

    ```python
    from huggingface_hub import snapshot_download
    model_id="<MODEL_REPO_ID>"
    snapshot_download(repo_id=model_id, local_dir="<YOUR_MODEL_PATH>", revision="main")
    ```


- Clone the llama.cpp repo and install requirements.txt

    ```bash
    git clone https://github.com/ggerganov/llama.cpp.git

    ```

- Convert Model, you can use any quantization type. E.g 'q8_0' or 'q4_k_m' 


    ```bash
    python llama.cpp/convert_hf_to_gguf.py <YOUR_MODEL_PATH>   --outfile <YOUR_MODEL_NAME>.gguf  --outtype q8_0
    ```

- Upload your model to HF Hub, now you can use it in common apps like LM Studio or in llama.cpp directly.

