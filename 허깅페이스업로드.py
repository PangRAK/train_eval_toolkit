from huggingface_hub import login, upload_folder


login()


upload_folder(folder_path="/workspace/sangrak/unsloth/ckpts/pia_v1_3_1token_merged", repo_id="PIA-SPACE-LAB/PIA_AI2team_VQA_falldown_v1.3", repo_type="model")

# Auth: run `huggingface-cli login` or set the HF_TOKEN env var — never hardcode tokens here.