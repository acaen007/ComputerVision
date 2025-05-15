from huggingface_hub import HfApi, create_repo, upload_folder

repo_name = "detr-fashionmnist"
hf_username = "acaen"  # ← replace with your actual HF username
repo_id = f"{hf_username}/{repo_name}"

# Create repo if it doesn't exist
create_repo(repo_id, exist_ok=True)

# Upload your folder to the model hub
upload_folder(
    folder_path="ANDY/detr_fashionmnist",  # <- Correct relative path from project root
    path_in_repo=".",
    repo_id=repo_id
)


print(f"Model uploaded to: https://huggingface.co/{repo_id}")
