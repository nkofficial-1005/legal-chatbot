import os
import sys
import tempfile

def get_writable_cache_dir():
    # Use the repository's cache directory (which you created and committed)
    repo_cache = os.path.join(os.getcwd(), "cache")
    try:
        os.makedirs(repo_cache, exist_ok=True)
        # Test writability by creating a temporary file
        test_path = os.path.join(repo_cache, "test.txt")
        with open(test_path, "w") as f:
            f.write("test")
        os.remove(test_path)
        return repo_cache
    except Exception as e:
        print("WARNING: Repository cache directory not writable. Falling back to /tmp.", file=sys.stderr)
        tmp_cache = os.path.join(tempfile.gettempdir(), "cache")
        os.makedirs(tmp_cache, exist_ok=True)
        return tmp_cache

# Choose the cache directory (either the repository's or fallback to /tmp)
cache_dir = get_writable_cache_dir()

# Set environment variables for Hugging Face libraries
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["XDG_CACHE_HOME"] = cache_dir
# Ensure HOME points to a writable directory (here, the repository's root)
os.environ["HOME"] = os.getcwd()

# Ensure the transformers cache directory exists
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Model and tokenizer details
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    prompt = request.message
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
