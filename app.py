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
os.environ["HOME"] = os.getcwd()

# Ensure the transformers cache directory exists
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model and tokenizer details
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

def generate_response(prompt: str) -> str:
    """
    Given a prompt, generate a response from the model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# --- Gradio UI Setup ---
import gradio as gr  # Ensure 'gradio' is added to your requirements

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your message", placeholder="Type your message here..."),
    outputs=gr.Textbox(label="Response"),
    title="Legal Chatbot",
    description="Enter a message to receive legal advice powered by Microsoft phi-2.",
    flagging="never"  # Disables flagging to prevent directory permission issues
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
