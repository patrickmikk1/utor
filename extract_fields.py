import os
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

model_id = "microsoft/Phi-3-vision-128k-instruct"

# Load the model configuration and disable FlashAttention2
config = AutoConfig.from_pretrained(model_id)
config.use_flash_attention = False  # Explicitly disable FlashAttention2

# Load the model with the updated configuration
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Define the prompt and messages
prompt_text = "What is shown in this image?"
messages = [
    {"role": "user", "content": prompt_text},
    {"role": "assistant", "content": ""}
]

# Define the directory containing the image files
directory = os.path.expanduser('~/socialwork')

# Function to process and extract information from an image
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0" if torch.cuda.is_available() else "cpu")

    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]  # Remove input tokens
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response

# Loop through all files in the directory and process each image
for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        image_path = os.path.join(directory, filename)
        print(f"Processing {filename}...")
        response = process_image(image_path)
        print(f"Response for {filename}: {response}\n")
