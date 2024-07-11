import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the model and processor
model_name = "microsoft/Phi-3-vision-128k-instruct-onnx-cuda"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Define the directory containing the .tif files
directory = '~/socialwork'

# Function to process and extract fields from an image
def extract_fields(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    fields = processor.decode(outputs[0], skip_special_tokens=True)
    return fields

# Loop through all .tif files in the directory and extract fields
results = {}
for filename in os.listdir(directory):
    if filename.endswith(".tif"):
        image_path = os.path.join(directory, filename)
        fields = extract_fields(image_path)
        results[filename] = fields
        print(f"Extracted fields from {filename}: {fields}")

# Optionally, save the results to a file
with open("extracted_fields.txt", "w") as f:
    for filename, fields in results.items():
        f.write(f"{filename}: {fields}\n")
