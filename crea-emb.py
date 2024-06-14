from transformers import AutoTokenizer, AutoModel
import torch

# Especificar el nombre del modelo Mistral en Hugging Face Model Hub
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Asegúrate de que este es el nombre correcto del modelo Mistral

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Establecer el pad_token como eos_token si no está definido
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Leer el archivo de texto
file_path = "pdf/icp.txt"  # Reemplaza con la ruta a tu archivo .txt
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Tokenizar el texto
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Obtener los embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

# Guardar los embeddings en un archivo .pt
output_path = "embeddings.pt"  # Reemplaza con la ruta donde deseas guardar el archivo
torch.save(embeddings, output_path)

print(f"Embeddings guardados en {output_path}")
