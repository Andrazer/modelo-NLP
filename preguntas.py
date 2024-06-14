import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cargar el modelo y el tokenizador
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Definir la pregunta
question = "¿Cuál es la capital de Francia?"

# Tokenizar la pregunta
input_ids = tokenizer.encode(question, return_tensors="pt")

# Generar una respuesta
with torch.no_grad():
    output = model.generate(input_ids=input_ids, max_length=50)

# Decodificar la respuesta
response = tokenizer.decode(output[0], skip_special_tokens=True)

print("Pregunta:", question)
print("Respuesta:", response)