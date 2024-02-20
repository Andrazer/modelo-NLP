from transformers import convert_to_onnx
from transformers import AutoModelForCausalLM

# Cargar el modelo entrenado
model = AutoModelForCausalLM.from_pretrained("modelo/modelo")

# Convertir el modelo a ONNX
convert_to_onnx(model, "modelo.onnx")
