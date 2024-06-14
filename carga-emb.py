import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def load_embeddings(embeddings_path):
    """Loads embeddings from a file."""
    try:
        embeddings = torch.load(embeddings_path)
        print("Embeddings loaded successfully.")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise

def generate_text(embeddings, model, tokenizer, max_length=4500):
    """Genera texto a partir de embeddings utilizando el modelo y el tokenizer proporcionados."""
    try:
        # Generar texto iterativamente a partir de los embeddings
        generated_ids = []
        past = None  # Inicializamos past como None

        for i in range(max_length):
            with torch.no_grad():
                # Generar salida del modelo
                outputs = model(inputs_embeds=embeddings, past_key_values=past)

                # Obtener logits y past para el siguiente paso
                logits = outputs.logits[:, -1, :]  # Tomar los logits del último token generado
                past = outputs.past_key_values

                # Samplear el siguiente token
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)

                # Añadir el nuevo token a la lista de tokens generados
                generated_ids.append(next_token_id)

                # Si el token generado es el token de fin de secuencia (EOS), terminar la generación
                if next_token_id == tokenizer.eos_token_id:
                    break

        # Decodificar los tokens generados
        generated_text = tokenizer.decode(torch.cat(generated_ids, dim=-1).squeeze(), skip_special_tokens=True)

        # Imprimir el texto generado
        print("Generated Text:", generated_text)
        
    except Exception as e:
        print(f"Error generating text: {e}")


if __name__ == "__main__":
    # Cargar embeddings y modelo/tokenizer Mistral
    embeddings_path = "embeddings.pt"
    embeddings = torch.load(embeddings_path)
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Generar texto a partir de los embeddings
    generate_text(embeddings, model, tokenizer)
