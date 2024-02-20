import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Selección de dispositivo (usar CPU si es necesario)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Ajustar a un modelo más pequeño si persisten las limitaciones de memoria
# Los modelos más pequeños también pueden lograr un buen rendimiento:
# https://huggingface.co/models?filter=causal-lm
# model_name = "EleutherAI/gpt-j-6B"
model_name = "facebook/bart-base"

# Cargar modelo y tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Cargar el modelo y el tokenizador
#model = AutoModelForCausalLM.from_pretrained("modelo/modelo")
#tokenizer = AutoTokenizer.from_pretrained("modelo/modelo")


model.to(device)  # Mover el modelo al dispositivo

# Conjunto de datos y DataLoader con tamaño de lote más pequeño y truncamiento explícito
class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):  # Ajustar max_length según sea necesario
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data_path = "pdf/1-3.txt"
dataset = MyDataset(data_path, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Experimentar con el tamaño de lote

# Función de entrenamiento con acumulación de gradiente y puntos de control (opcional)
def train_model(model, dataloader, optimizer, device, accumulation_steps=2, save_freq=1000):
    model.train()
    total_loss = 0
    step = 0
    # Considerar la creación de puntos de control para evitar perder progreso si hay problemas de memoria
    for batch in tqdm(dataloader, desc="Training"):
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # Acumulación de gradiente
        loss = loss / accumulation_steps
        loss.backward()
        step += 1

        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            # Puntos de control opcionales para evitar el agotamiento de la memoria
            if step % save_freq == 0:
                model.save_pretrained(f"modelo/checkpoint_{step}")

    return total_loss / len(dataloader)

# Parámetros de entrenamiento
epochs = 3
learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Entrenar en chunks más pequeños si persisten problemas de memoria
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    avg_loss = train_model(model, dataloader, optimizer, device)
