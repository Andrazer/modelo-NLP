import torch
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


#device = "cuda"
#device = "cpu"

# Definir una clase para tu conjunto de datos personalizado
class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Leer tus datos y procesarlos
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

# Ruta a tus datos
data_path = "pdf/test.txt"

# Tokenizador y modelo preentrenado
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Dataset y DataLoader
dataset = MyDataset(data_path, tokenizer)
#dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Función de entrenamiento
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Parámetros de entrenamiento
epochs = 3
learning_rate = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Entrenamiento
model.to(device)
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    avg_loss = train_model(model, dataloader, optimizer, device)
    print(f"Average Loss: {avg_loss:.4f}")

# Guardar el modelo entrenado
model.save_pretrained("modelo/modelo")
tokenizer.save_pretrained("modelo/tokenizador")
