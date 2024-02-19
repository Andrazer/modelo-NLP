import torch

# Verifica la disponibilidad de CUDA
if torch.cuda.is_available():
    print("¡Tu GPU está disponible!")
    # Obtiene información sobre la GPU
    device = torch.cuda.current_device()
    print(f"Nombre del dispositivo: {torch.cuda.get_device_name(device)}")
    print(f"Memoria total: {torch.cuda.get_device_properties(device).total_memory}")
    
    # Ejemplo de operación en GPU
    x = torch.randn(1000, 1000, device=device)
    y = torch.mm(x, x)
    print(f"Resultado de la multiplicación de matrices: {y.size()}")
else:
    print("No se ha detectado una GPU.")