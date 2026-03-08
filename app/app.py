import torch
import os

print("--- DIAGNOSTIC ROCM ---")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("ALERTE: ROCm ne détecte pas de GPU utilisable.")
print("-----------------------")
