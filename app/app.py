import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, send_file
import soundfile as sf
import io

app = Flask(__name__)

# Diagnostic GPU
print(f"HSA_OVERRIDE_GFX_VERSION: {os.getenv('HSA_OVERRIDE_GFX_VERSION')}")
cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
print(f"Cuda (ROCm) disponible: {cuda_available}")
if cuda_available:
    print(f"Device nom: {torch.cuda.get_device_name(0)}")

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

print(f"Chargement de {model_id} sur {device}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # On utilise AutoModelForCausalLM car Qwen3-TTS est basé sur un backbone LLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if cuda_available else torch.float32,
        device_map="auto" if cuda_available else None
    )
    print("Succès : Modèle chargé !")
except Exception as e:
    print(f"ERREUR FATALE : {e}")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get("text", "Bonjour, ceci est un test de synthèse vocale sur architecture AMD.")

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        audio_values = model.generate(**inputs)

    audio_data = audio_values[0].cpu().float().numpy()

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, 24000, format='WAV')
    buffer.seek(0)

    return send_file(buffer, mimetype="audio/wav")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=48000)
