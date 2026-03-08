import torch
import io
import soundfile as sf
from flask import Flask, request, send_file
from transformers import  AutoModel, AutoTokenizer

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
#model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

print(f"Chargement du modèle sur {device} (gfx1150)...")

# 1. Charger la config en autorisant le code distant
#config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

try:
    print("Tentative de chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print("Tentative de chargement du modèle (ceci peut prendre du temps)...")
    # On évite AutoConfig et on laisse AutoModel tout gérer avec trust_remote_code
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement : {e}")
    # Plan B : Si AutoModel échoue encore, on affiche les fichiers du repo pour debug
    import os
    print(f"Fichiers dans le cache : {os.listdir(os.environ.get('TRANSFORMERS_CACHE', '.'))}")

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
