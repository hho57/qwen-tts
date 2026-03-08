import torch
import io
import soundfile as sf
from flask import Flask, request, send_file
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

print(f"Chargement du modèle sur {device} (gfx1150)...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

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
