from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import tempfile
import uuid

app = Flask(__name__)
CORS(app)  # Allow frontend from Vercel

# Lazy load model
_model = None
def get_model():
    global _model
    if _model is None:
        # Monkeypatch for av.open issue
        import av
        _orig = av.open
        def patched(*a, **k):
            k.pop('metadata_errors', None)
            return _orig(*a, **k)
        av.open = patched
        
        from faster_whisper import WhisperModel
        print("Loading faster-whisper base...", flush=True)
        _model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Model loaded!", flush=True)
    return _model

def download_audio(url):
    """Download audio from URL using yt-dlp, return path to audio file."""
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.m4a")
    cmd = [
        "yt-dlp",
        "-x",  # extract audio
        "--audio-format", "m4a",
        "--audio-quality", "0",
        "-o", out_path,
        "--no-playlist",
        "--max-filesize", "500M",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not os.path.exists(out_path):
        # Try without extension (yt-dlp may add it)
        for f in os.listdir(tmpdir):
            if f.endswith('.m4a'):
                return os.path.join(tmpdir, f)
        raise Exception(f"Download failed: {result.stderr[:500]}")
    return out_path

def transcribe(audio_path):
    """Transcribe audio, return segments."""
    model = get_model()
    segments, info = model.transcribe(audio_path, word_timestamps=False)
    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    return result, info.language

def extract_bullets(segments, max_bullets=7):
    """Simple extractive bullet points from transcript."""
    # Combine all text
    full_text = " ".join(s["text"] for s in segments)
    # Split into sentences (rough)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return []
    # Score: longer sentences + earlier position bonus
    scored = []
    for i, s in enumerate(sentences):
        score = len(s) - (i * 2)  # prefer earlier, longer
        scored.append((score, s))
    scored.sort(reverse=True)
    bullets = [s for _, s in scored[:max_bullets]]
    # Keep original order
    bullets.sort(key=lambda s: sentences.index(s))
    return bullets

@app.route('/recap', methods=['POST'])
def recap():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        if not url:
            return jsonify({"error": "URL required"}), 400
        
        print(f"Downloading: {url}", flush=True)
        audio_path = download_audio(url)
        print(f"Downloaded to {audio_path}", flush=True)
        
        print("Transcribing...", flush=True)
        segments, lang = transcribe(audio_path)
        print(f"Transcribed {len(segments)} segments, lang={lang}", flush=True)
        
        bullets = extract_bullets(segments)
        
        # Cleanup
        try:
            os.remove(audio_path)
            os.rmdir(os.path.dirname(audio_path))
        except:
            pass
        
        return jsonify({
            "language": lang,
            "segments": len(segments),
            "bullets": bullets,
            "transcript": " ".join(s["text"] for s in segments)[:2000]
        })
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)[:500]}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
