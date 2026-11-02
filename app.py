from flask import Flask, request, jsonify, send_file
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

def download_audio(url, cookies_path=None):
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
        # YouTube: use android client to avoid 429 rate limiting
        "--extractor-args", "youtube:player_client=android,web",
        "--sleep-interval", "2",
        "--retries", "3",
        "--user-agent", "Mozilla/5.0 (Linux; Android 13; Redmi Note 12) AppleWebKit/537.36",
    ]
    if cookies_path:
        cmd.extend(["--cookies", cookies_path])
    cmd.append(url)
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
    # Support both JSON (url only) and multipart (url + cookies file)
    tmpdir = tempfile.mkdtemp()
    url = None
    cookies_path = None
    
    try:
        if request.is_json:
            data = request.get_json()
            url = data.get('url', '').strip()
        else:
            url = request.form.get('url', '').strip()
            if 'cookies' in request.files:
                cf = request.files['cookies']
                if cf.filename:
                    cookies_path = os.path.join(tmpdir, "cookies.txt")
                    cf.save(cookies_path)
                    print(f"Cookies saved", flush=True)
        
        if not url:
            return jsonify({"error": "URL required"}), 400
        
        print(f"Downloading: {url}", flush=True)
        audio_path = download_audio(url, cookies_path)
        print(f"Downloaded to {audio_path}", flush=True)
        
        return do_recap(audio_path)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)[:500]}), 500
    finally:
        # Cleanup cookies
        if cookies_path and os.path.exists(cookies_path):
            try: os.remove(cookies_path)
            except: pass

@app.route('/recap-file', methods=['POST'])
def recap_file():
    """Accept direct file upload (bypasses YouTube blocking)."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        
        tmpdir = tempfile.mkdtemp()
        # Save with original extension
        ext = os.path.splitext(f.filename)[1] or '.mp4'
        audio_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{ext}")
        f.save(audio_path)
        print(f"Uploaded to {audio_path}", flush=True)
        
        return do_recap(audio_path)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)[:500]}), 500

def do_recap(audio_path):
    """Transcribe and extract bullets from audio file."""
    try:
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

@app.route('/recap-video', methods=['POST'])
def recap_video():
    """Create a video recap with key scenes. Accepts file upload."""
    tmpdir = tempfile.mkdtemp()
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        
        ext = os.path.splitext(f.filename)[1] or '.mp4'
        video_path = os.path.join(tmpdir, f"input{ext}")
        f.save(video_path)
        print(f"Uploaded to {video_path}", flush=True)
        
        # Transcribe to get segments with timestamps
        print("Transcribing...", flush=True)
        segments, lang = transcribe(video_path)
        print(f"Transcribed {len(segments)} segments", flush=True)
        
        if not segments:
            return jsonify({"error": "No speech found"}), 400
        
        # Score segments (same logic as bullets, but keep timestamps)
        scored = []
        n = len(segments)
        for i, s in enumerate(segments):
            txt = s["text"].strip()
            if len(txt) < 15:
                continue
            score = len(txt)  # longer = more content
            # Prefer middle segments (avoid intro/outro)
            if n > 4 and (i < 2 or i > n - 3):
                score *= 0.5
            scored.append((score, s))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        # Target 6-10 minutes: ~25-30 scenes × ~18s avg = 7-9 min
        # For 2-hour video → 6-10 min recap
        num_scenes = min(30, len(scored))
        top = [s for _, s in scored[:num_scenes]]
        top.sort(key=lambda s: s["start"])  # chronological order
        
        # Extract clips
        clips = []
        for i, s in enumerate(top):
            start = max(0, s["start"] - 2)
            duration = min(20, s["end"] - s["start"] + 4)
            clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-t", str(duration),
                "-i", video_path,
                "-c", "copy",
                "-avoid_negative_ts", "1",
                clip_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
                clips.append(clip_path)
        
        if not clips:
            return jsonify({"error": "Could not extract clips"}), 500
        
        # Concatenate
        list_path = os.path.join(tmpdir, "list.txt")
        with open(list_path, 'w') as lf:
            for c in clips:
                lf.write(f"file '{c}'\n")
        
        output_path = os.path.join(tmpdir, "recap.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)
        
        if not os.path.exists(output_path):
            return jsonify({"error": "Failed to assemble video"}), 500
        
        print(f"Recap video ready: {output_path}", flush=True)
        return send_file(output_path, mimetype='video/mp4', as_attachment=True, download_name='recap.mp4')
    
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)[:500]}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
