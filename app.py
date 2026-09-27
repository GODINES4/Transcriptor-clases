"""
Transcriptor de clases — versión hospedada (Render).
Pega un clip ID o URL de YouTube, Vimeo, Loom, Hotmart o Skool.

Primero intenta usar los subtítulos que el propio video ya trae (gratis, ~2s).
Si no los tiene, y hay una GROQ_API_KEY configurada, transcribe el audio con
la API de Groq (gratis, muy rápida). Sin esa clave, avisa que no hay subtítulos.

Nota: por las IPs compartidas de los hostings gratuitos, YouTube suele bloquear
las descargas — con Vimeo, Loom, Hotmart y Skool no hay ese problema.
"""
import os
import re
import tempfile
import threading
import uuid

from flask import Flask, jsonify, request, Response
import yt_dlp

app = Flask(__name__)

GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()
jobs = {}  # job_id -> {status, progress, segments, error, title, done}


# ---------- ID -> URL ----------
def build_url(platform: str, value: str):
    v = value.strip()
    if v.startswith("http"):
        referer = "https://www.skool.com/" if platform in ("vimeo", "hotmart", "skool") else None
        return v, referer

    # Un ID solo numérico (o numérico/hash) siempre es de Vimeo, aunque venga de Skool
    if re.fullmatch(r"\d{6,}(/[0-9a-f]+)?", v, re.I):
        platform = "vimeo"

    if platform == "youtube":
        return f"https://www.youtube.com/watch?v={v}", None
    if platform == "vimeo":
        if "/" in v:
            vid, h = v.split("/", 1)
            return f"https://player.vimeo.com/video/{vid}?h={h}", "https://www.skool.com/"
        return f"https://player.vimeo.com/video/{v}", "https://www.skool.com/"
    if platform == "loom":
        return f"https://www.loom.com/share/{v}", None
    if platform == "hotmart":
        return f"https://player.hotmart.com/embed/{v}", "https://www.skool.com/"
    if platform == "skool":
        if re.fullmatch(r"[a-f0-9]{32}", v, re.I):
            return f"https://stream.video.skool.com/{v}.m3u8", "https://www.skool.com/"
        return v, "https://www.skool.com/"
    raise ValueError("Plataforma no reconocida")


# ---------- Subtítulos nativos del video ----------
def fmt_ts(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_vtt(text):
    segs, last = [], None
    for b in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = b.strip().split("\n")
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        start = lines[idx].split("-->")[0].strip()
        parts = [float(p) for p in start.replace(",", ".").split(":")]
        secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
        body = " ".join(re.sub(r"<[^>]+>", "", l).strip() for l in lines[idx + 1:]).strip()
        if body and body != last:
            segs.append({"t": fmt_ts(secs), "text": body})
            last = body
    return segs


def try_subtitles(ydl, info, language):
    pools = [info.get("subtitles") or {}, info.get("automatic_captions") or {}]
    wanted = ([language] if language else []) + ["es", "es-419", "es-MX", "en", "pt"]
    for pool in pools:
        keys = list(pool.keys())
        pick = next((k for w in wanted for k in keys if k == w or k.startswith(w + "-")), None)
        pick = pick or (keys[0] if keys and pool is pools[0] else None)
        if not pick:
            continue
        track = next((f for f in pool[pick] if f.get("ext") == "vtt"), None)
        if not track:
            continue
        raw = ydl.urlopen(track["url"]).read().decode("utf-8", "ignore")
        segs = parse_vtt(raw)
        if segs:
            return segs
    return None


# ---------- Respaldo: transcribir audio con Groq ----------
def groq_transcribe(audio_path, language):
    import requests
    with open(audio_path, "rb") as f:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            files={"file": (os.path.basename(audio_path), f)},
            data={"model": "whisper-large-v3-turbo", "response_format": "verbose_json",
                  **({"language": language} if language else {})},
            timeout=600,
        )
    r.raise_for_status()
    return [{"t": fmt_ts(s["start"]), "text": s["text"].strip()} for s in r.json().get("segments", [])]


# ---------- Trabajo principal ----------
def run_job(job_id, platform, value, language, cookies_browser):
    job = jobs[job_id]
    try:
        url, referer = build_url(platform, value)
        job.update(status="Buscando subtítulos del video…", progress=15)

        base = {"quiet": True, "noprogress": True}
        if referer:
            base["http_headers"] = {"Referer": referer}
        if cookies_browser:
            base["cookiesfrombrowser"] = (cookies_browser,)

        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
            job["title"] = info.get("title") or value
            segs = try_subtitles(ydl, info, language)

        if segs:
            job.update(segments=segs, status="Listo (subtítulos del video)", progress=100, done=True)
            return

        if not GROQ_KEY:
            job.update(
                status="Sin subtítulos",
                error="Este video no trae subtítulos propios y no hay una clave de Groq configurada "
                      "para transcribir el audio. Configura GROQ_API_KEY en Render para activar ese respaldo.",
                done=True,
            )
            return

        job.update(status="No hay subtítulos. Descargando audio…", progress=35)
        with tempfile.TemporaryDirectory(prefix="skooltx_") as tmp:
            opts = dict(base)
            opts.update({
                "format": "bestaudio/best",
                "outtmpl": os.path.join(tmp, "audio.%(ext)s"),
            })
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
            audio = next((os.path.join(tmp, f) for f in os.listdir(tmp) if f.startswith("audio")), None)
            if not audio:
                raise RuntimeError("No se pudo obtener el audio.")

            job.update(status="Transcribiendo con Groq…", progress=65)
            job.update(segments=groq_transcribe(audio, language),
                       status="Listo (Groq)", progress=100, done=True)

    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "403" in msg or "login" in low or "private" in low:
            msg += "\n\nPista: activa 'Usar sesión de mi navegador' y ten la sesión iniciada en esa plataforma."
        if "sign in" in low or "confirm you" in low or "captcha" in low:
            msg += "\n\nYouTube suele bloquear las descargas desde hostings gratuitos como este."
        if "drm" in low:
            msg += "\n\nEste video tiene protección DRM y no se puede transcribir."
        job.update(status="Error", error=msg, done=True)


@app.post("/api/transcribe")
def transcribe():
    data = request.get_json(force=True)
    value = (data.get("value") or "").strip()
    if not value:
        return jsonify(error="Pega un ID o una URL."), 400
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "En cola…", "progress": 0, "segments": [], "done": False}
    threading.Thread(
        target=run_job,
        args=(job_id, data.get("platform", "youtube"), value,
              data.get("language", ""), data.get("cookies", "")),
        daemon=True,
    ).start()
    return jsonify(job_id=job_id)


@app.get("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="No existe ese trabajo."), 404
    return jsonify(job)


@app.get("/")
def index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
