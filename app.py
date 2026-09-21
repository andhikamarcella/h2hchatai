import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from datetime import date

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from ollama import Client

app = Flask(__name__)

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")

SYSTEM_PROMPT = """Kamu teman ngobrol sesama K-poper Indonesia, terutama mengikuti Hearts2Hearts. Jawab santai, natural, hangat, dan seperti fans asli yang ngobrol, bukan artikel atau chatbot formal. Hindari markdown seperti bold, heading, bullet, dan numbering kecuali diminta. Jangan memaksakan slang atau emoji. Jangan mengarang fakta. Untuk hal terkini, gunakan hasil web search yang disediakan dan jelaskan secara jujur jika hasilnya kurang. Anggap pengguna sudah mengenal H2H. Gunakan bahasa Indonesia kecuali diminta lain."""

def env_value(name):
    return os.getenv(name, "").strip()

def get_model_client():
    host = env_value("OLLAMA_HOST")
    if not host:
        raise RuntimeError("Environment Variable OLLAMA_HOST belum diset di Vercel.")
    return Client(host=host, timeout=280)

def get_search_client():
    key = env_value("OLLAMA_API_KEY")
    if not key:
        raise RuntimeError("Environment Variable OLLAMA_API_KEY belum diset di Vercel.")
    return key

def web_search_direct(query, max_results=4):
    """Call Ollama's official Web Search REST API directly."""
    key = get_search_client()
    payload = json.dumps({
        "query": query,
        "max_results": max_results,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://ollama.com/api/web_search",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "H2H-Chat/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama Web Search HTTP {exc.code}: {body[:500]}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama Web Search network error: {exc.reason}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama Web Search returned invalid JSON: {exc}")

    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"Format hasil Web Search tidak dikenali: {str(data)[:500]}")
    return [
        {
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or ""),
            "content": str(item.get("content") or "")[:900],
        }
        for item in results[:max_results]
        if isinstance(item, dict)
    ]

def sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

def keepalive():
    return ": keep-alive\n\n"

def message_field(message, name, default=""):
    if message is None:
        return default
    if isinstance(message, dict):
        return message.get(name, default) or default
    return getattr(message, name, default) or default

def response_done(chunk):
    if isinstance(chunk, dict):
        return bool(chunk.get("done", False))
    return bool(getattr(chunk, "done", False))

def stream_model_with_heartbeats(client, **kwargs):
    events = queue.Queue()

    def worker():
        try:
            stream = client.chat(**kwargs)
            for chunk in stream:
                events.put(("chunk", chunk))
        except Exception as exc:
            events.put(("error", exc))
        finally:
            events.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        try:
            kind, payload = events.get(timeout=4)
        except queue.Empty:
            yield None
            continue

        if kind == "chunk":
            yield payload
        elif kind == "error":
            raise payload
        else:
            break

@app.get("/")
def index():
    return render_template("index.html", model=MODEL)

@app.get("/api/health")
def health():
    try:
        models = get_model_client().list()
        names = [m.model for m in models.models]
        return jsonify({
            "ok": True,
            "model": MODEL,
            "remote_models": names,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502

@app.post("/api/chat")
def chat_api():
    data = request.get_json(silent=True) or {}
    question = str(data.get("message") or "").strip()
    history = data.get("history") or []
    use_web = bool(data.get("use_web", True))

    if not question:
        return jsonify({"error": "Pesan kosong."}), 400

    @stream_with_context
    def generate():
        try:
            sources = []
            web_context = ""

            yield sse({"type": "status", "text": "Menerima pertanyaan…"})
            yield sse({"type": "status", "text": f"Model · {MODEL}"})

            if use_web:
                yield sse({"type": "status", "text": "Mencari informasi terbaru di web…"})
                try:
                    sources = web_search_direct(
                        query=f"{question} latest current {date.today().isoformat()} Hearts2Hearts K-pop",
                        max_results=4,
                    )
                    yield sse({
                        "type": "status",
                        "text": f"Web search selesai · {len(sources)} sumber ditemukan",
                    })
                    if sources:
                        yield sse({
                            "type": "status",
                            "text": "Membaca hasil pencarian dan menyusun konteks…",
                        })
                    web_context = "\n\n".join(
                        f"Judul: {x['title']}\nURL: {x['url']}\n{x['content']}"
                        for x in sources
                    )
                except Exception as exc:
                    yield sse({
                        "type": "status",
                        "text": f"Web search gagal · {str(exc)[:220]}",
                    })
                    web_context = (
                        "Pencarian web gagal. Jangan mengklaim bahwa kamu telah "
                        "memverifikasi informasi lewat web."
                    )
            else:
                yield sse({"type": "status", "text": "Web search mati · memakai konteks chat"})

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for item in history[-12:]:
                role = item.get("role")
                content = str(item.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

            current = question
            if web_context:
                current += "\n\nKONTEKS WEB:\n" + web_context
            messages.append({"role": "user", "content": current})

            yield sse({"type": "status", "text": "Menganalisis konteks…"})
            yield sse({"type": "status", "text": "Menulis jawaban…"})
            yield keepalive()

            client = get_model_client()
            model_kwargs = {
                "model": MODEL,
                "messages": messages,
                "stream": True,
                "keep_alive": "5m",
                "options": {
                    "temperature": 0.7,
                    "num_predict": 640,
                },
            }

            # Generate the final answer directly. We do not request a separate
            # thinking stream here, so message.content is always the answer text.
            stream = stream_model_with_heartbeats(client, **model_kwargs)

            last_keepalive = time.monotonic()
            answer_parts = []
            for chunk in stream:
                if chunk is None:
                    now = time.monotonic()
                    if now - last_keepalive >= 4:
                        yield keepalive()
                        last_keepalive = now
                    continue

                message = message_field(chunk, "message", None)
                token = message_field(message, "content", "")

                if token:
                    answer_parts.append(token)
                    yield sse({"type": "token", "text": token})

                if response_done(chunk):
                    if not "".join(answer_parts).strip():
                        yield sse({"type": "status", "text": "Tidak ada teks dari stream · mencoba respons langsung…"})
                        fallback = client.chat(
                            model=MODEL,
                            messages=messages,
                            stream=False,
                            keep_alive="5m",
                            options={"temperature": 0.7, "num_predict": 640},
                        )
                        fallback_message = message_field(fallback, "message", None)
                        fallback_text = message_field(fallback_message, "content", "")
                        if fallback_text:
                            answer_parts.append(fallback_text)
                            yield sse({"type": "token", "text": fallback_text})
                        else:
                            yield sse({"type": "error", "error": "Gemma selesai tetapi tidak mengirim teks jawaban. Jalankan ollama run gemma4:26b di Kaggle untuk mengecek model."})
                            return

                    yield sse({
                        "type": "done",
                        "sources": [
                            {"title": x["title"], "url": x["url"]}
                            for x in sources
                            if x["title"] and x["url"]
                        ],
                    })
                    return

            if not "".join(answer_parts).strip():
                yield sse({"type": "status", "text": "Stream berakhir tanpa teks · mencoba respons langsung…"})
                fallback = client.chat(
                    model=MODEL,
                    messages=messages,
                    stream=False,
                    keep_alive="5m",
                    options={"temperature": 0.7, "num_predict": 640},
                )
                fallback_message = message_field(fallback, "message", None)
                fallback_text = message_field(fallback_message, "content", "")
                if fallback_text:
                    yield sse({"type": "token", "text": fallback_text})
                else:
                    yield sse({"type": "error", "error": "Gemma selesai tetapi tidak mengirim teks jawaban. Jalankan ollama run gemma4:26b di Kaggle untuk mengecek model."})
                    return

            yield sse({
                "type": "done",
                "sources": [
                    {"title": x["title"], "url": x["url"]}
                    for x in sources
                    if x["title"] and x["url"]
                ],
            })
        except Exception as exc:
            yield sse({"type": "error", "error": f"Ollama gagal: {exc}"})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
