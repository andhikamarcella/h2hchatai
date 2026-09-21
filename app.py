import json
import os
import queue
import threading
import time
from datetime import date

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from ollama import Client

app = Flask(__name__)

FAST_MODEL = os.getenv("FAST_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:4b"))
THINKING_MODEL = os.getenv("THINKING_MODEL", "gemma4:26b")

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
    return Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {key}"},
        timeout=100,
    )

def compact_results(results):
    return [
        {
            "title": getattr(r, "title", "") or "",
            "url": getattr(r, "url", "") or "",
            "content": (getattr(r, "content", "") or "")[:1000],
        }
        for r in results.results
    ]

def sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

def keepalive():
    # SSE comment: invisible to the client, but sends bytes through proxies.
    return ": keep-alive\n\n"

def model_for_mode(mode):
    return THINKING_MODEL if mode == "thinking" else FAST_MODEL

def stream_model_with_heartbeats(client, **kwargs):
    """Run model streaming in a worker so the HTTP response can send heartbeats."""
    events = queue.Queue()
    done = object()

    def worker():
        try:
            stream = client.chat(**kwargs)
            for chunk in stream:
                events.put(("chunk", chunk))
        except Exception as exc:
            events.put(("error", exc))
        finally:
            events.put(("done", done))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        try:
            kind, payload = events.get(timeout=4)
        except queue.Empty:
            # Keeps Cloudflare's proxy connection active while a slow model is generating.
            yield None
            continue

        if kind == "chunk":
            yield payload
        elif kind == "error":
            raise payload
        elif kind == "done":
            break

@app.get("/")
def index():
    return render_template(
        "index.html",
        fast_model=FAST_MODEL,
        thinking_model=THINKING_MODEL,
    )

@app.get("/api/health")
def health():
    try:
        models = get_model_client().list()
        names = [m.model for m in models.models]
        return jsonify({
            "ok": True,
            "fast_model": FAST_MODEL,
            "thinking_model": THINKING_MODEL,
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
    mode = str(data.get("mode") or "fast").lower()
    if mode not in {"fast", "thinking"}:
        mode = "fast"

    if not question:
        return jsonify({"error": "Pesan kosong."}), 400

    model = model_for_mode(mode)

    @stream_with_context
    def generate():
        try:
            sources, web_context = [], ""

            yield sse({
                "type": "status",
                "text": f"Mode {('thinking' if mode == 'thinking' else 'cepat')} · {model}",
            })

            if use_web:
                yield sse({"type": "status", "text": "Membuka pencarian web…"})
                try:
                    search = get_search_client().web_search(
                        query=f"{question} latest current {date.today().isoformat()} Hearts2Hearts K-pop",
                        max_results=4,
                    )
                    sources = compact_results(search)
                    yield sse({
                        "type": "status",
                        "text": f"Web search selesai · {len(sources)} sumber ditemukan",
                    })
                    if sources:
                        yield sse({
                            "type": "status",
                            "text": f"Membaca {min(len(sources), 6)} sumber dan menyusun konteks…",
                        })
                    web_context = "\n\n".join(
                        f"Judul: {x['title']}\nURL: {x['url']}\n{x['content']}"
                        for x in sources
                    )
                except Exception as exc:
                    yield sse({
                        "type": "status",
                        "text": "Web search gagal · lanjut memakai percakapan",
                    })
                    web_context = (
                        f"Pencarian web gagal ({exc}). "
                        "Jangan berpura-pura telah melakukan pencarian."
                    )
            else:
                yield sse({"type": "status", "text": "Web search mati · memakai konteks chat"})

            system_prompt = SYSTEM_PROMPT
            # Gemma 4 documents thinking via the <|think|> control token.
            if mode == "thinking":
                system_prompt = "<|think|>\n" + system_prompt

            messages = [{"role": "system", "content": system_prompt}]
            for item in history[-12:]:
                role = item.get("role")
                content = str(item.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

            current = question + (
                "\n\nKONTEKS WEB:\n" + web_context if web_context else ""
            )
            messages.append({"role": "user", "content": current})

            yield sse({
                "type": "status",
                "text": (
                    "Menganalisis pertanyaan dan konteks…"
                    if mode == "thinking"
                    else "Menyiapkan jawaban cepat…"
                ),
            })

            client = get_model_client()
            kwargs = {
                "model": model,
                "messages": messages,
                "stream": True,
                "keep_alive": "5m",
                "options": {
                    "temperature": 0.7,
                    "num_predict": 384 if mode == "fast" else 640,
                },
            }

            stream = stream_model_with_heartbeats(client, **kwargs)

            # Send an initial chunk so proxies start flushing the SSE response.
            yield sse({"type": "status", "text": "Model mulai bekerja…"})
            yield keepalive()
            last_heartbeat = time.monotonic()

            finished = False
            for chunk in stream:
                if chunk is None:
                    now = time.monotonic()
                    if now - last_heartbeat >= 4:
                        yield keepalive()
                        last_heartbeat = now
                    continue

                token = getattr(getattr(chunk, "message", None), "content", "") or ""
                if token:
                    yield sse({"type": "token", "text": token})

                # Ollama marks the final streamed response with done=True.
                # Signal completion immediately instead of waiting for the
                # generator/HTTP connection to close.
                if bool(getattr(chunk, "done", False)):
                    finished = True
                    yield sse({
                        "type": "done",
                        "sources": [
                            {"title": x["title"], "url": x["url"]}
                            for x in sources
                            if x["title"] and x["url"]
                        ],
                    })
                    break

            if not finished:
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
