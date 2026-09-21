import json
import os
import urllib.error
import urllib.request
from datetime import date

from flask import Flask, jsonify, render_template, request
from ollama import Client

app = Flask(__name__)

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")

SYSTEM_PROMPT = """Kamu teman ngobrol sesama K-poper Indonesia, terutama mengikuti Hearts2Hearts. Jawab santai, natural, hangat, dan seperti fans asli yang ngobrol, bukan artikel atau chatbot formal. Hindari markdown seperti bold, heading, bullet, dan numbering kecuali diminta. Jangan memaksakan slang atau emoji. Jangan mengarang fakta. Untuk hal terkini, gunakan hasil web search yang disediakan dan jelaskan secara jujur jika hasilnya kurang. Anggap pengguna sudah mengenal H2H. Gunakan bahasa Indonesia kecuali diminta lain."""

def env_value(name):
    return os.getenv(name, "").strip()

def get_model_client():
    host = env_value("OLLAMA_HOST")
    if not host:
        raise RuntimeError("Environment Variable OLLAMA_HOST belum diset di Vercel.")
    return Client(host=host, timeout=280)

def get_search_key():
    key = env_value("OLLAMA_API_KEY")
    if not key:
        raise RuntimeError("Environment Variable OLLAMA_API_KEY belum diset di Vercel.")
    return key

def web_search_direct(query, max_results=4):
    key = get_search_key()
    payload = json.dumps({"query": query}).encode("utf-8")
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
            data = json.loads(response.read().decode("utf-8"))
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

def message_field(message, name, default=""):
    if message is None:
        return default
    if isinstance(message, dict):
        return message.get(name, default) or default
    return getattr(message, name, default) or default

@app.get("/")
def index():
    return render_template("index.html", model=MODEL)

@app.get("/api/health")
def health():
    try:
        models = get_model_client().list()
        return jsonify({
            "ok": True,
            "model": MODEL,
            "remote_models": [m.model for m in models.models],
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

    sources = []
    web_context = ""

    if use_web:
        try:
            primary_query = (
                f"{question} latest current {date.today().isoformat()} "
                "Hearts2Hearts K-pop"
            )
            sources = web_search_direct(primary_query, 4)
            if not sources:
                sources = web_search_direct(question, 4)

            web_context = "\n\n".join(
                f"Judul: {x['title']}\nURL: {x['url']}\n{x['content']}"
                for x in sources
            )
        except Exception:
            web_context = (
                "Pencarian web gagal. Jangan mengklaim bahwa kamu telah "
                "memverifikasi informasi lewat web."
            )

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

    try:
        result = get_model_client().chat(
            model=MODEL,
            messages=messages,
            stream=False,
            think=False,
            keep_alive=-1,
            options={
                "temperature": 0.7,
                "num_predict": 384,
                "num_ctx": 4096,
            },
        )
    except TypeError:
        result = get_model_client().chat(
            model=MODEL,
            messages=messages,
            stream=False,
            keep_alive=-1,
            options={
                "temperature": 0.7,
                "num_predict": 512,
                "num_ctx": 8192,
            },
        )
    except Exception as exc:
        return jsonify({"error": f"Ollama gagal: {exc}"}), 502

    message = message_field(result, "message", None)
    answer = message_field(message, "content", "").strip()

    if not answer:
        return jsonify({
            "error": (
                "Gemma selesai tetapi tidak mengirim teks jawaban. "
                "Periksa model Gemma di Kaggle."
            )
        }), 502

    return jsonify({
        "answer": answer,
        "sources": [
            {"title": x["title"], "url": x["url"]}
            for x in sources
            if x["title"] and x["url"]
        ],
    })

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
