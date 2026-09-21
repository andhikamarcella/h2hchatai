import os
from datetime import date
from flask import Flask, jsonify, render_template, request
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
    return Client(host="https://ollama.com", headers={"Authorization": f"Bearer {key}"}, timeout=100)

def compact_results(results):
    return [{"title": getattr(r, "title", "") or "", "url": getattr(r, "url", "") or "", "content": (getattr(r, "content", "") or "")[:1800]} for r in results.results]

@app.get("/")
def index():
    return render_template("index.html", model=MODEL)

@app.get("/api/health")
def health():
    try:
        models = get_model_client().list()
        return jsonify({"ok": True, "model": MODEL, "remote_models": [m.model for m in models.models]})
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

    sources, web_context = [], ""
    if use_web:
        try:
            search = get_search_client().web_search(query=f"{question} latest current {date.today().isoformat()} Hearts2Hearts K-pop", max_results=6)
            sources = compact_results(search)
            web_context = "\n\n".join(f"Judul: {x['title']}\nURL: {x['url']}\n{x['content']}" for x in sources)
        except Exception as exc:
            web_context = f"Pencarian web gagal ({exc}). Jangan berpura-pura telah melakukan pencarian."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Client sends prior turns only; append the current question exactly once.
    for item in history[-16:]:
        role, content = item.get("role"), str(item.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    current = question + ("\n\nKONTEKS WEB:\n" + web_context if web_context else "")
    messages.append({"role": "user", "content": current})
    try:
        result = get_model_client().chat(model=MODEL, messages=messages, stream=False)
        answer = (result.message.content or "").strip()
        return jsonify({"answer": answer, "sources": [{"title": x["title"], "url": x["url"]} for x in sources if x["title"] and x["url"]]})
    except Exception as exc:
        return jsonify({"error": f"Ollama gagal: {exc}"}), 502

# Vercel imports this module; local development can run it directly.
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
