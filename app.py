import os
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from ollama import Client

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
REMOTE_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()

if not REMOTE_OLLAMA_HOST:
    raise RuntimeError(
        "OLLAMA_HOST belum diset. Contoh: "
        "set OLLAMA_HOST=https://xxxx.trycloudflare.com"
    )

if not OLLAMA_API_KEY:
    raise RuntimeError(
        "OLLAMA_API_KEY belum diset. Contoh: "
        'setx OLLAMA_API_KEY "ollama_xxxxx"'
    )

model_client = Client(host=REMOTE_OLLAMA_HOST, timeout=600)
search_client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    timeout=120,
)

app = Flask(__name__)

SYSTEM_PROMPT = """
Kamu adalah seorang K-pop fan yang benar-benar mengikuti dunia K-pop,
terutama comeback, schedule, teaser, music video, stage, event, fandom,
rilisan musik, dan berita idol.

Kamu sedang ngobrol santai dengan sesama K-poper Indonesia.
Jangan terdengar seperti artikel berita, laporan, ensiklopedia, customer
service, atau chatbot formal.

Gunakan bahasa Indonesia yang natural, santai, hangat, dan conversational.
Boleh memakai istilah K-pop seperti comeback, schedule, teaser, stage,
era, promo, title track, b-side, fandom, fansign, selca, dan update kalau
memang cocok.

Jangan menggunakan markdown formatting dalam jawaban.
Jangan menggunakan **bold**, *italic*, heading, bullet point, numbered list,
atau format laporan.

Gunakan paragraf pendek seperti sedang chatting.

Jangan memulai dengan "Berikut adalah", "Berdasarkan informasi",
"Sebagai AI", "Saya tidak dapat", atau pembukaan formal sejenis.

Boleh bereaksi natural seperti "iyaa", "wah", "ohh", "gila", "jujur",
"ternyata", "gemes banget", atau "kayaknya", tetapi jangan dipaksakan.
Emoji boleh sesekali, jangan berlebihan.

Kalau membahas Hearts2Hearts, anggap lawan bicara sudah tahu dasar-dasar
grup tersebut. Tidak perlu menjelaskan apa itu idol/K-pop kecuali diminta.

Tanggal sekarang mengikuti tanggal aktual server aplikasi. Jangan mengklaim
tahun 2026 sebagai masa depan.

Untuk pertanyaan current/recent, prioritaskan WEB RESULTS.
Jangan mengarang. Kalau sumber tidak cukup, katakan secara natural bahwa
informasinya belum pasti.

Jangan menuliskan daftar sumber dalam bullet list. Sumber akan ditampilkan
oleh aplikasi di bawah jawaban.
"""

def compact_results(results):
    items = []
    for r in results.results:
        content = (getattr(r, "content", "") or "").strip()
        items.append(
            {
                "title": getattr(r, "title", "") or "",
                "url": getattr(r, "url", "") or "",
                "content": content[:2200],
            }
        )
    return items

@app.get("/")
def index():
    return render_template("index.html", model=MODEL)

@app.get("/api/health")
def health():
    try:
        tags = model_client.list()
        names = [m.model for m in tags.models]
        return jsonify({"ok": True, "model": MODEL, "remote_models": names})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502

@app.post("/api/chat")
def chat_api():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    history = data.get("history") or []
    use_web = bool(data.get("use_web", True))

    if not question:
        return jsonify({"error": "Pesan kosong."}), 400

    web_items = []
    web_context = ""

    if use_web:
        search_query = (
            f"{question} "
            f"latest recent news current {date.today().isoformat()} "
            "Hearts2Hearts H2H K-pop"
        )
        try:
            search = search_client.web_search(
                query=search_query,
                max_results=8,
            )
            web_items = compact_results(search)
            web_context = "\n\n".join(
                f"SOURCE {i}\n"
                f"TITLE: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"CONTENT:\n{item['content']}"
                for i, item in enumerate(web_items, 1)
            )
        except Exception as e:
            # Don't fail the whole chat when search is unavailable.
            web_context = (
                "WEB SEARCH FAILED. Do not pretend that a search succeeded. "
                f"Search error: {e}"
            )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep only a reasonable amount of previous chat history.
    for msg in history[-12:]:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    current_content = question
    if web_context:
        current_content += (
            "\n\nWEB RESULTS:\n"
            + web_context
            + "\n\nGunakan WEB RESULTS sebagai sumber utama untuk hal-hal terbaru."
        )

    messages.append({"role": "user", "content": current_content})

    try:
        response = model_client.chat(
            model=MODEL,
            messages=messages,
            think=True,
            stream=False,
        )
        answer = (response.message.content or "").strip()
    except Exception as e:
        return jsonify({"error": f"Gemma/Ollama gagal: {e}"}), 502

    return jsonify(
        {
            "answer": answer,
            "sources": [
                {"title": x["title"], "url": x["url"]}
                for x in web_items
                if x["title"] and x["url"]
            ],
        }
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
