import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from rag.document_store import DocumentStore
from rag.pdf_reader import extract_pdf_text
from rag.rag_engine import RagEngine


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
INDEX_DIR = BASE_DIR / "storage" / "indexes"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    store = DocumentStore(INDEX_DIR)
    rag = RagEngine(store)

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "ollama_model": rag.ollama_model})

    @app.post("/api/upload")
    def upload_pdf():
        uploaded = request.files.get("pdf")
        if not uploaded:
            return jsonify({"error": "Upload a PDF file first."}), 400

        if not uploaded.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported."}), 400

        doc_id = uuid.uuid4().hex
        safe_name = Path(uploaded.filename).name
        pdf_path = UPLOAD_DIR / f"{doc_id}_{safe_name}"
        uploaded.save(pdf_path)

        try:
            text, page_count = extract_pdf_text(pdf_path)
            indexed = rag.index_document(doc_id=doc_id, filename=safe_name, text=text, page_count=page_count)
        except ValueError as exc:
            pdf_path.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            pdf_path.unlink(missing_ok=True)
            return jsonify({"error": f"Could not process this PDF: {exc}"}), 500

        return jsonify(
            {
                "doc_id": doc_id,
                "filename": safe_name,
                "page_count": page_count,
                "chunk_count": indexed["chunk_count"],
                "preview": indexed["preview"],
            }
        )

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        doc_id = payload.get("doc_id")
        question = (payload.get("question") or "").strip()

        if not doc_id:
            return jsonify({"error": "Upload and analyze a PDF before asking questions."}), 400
        if not question:
            return jsonify({"error": "Type a question about the PDF."}), 400

        try:
            answer = rag.answer_question(doc_id=doc_id, question=question)
        except FileNotFoundError:
            return jsonify({"error": "That document index was not found. Please upload the PDF again."}), 404
        except Exception as exc:
            return jsonify({"error": f"Question answering failed: {exc}"}), 500

        return jsonify(answer)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
