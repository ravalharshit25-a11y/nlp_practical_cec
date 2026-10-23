import io, os
from flask import Flask, render_template, request, jsonify
import nlp_engine as engine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

def read_upload(f):
    name = f.filename.lower(); data = f.read()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/sample")
def sample():
    return jsonify(text=open(os.path.join(app.root_path, "samples", "sample_contract.txt"), encoding="utf-8").read())

@app.route("/api/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "")
    if "file" in request.files and request.files["file"].filename:
        try: text = read_upload(request.files["file"])
        except Exception as e: return jsonify(error=f"Could not read file: {e}"), 400
    if len(text.strip()) < 60: return jsonify(error="Please provide at least a few sentences of contract text."), 400
    res = engine.analyze(text, request.form.get("query") or None)
    return (jsonify(res), 400) if "error" in res else jsonify(res)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
