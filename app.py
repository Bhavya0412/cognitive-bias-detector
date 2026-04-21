"""
Flask web application for the Cognitive Bias Detection & Debiasing
Assistant.

Routes:
    GET  /              - Single-page web UI
    POST /api/analyze   - JSON endpoint: {"text": "..."} -> analysis result
    GET  /api/metrics   - Training metrics (accuracy, per-class F1, etc.)
    GET  /api/health    - Simple health check
"""

import json
import os
import sys

from flask import Flask, jsonify, render_template, request

# Make `src` importable when running `python app.py` from project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from bias_detector import BiasDetector  # noqa: E402

app = Flask(__name__, template_folder="templates", static_folder="static")

# Instantiate once at startup - model stays warm in memory
print("Loading bias detection model...")
detector = BiasDetector()
print(f"Model ready. Classes: {detector.classes}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 10_000:
        return jsonify({"error": "Text too long (max 10,000 characters)"}), 400
    try:
        result = detector.analyze(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500


@app.route("/api/metrics")
def api_metrics():
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "Metrics not found. Train the model first."}), 404
    with open(metrics_path) as f:
        return jsonify(json.load(f))


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "classes": detector.classes})


@app.route("/metrics")
def metrics_page():
    return render_template("metrics.html")


if __name__ == "__main__":
    # host=0.0.0.0 so it is reachable from localhost on any interface;
    # debug=False keeps the app silent in the terminal during demo.
    app.run(host="0.0.0.0", port=5000, debug=False)
