"""
FastAPI Backend — HarmGuard NLP API v2.2
Fixed: Python 3.8 compatibility, proper JSON error responses
"""

import os, json, pickle, re, string, logging, subprocess
from pathlib import Path
from typing import Optional, List

import nltk
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Download NLTK data
for r in ["punkt", "stopwords", "punkt_tab"]:
    try: nltk.download(r, quiet=True)
    except: pass

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "models" / "baseline_pipeline.pkl"
BERT_DIR      = ROOT / "models" / "bert"
METRICS_BERT  = ROOT / "models" / "bert_metrics.json"
METRICS_BASE  = ROOT / "models" / "baseline_metrics.json"

app = FastAPI(title="HarmGuard NLP API", version="2.2.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

# ─── Global JSON error handler (no more HTML 500 pages) ──────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {str(exc)}"}
    )

# ─── Model Registry ──────────────────────────────────────────────────────────
class Reg:
    baseline   = None
    bert_model = None
    bert_tok   = None
    device     = "cpu"

reg = Reg()

def _load_baseline():
    try:
        if BASELINE_PATH.exists():
            with open(BASELINE_PATH, "rb") as f:
                reg.baseline = pickle.load(f)
            logger.info("Baseline model loaded.")
    except Exception as e:
        logger.warning(f"Baseline load failed: {e}")

def _load_bert():
    try:
        if not BERT_DIR.exists():
            return
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        reg.device    = "cuda" if torch.cuda.is_available() else "cpu"
        reg.bert_tok  = DistilBertTokenizerFast.from_pretrained(str(BERT_DIR))
        reg.bert_model = DistilBertForSequenceClassification.from_pretrained(str(BERT_DIR)).to(reg.device)
        reg.bert_model.eval()
        logger.info(f"BERT loaded ({reg.device}).")
    except Exception as e:
        logger.warning(f"BERT load failed: {e}")

@app.on_event("startup")
async def startup():
    _load_baseline()
    _load_bert()

# ─── Rule-Based Fallback ─────────────────────────────────────────────────────
_HATE_KEYWORDS = {
    "idiot","moron","stupid","loser","worthless","useless","disgusting",
    "hate","kill","die","retard","retarded","dumb","pathetic","scum",
    "trash","garbage","filth","filthy","ugly","fat","freak","creep",
    "pervert","racist","sexist","coward","imbecile","brainless",
}
_GROUPS = [
    "women","men","old people","young people","poor people","rich people",
    "black people","white people","immigrants","muslims","hindus","jews",
    "christians","gay","lesbian","transgender",
]
_STEREOTYPE_PATTERNS = [
    r"are\s+not\s+good\s+at", r"are\s+bad\s+at",
    r"are\s+better\s+.*\s+than", r"are\s+worse\s+.*\s+than",
    r"are\s+lazy", r"are\s+greedy", r"are\s+stupid",
    r"are\s+slow",  r"are\s+weak",  r"are\s+emotional",
    r"should\s+not\s+be\s+allowed", r"don'?t\s+belong",
    r"go\s+back\s+to",
]

def _rule_based_predict(text):
    t     = str(text).lower()
    words = set(re.sub(r"[^\w\s]", "", t).split())
    hits  = words & _HATE_KEYWORDS
    mdl   = "Rule-Based Fallback (train a model for higher accuracy)"
    if hits:
        score = min(0.55 + 0.08 * len(hits), 0.97)
        return 1, score, mdl
    for group in _GROUPS:
        if group in t:
            for pat in _STEREOTYPE_PATTERNS:
                if re.search(pat, t):
                    return 1, 0.89, mdl
    return 0, 0.82, mdl

# ─── Text Cleaning (for baseline) ────────────────────────────────────────────
try:
    _stop = set(stopwords.words("english"))
except Exception:
    _stop = set()
_stem = PorterStemmer()
_url  = re.compile(r"http\S+|www\.\S+")
_men  = re.compile(r"@\w+")
_ht   = re.compile(r"#(\w+)")

def clean(text):
    t = str(text).lower()
    t = _url.sub(" ", t)
    t = _men.sub(" ", t)
    t = _ht.sub(r" \1 ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    return " ".join(
        _stem.stem(w) for w in t.split()
        if w not in _stop and len(w) > 1
    )

# ─── Schemas ─────────────────────────────────────────────────────────────────
class PredReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model_choice: Optional[str] = "auto"

class PipelineReq(BaseModel):
    command: str

LABEL_MAP = {0: "non-harmful", 1: "harmful"}

# ─── Predict Helpers ─────────────────────────────────────────────────────────
def _pred_baseline(text):
    p   = reg.baseline.predict_proba([clean(text)])[0]
    lid = int(p.argmax())
    return lid, float(p[lid]), "TF-IDF + Logistic Regression"

def _pred_bert(text):
    import torch
    enc = reg.bert_tok(
        text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    enc = {k: v.to(reg.device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(
            reg.bert_model(**enc).logits, dim=-1
        ).cpu().numpy()[0]
    lid = int(probs.argmax())
    return lid, float(probs[lid]), "DistilBERT (fine-tuned)"

# ─── Predict Endpoint ────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(req: PredReq):
    try:
        if not req.text.strip():
            raise HTTPException(422, "Empty text")

        ch = (req.model_choice or "auto").lower()

        if ch == "bert" and reg.bert_model:
            lid, conf, mdl = _pred_bert(req.text)
        elif ch == "baseline" and reg.baseline:
            lid, conf, mdl = _pred_baseline(req.text)
        elif reg.bert_model:
            lid, conf, mdl = _pred_bert(req.text)
        elif reg.baseline:
            lid, conf, mdl = _pred_baseline(req.text)
        else:
            lid, conf, mdl = _rule_based_predict(req.text)

        return {
            "label":        LABEL_MAP[lid],
            "label_id":     lid,
            "confidence":   round(conf, 4),
            "model_used":   mdl,
            "text_preview": req.text[:80] + ("..." if len(req.text) > 80 else ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predict error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Prediction error: {str(e)}"}
        )

# ─── Pipeline Runner ─────────────────────────────────────────────────────────
CMD_MAP = {
    "validate":       ["python", "scripts/1_validate_data.py"],
    "train_baseline": ["python", "scripts/2_train_baseline.py"],
    "train_bert":     ["python", "scripts/3_train_bert.py"],
}
pipeline_log = []   # plain list — Python 3.8 compatible

def _run_pipeline(cmd_key):
    pipeline_log.clear()
    cmd = CMD_MAP.get(cmd_key)
    if not cmd:
        pipeline_log.append(f"Unknown command: {cmd_key}")
        return
    pipeline_log.append(f"Running: {' '.join(cmd)}\n")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in proc.stdout:
            pipeline_log.append(line)
        proc.wait()
        pipeline_log.append(f"\nFinished (exit {proc.returncode})")
        _load_baseline()
        _load_bert()
    except Exception as e:
        pipeline_log.append(f"\nError: {e}")

@app.post("/pipeline/run")
async def run_pipeline(req: PipelineReq, bg: BackgroundTasks):
    if req.command not in CMD_MAP:
        raise HTTPException(400, f"Unknown command. Choose: {list(CMD_MAP)}")
    pipeline_log.clear()
    pipeline_log.append("Starting...")
    bg.add_task(_run_pipeline, req.command)
    return {"status": "started", "command": req.command}

@app.get("/pipeline/log")
async def get_log():
    return {"log": "".join(pipeline_log)}

# ─── Health ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    bm, blm = {}, {}
    try:
        if METRICS_BERT.exists():
            with open(METRICS_BERT) as f:
                raw = json.load(f)
            bm = {k: raw[k] for k in ("accuracy","precision","recall","f1") if k in raw}
        if METRICS_BASE.exists():
            with open(METRICS_BASE) as f:
                raw = json.load(f)
            blm = {k: raw[k] for k in ("accuracy","precision","recall","f1") if k in raw}
    except Exception as e:
        logger.warning(f"Metrics read error: {e}")
    return {
        "baseline_loaded":  reg.baseline is not None,
        "bert_loaded":      reg.bert_model is not None,
        "fallback_active":  reg.baseline is None and reg.bert_model is None,
        "bert_metrics":     bm,
        "baseline_metrics": blm,
    }

# ─── Serve Web UI ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = ROOT / "api" / "ui.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>ui.html not found in api/</h2>", status_code=404)
