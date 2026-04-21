"""
Updated api/app.py — FastAPI backend that ALSO serves the standalone web UI.
Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
Then open:  http://localhost:8000
"""

import os, json, pickle, re, string, logging, subprocess, sys
from pathlib import Path
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

for r in ["punkt", "stopwords", "punkt_tab"]:
    try: nltk.download(r, quiet=True)
    except: pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "models" / "baseline_pipeline.pkl"
BERT_DIR      = ROOT / "models" / "bert"
METRICS_BERT  = ROOT / "models" / "bert_metrics.json"
METRICS_BASE  = ROOT / "models" / "baseline_metrics.json"

app = FastAPI(title="HarmGuard NLP API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ─── Model Registry ──────────────────────────────────────────────────────────
class Reg:
    baseline=None; bert_model=None; bert_tok=None; device="cpu"

reg = Reg()

def _load_baseline():
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH,"rb") as f: reg.baseline=pickle.load(f)
        logger.info("Baseline loaded.")

def _load_bert():
    if not BERT_DIR.exists(): return
    try:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        reg.device = "cuda" if torch.cuda.is_available() else "cpu"
        reg.bert_tok = DistilBertTokenizerFast.from_pretrained(str(BERT_DIR))
        reg.bert_model = DistilBertForSequenceClassification.from_pretrained(str(BERT_DIR)).to(reg.device)
        reg.bert_model.eval()
        logger.info(f"BERT loaded ({reg.device}).")
    except Exception as e:
        logger.warning(f"BERT load failed: {e}")

@app.on_event("startup")
async def startup(): _load_baseline(); _load_bert()

# ─── Text Cleaning ───────────────────────────────────────────────────────────
_stop=set(stopwords.words("english")); _stem=PorterStemmer()
_url=re.compile(r"http\S+|www\.\S+"); _men=re.compile(r"@\w+"); _ht=re.compile(r"#(\w+)")

def clean(text):
    t=str(text).lower()
    t=_url.sub(" ",t); t=_men.sub(" ",t); t=_ht.sub(r" \1 ",t)
    t=t.translate(str.maketrans("","",string.punctuation))
    return " ".join(_stem.stem(w) for w in t.split() if w not in _stop and len(w)>1)

# ─── Schemas ─────────────────────────────────────────────────────────────────
class PredReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model_choice: Optional[str] = "auto"

class PipelineReq(BaseModel):
    command: str  # "validate" | "train_baseline" | "train_bert"

LABEL_MAP = {0:"non-harmful", 1:"harmful"}

# ─── Predict ─────────────────────────────────────────────────────────────────
def _pred_baseline(text):
    p=reg.baseline.predict_proba([clean(text)])[0]
    lid=int(p.argmax())
    return lid,float(p[lid]),"TF-IDF + Logistic Regression"

def _pred_bert(text):
    import torch
    enc=reg.bert_tok(text,max_length=128,padding="max_length",
                     truncation=True,return_tensors="pt")
    enc={k:v.to(reg.device) for k,v in enc.items()}
    with torch.no_grad():
        probs=torch.softmax(reg.bert_model(**enc).logits,dim=-1).cpu().numpy()[0]
    lid=int(probs.argmax())
    return lid,float(probs[lid]),"DistilBERT (fine-tuned)"

@app.post("/predict")
async def predict(req: PredReq):
    if not req.text.strip(): raise HTTPException(422,"Empty text")
    ch=(req.model_choice or "auto").lower()
    if ch=="bert" and reg.bert_model:        lid,conf,mdl=_pred_bert(req.text)
    elif ch=="baseline" and reg.baseline:    lid,conf,mdl=_pred_baseline(req.text)
    elif reg.bert_model:                     lid,conf,mdl=_pred_bert(req.text)
    elif reg.baseline:                       lid,conf,mdl=_pred_baseline(req.text)
    else: raise HTTPException(503,"No model loaded. Run training first.")
    return {"label":LABEL_MAP[lid],"label_id":lid,"confidence":round(conf,4),
            "model_used":mdl,"text_preview":req.text[:80]+("…"if len(req.text)>80 else "")}

# ─── Pipeline Runner ─────────────────────────────────────────────────────────
CMD_MAP = {
    "validate":      ["python","scripts/1_validate_data.py"],
    "train_baseline":["python","scripts/2_train_baseline.py"],
    "train_bert":    ["python","scripts/3_train_bert.py"],
}
pipeline_log: list[str] = []

def _run_pipeline(cmd_key: str):
    pipeline_log.clear()
    cmd = CMD_MAP.get(cmd_key)
    if not cmd:
        pipeline_log.append(f"Unknown command: {cmd_key}"); return
    pipeline_log.append(f"▶ Running: {' '.join(cmd)}\n")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            pipeline_log.append(line)
        proc.wait()
        pipeline_log.append(f"\n✅ Finished (exit {proc.returncode})")
        if cmd_key in ("train_baseline","train_bert"):
            _load_baseline(); _load_bert()  # hot-reload
    except Exception as e:
        pipeline_log.append(f"\n❌ Error: {e}")

@app.post("/pipeline/run")
async def run_pipeline(req: PipelineReq, bg: BackgroundTasks):
    if req.command not in CMD_MAP:
        raise HTTPException(400, f"Unknown command. Choose: {list(CMD_MAP)}")
    pipeline_log.clear()
    pipeline_log.append("⏳ Starting…")
    bg.add_task(_run_pipeline, req.command)
    return {"status":"started","command":req.command}

@app.get("/pipeline/log")
async def get_log():
    return {"log":"".join(pipeline_log)}

@app.get("/health")
async def health():
    bm,blm={},{}
    if METRICS_BERT.exists():
        with open(METRICS_BERT) as f: raw=json.load(f)
        bm={k:raw[k] for k in("accuracy","precision","recall","f1") if k in raw}
    if METRICS_BASE.exists():
        with open(METRICS_BASE) as f: raw=json.load(f)
        blm={k:raw[k] for k in("accuracy","precision","recall","f1") if k in raw}
    return {"baseline_loaded":reg.baseline is not None,
            "bert_loaded":reg.bert_model is not None,
            "bert_metrics":bm,"baseline_metrics":blm}

# ─── Serve Web UI ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = ROOT / "api" / "ui.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>UI not found — place ui.html in api/</h2>", status_code=404)
