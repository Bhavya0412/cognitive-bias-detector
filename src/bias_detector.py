"""
Bias detector / debiaser core module.

- Loads the trained classifier and label encoder.
- Exposes analyze(text) -> structured result with:
    * detected bias type (+ probability distribution)
    * highlighted phrases that triggered the detection
    * rule-based debiased rewrite
    * plain-English explanation
- Works on multi-sentence input by analyzing each sentence independently
  and aggregating.
"""

import os
import pickle
import re
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "bias_classifier.pkl")
LE_PATH = os.path.join(ROOT, "models", "label_encoder.pkl")

# Confidence threshold: below this, we treat the sentence as neutral
# even if a bias label wins the argmax.
CONFIDENCE_THRESHOLD = 0.45


# ----------------------------------------------------------------------
# Lexicon of marker phrases — used for highlighting and as a rule-based
# fallback when the model is uncertain.
# ----------------------------------------------------------------------
BIAS_MARKERS = {
    "confirmation_bias": [
        r"\balways works\b", r"\bnever fails\b", r"\bobviously\b",
        r"\bclearly\b", r"\bundeniably\b", r"\bdefinitely\b",
        r"\bcertainly\b", r"\bwithout a doubt\b", r"\babsolutely\b",
        r"\bproves my point\b", r"\bI (?:knew|know) it\b",
        r"\bcritics are (?:wrong|mistaken)\b",
        r"\brefuse to consider\b", r"\bI don'?t need (?:more )?evidence\b",
        r"\bignore (?:the )?(?:counter|contrary|negative) (?:evidence|data|reports?)\b",
        r"\bcherry[- ]pick\b", r"\bno (?:need|point) (?:in|for) (?:more )?(?:analysis|research|review)\b",
        r"\banyone who disagrees\b", r"\bmy (?:intuition|gut|feeling) (?:about|is)\b",
        r"\bthat settles (?:it|the matter)\b",
        r"\b(?:every|all) (?:example|case|instance) (?:I know of |)supports\b",
        r"\bflawless\b", r"\bzero downsides\b", r"\bguaranteed to succeed\b",
        r"\brefuse to read\b", r"\bI only (?:trust|cite|read)\b",
    ],
    "anchoring_bias": [
        r"\bfirst (?:price|number|estimate|quote|offer|impression|bid|figure|forecast|report|article|version|conversation|meeting|slide|data point|rough guess|benchmark)\b",
        r"\binitial (?:estimate|price|quote|offer|projection|forecast|analysis|benchmark|read|guess|impression)\b",
        r"\bopening (?:bid|offer|number|quote)\b",
        r"\boriginal (?:quote|estimate|deadline|budget|price)\b",
        r"\bstarting (?:estimate|point|figure|price)\b",
        r"\banchored? (?:to|on)\b", r"\banchor(?:s|ing) (?:for|in)\b",
        r"\bthe very first\b", r"\bfrom day one\b",
        r"\bstuck (?:in my head|on (?:it|that|this|that number))\b",
        r"\bearliest (?:forecast|guess|estimate|analysis)\b",
        r"\bkeep(?:s)? (?:comparing|framing)\b",
        r"\bI (?:still )?(?:use|treat) (?:that|it) as (?:the )?(?:baseline|reference)\b",
        r"\bcannot (?:let go of|shake) (?:that|the) (?:number|figure|estimate)\b",
        r"\bset(?:s)? (?:my|the) expectation\b",
    ],
    "availability_heuristic": [
        r"\bjust (?:saw|heard|read)\b",
        r"\b(?:saw|read|heard) (?:it |about it |)on (?:the )?(?:news|TV|radio|social media)\b",
        r"\btrending (?:on social media|on [a-z]+|today|this week)\b",
        r"\bmy (?:friend|neighbor|cousin|coworker|colleague|manager|family) (?:told|mentioned|had|experienced)\b",
        r"\bviral (?:video|post|story|article)\b",
        r"\b(?:I |)can (?:easily |)recall\b",
        r"\b(?:I )?remember (?:a |)(?:dramatic |vivid |striking |)(?:case|story|example|anecdote) (?:about|of)\b",
        r"\bthree (?:people|examples|articles|stories|cases)\b",
        r"\bkeeps? (?:coming up|showing up|mentioning)\b",
        r"\bin (?:the|my) (?:news|feed|headlines)\b",
        r"\b(?:front|head)line (?:said|discussed|mentioned)\b",
        r"\bpodcast (?:just |recently |)covered\b",
        r"\bmust be (?:very )?(?:common|widespread|happening everywhere)\b",
        r"\bmust be (?:a |an )?(?:epidemic|trend|widespread)\b",
        r"\bdominat(?:es|ing) (?:the )?news\b",
        r"\bfamous person (?:discussed|mentioned|talked about)\b",
    ],
    "framing_effect": [
        r"\bsuccess rate\b.{0,40}\bfailure rate\b",
        r"\bfat[- ]free\b", r"\bfat\b.{0,20}\b\d+ ?%",
        r"\bsaves? \d+ lives\b", r"\b\d+ people (?:still )?die\b",
        r"\bsounds (?:better|worse|healthier|stronger|more (?:positive|appealing|generous|compelling|attractive|reassuring|persuasive|desirable|prestigious|safer))\b",
        r"\b(?:more |)(?:reassuring|persuasive|positive|compelling|attractive|appealing) than\b",
        r"\bfeels (?:better|worse|cheaper|safer|healthier|more (?:attractive|generous))\b",
        r"\bfeels? like (?:a |)(?:bargain|steal|deal|gift)\b",
        r"\b(?:pre[- ]owned|used car)\b",
        r"\binvestment (?:opportunity|in infrastructure)\b",
        r"\bcomplimentary (?:trial|offer)\b",
        r"\bexclusive (?:access|membership|savings|offer|deal)\b",
        r"\b(?:workforce optimization|service fee|schedule adjustment|revenue[- ]neutral|cost[- ]saving|streamlined|natural flavors)\b",
        r"\b(?:only|just) \$?\d+ (?:per day|per week|a day|a week)\b",
        r"\blimited[- ]time\b",
        r"\bguaranteed return\b",
        r"\bonly \d+ (?:seats|spots|left)\b",
        r"\bup to \d+% off\b",
        r"\bpremium (?:member|quality)\b",
    ],
}

# Compile once
_COMPILED_MARKERS = {
    cls: [re.compile(p, re.IGNORECASE) for p in pats]
    for cls, pats in BIAS_MARKERS.items()
}


# ----------------------------------------------------------------------
# Rule-based debiasing: substitute absolutist / loaded phrasing with
# neutral, qualified alternatives.
# ----------------------------------------------------------------------
DEBIAS_RULES = [
    # Confirmation bias - absolutes -> qualifiers
    (re.compile(r"\balways works\b", re.I), "has worked in some cases"),
    (re.compile(r"\bnever fails\b", re.I), "has not failed in the cases observed so far"),
    (re.compile(r"\bobviously\b", re.I), "it appears that"),
    (re.compile(r"\bclearly\b", re.I), "apparently"),
    (re.compile(r"\bundeniably\b", re.I), "evidence suggests"),
    (re.compile(r"\bdefinitely\b", re.I), "likely"),
    (re.compile(r"\bwithout a doubt\b", re.I), "based on current evidence"),
    (re.compile(r"\babsolutely\b", re.I), "apparently"),
    (re.compile(r"\bcertainly\b", re.I), "likely"),
    (re.compile(r"\bunquestionably\b", re.I), "based on available data"),
    (re.compile(r"\bproves (?:my point|it)\b", re.I), "supports this view, though further evidence would help"),
    (re.compile(r"\bzero downsides\b", re.I), "few reported downsides so far"),
    (re.compile(r"\bflawless\b", re.I), "effective in the cases reviewed"),
    (re.compile(r"\bguaranteed to succeed\b", re.I), "likely to succeed based on current evidence"),
    (re.compile(r"\beveryone\b", re.I), "many people"),
    (re.compile(r"\bnobody\b", re.I), "few people"),
    (re.compile(r"\bno one\b", re.I), "few people"),
    (re.compile(r"\bmust\b", re.I), "may"),
    (re.compile(r"\bproves\b", re.I), "suggests"),
    (re.compile(r"\bthat settles (?:it|the matter)\b", re.I),
     "but additional evidence should still be considered"),

    # Anchoring - explicit anchor talk -> acknowledge and broaden
    (re.compile(r"\bthe first (?:price|number|estimate|quote|offer)\b", re.I),
     "one early reference point"),
    (re.compile(r"\binitial estimate\b", re.I), "one early estimate"),
    (re.compile(r"\bopening (?:bid|offer|quote)\b", re.I), "one early offer"),
    (re.compile(r"\banchored? (?:to|on)\b", re.I), "influenced by"),
    (re.compile(r"\bstuck (?:in my head|on (?:that|the) number)\b", re.I),
     "worth reconsidering in light of newer data"),

    # Availability - anecdote -> caveat
    (re.compile(r"\bmust be (?:very )?common\b", re.I), "may be more common, though frequency is not clear"),
    (re.compile(r"\bmust be (?:a |an )?epidemic\b", re.I), "may be an emerging trend"),
    (re.compile(r"\bmust be happening everywhere\b", re.I), "may be occurring more widely, though data is limited"),
    (re.compile(r"\bmust be widespread\b", re.I), "may be more widespread, though this is anecdotal"),
    (re.compile(r"\bmust be a (?:major )?trend\b", re.I), "could be a trend, though anecdotal"),
    (re.compile(r"\bis clearly the norm\b", re.I), "may be more frequent than usual, though this is anecdotal"),
    (re.compile(r"\bjust saw\b", re.I), "recently encountered"),
    (re.compile(r"\bjust heard\b", re.I), "recently heard"),

    # Framing - loaded positive labels toggled to neutral
    (re.compile(r"\bsounds (?:much |)better than\b", re.I), "is numerically equivalent to"),
    (re.compile(r"\bfeels better than\b", re.I), "is arithmetically equivalent to"),
    (re.compile(r"\bmore (?:reassuring|persuasive|positive|compelling) than\b", re.I),
     "is mathematically equivalent to"),

    # General hedging
    (re.compile(r"\bI'?m confident\b", re.I), "current evidence suggests"),
    (re.compile(r"\bI refuse to\b", re.I), "I have been reluctant to"),
    (re.compile(r"\bcritics are wrong\b", re.I), "critics raise valid concerns worth examining"),
]


EXPLANATIONS = {
    "confirmation_bias": (
        "Confirmation bias — the statement favors information that supports a "
        "prior belief while dismissing or ignoring contradictory evidence. "
        "Absolute language ('always', 'never', 'clearly') and phrases that "
        "refuse to engage with opposing views are common markers."
    ),
    "anchoring_bias": (
        "Anchoring bias — the reasoning relies too heavily on the first piece "
        "of information encountered (a price, estimate, or number). The "
        "initial value becomes a reference point that distorts later judgments."
    ),
    "availability_heuristic": (
        "Availability heuristic — the statement estimates the likelihood of "
        "something based on how easily examples come to mind (recent news, a "
        "friend's story, a vivid anecdote), rather than on base-rate data."
    ),
    "framing_effect": (
        "Framing effect — the same fact is presented in a way that steers the "
        "reader's feelings (e.g., '90% success' vs '10% failure'). The "
        "underlying numbers are identical, but the wording changes the "
        "emotional reaction."
    ),
    "neutral": (
        "No strong bias detected. The statement qualifies its claims, cites "
        "uncertainty, or acknowledges alternatives — the hallmarks of neutral, "
        "evidence-based writing."
    ),
}


class BiasDetector:
    def __init__(self, model_path=MODEL_PATH, le_path=LE_PATH):
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        with open(le_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        self.classes: List[str] = list(self.label_encoder.classes_)

    # ----- sentence splitting -----
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        # Simple regex-based splitter; avoids nltk punkt_tab download issues
        # in offline environments.
        text = text.strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
        return [p.strip() for p in parts if p.strip()]

    # ----- classification -----
    def classify_sentence(self, sentence: str) -> Tuple[str, Dict[str, float]]:
        probs = self.pipeline.predict_proba([sentence])[0]
        prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
        pred = max(prob_dict, key=prob_dict.get)
        # Low-confidence fallback -> neutral
        if prob_dict[pred] < CONFIDENCE_THRESHOLD and pred != "neutral":
            pred = "neutral"
        return pred, prob_dict

    # ----- marker highlighting -----
    @staticmethod
    def find_markers(sentence: str, bias_type: str) -> List[str]:
        if bias_type == "neutral":
            return []
        matches = []
        for pattern in _COMPILED_MARKERS.get(bias_type, []):
            for m in pattern.finditer(sentence):
                matches.append(m.group(0))
        # Deduplicate, preserve order
        seen = set()
        out = []
        for m in matches:
            key = m.lower()
            if key not in seen:
                seen.add(key)
                out.append(m)
        return out

    # ----- debiasing -----
    @staticmethod
    def debias_sentence(sentence: str, bias_type: str) -> str:
        if bias_type == "neutral":
            return sentence
        rewritten = sentence
        for pattern, replacement in DEBIAS_RULES:
            rewritten = pattern.sub(replacement, rewritten)

        # Append qualifier if none of the rules fired and the text still
        # looks absolute.
        if rewritten == sentence:
            suffix = {
                "confirmation_bias":
                    " However, counter-evidence should also be considered before drawing a firm conclusion.",
                "anchoring_bias":
                    " That early reference point may not reflect the full range of current evidence.",
                "availability_heuristic":
                    " Frequency statistics, not anecdotes, would give a more accurate picture.",
                "framing_effect":
                    " The same fact could be stated with the complementary figure for a balanced view.",
            }.get(bias_type, "")
            if suffix and not rewritten.rstrip().endswith(("."  , "!", "?")):
                rewritten = rewritten.rstrip() + "."
            rewritten = rewritten.rstrip() + suffix

        # Clean up double spaces
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        # Capitalize first letter
        if rewritten:
            rewritten = rewritten[0].upper() + rewritten[1:]
        return rewritten

    # ----- single public entry point -----
    def analyze(self, text: str) -> Dict:
        sentences = self.split_sentences(text)
        if not sentences:
            return {
                "overall_bias": "neutral",
                "overall_confidence": 1.0,
                "sentences": [],
                "debiased_text": "",
                "explanation": EXPLANATIONS["neutral"],
                "bias_summary": {cls: 0 for cls in self.classes},
            }

        sentence_results = []
        bias_counter = {cls: 0 for cls in self.classes}
        debiased_parts = []

        for s in sentences:
            label, probs = self.classify_sentence(s)
            markers = self.find_markers(s, label)
            debiased = self.debias_sentence(s, label)
            sentence_results.append({
                "text": s,
                "bias": label,
                "confidence": round(probs[label], 4),
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
                "markers": markers,
                "debiased": debiased,
                "explanation": EXPLANATIONS[label],
            })
            bias_counter[label] += 1
            debiased_parts.append(debiased)

        # Determine overall bias = most frequent non-neutral, else neutral
        non_neutral = {k: v for k, v in bias_counter.items() if k != "neutral"}
        if any(v > 0 for v in non_neutral.values()):
            overall = max(non_neutral, key=non_neutral.get)
        else:
            overall = "neutral"

        # Average confidence across sentences labeled with the overall bias
        matching = [r["confidence"] for r in sentence_results if r["bias"] == overall]
        overall_conf = round(sum(matching) / len(matching), 4) if matching else 1.0

        return {
            "overall_bias": overall,
            "overall_confidence": overall_conf,
            "sentences": sentence_results,
            "debiased_text": " ".join(debiased_parts),
            "explanation": EXPLANATIONS[overall],
            "bias_summary": bias_counter,
        }


# ----------------------------------------------------------------------
# Quick manual test when run directly.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    bd = BiasDetector()
    samples = [
        "This method always works because it worked once before.",
        "The first price I saw for the laptop was $1000, so anything above $800 feels expensive.",
        "I just saw a news story about it, so this disease must be very common.",
        "The surgery has a 90% success rate, which sounds much better than a 10% failure rate.",
        "Preliminary evidence suggests the method may help, though more research is needed.",
        "Our plan saves 200 lives. Critics are wrong about this strategy — it clearly works.",
    ]
    for s in samples:
        print("=" * 70)
        print(f"INPUT : {s}")
        result = bd.analyze(s)
        print(f"BIAS  : {result['overall_bias']}  (conf={result['overall_confidence']})")
        print(f"REWRITE: {result['debiased_text']}")
        if result["sentences"]:
            markers = result["sentences"][0]["markers"]
            if markers:
                print(f"MARKERS: {markers}")
        print(f"EXPL  : {result['explanation']}")
