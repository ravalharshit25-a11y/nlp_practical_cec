"""Legal NLP engine: segmentation, NER, classification, semantic risk, summary, simplification."""
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ---------- optional heavy models (auto-detected) ----------
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None
try:
    from sentence_transformers import SentenceTransformer
    _st = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _st = None

# ---------- clause categories: seed data for the classifier ----------
SEEDS = {
 "Termination": ["Either party may terminate this Agreement upon thirty days written notice.",
   "This Agreement shall terminate immediately upon material breach by the other party.",
   "Company may terminate for convenience at any time without cause."],
 "Payment": ["Client shall pay all invoices within thirty days of receipt.",
   "A late fee of 2% per month shall accrue on overdue amounts.",
   "Fees are non-refundable and payable in advance."],
 "Confidentiality": ["Each party shall keep Confidential Information strictly confidential and not disclose it to third parties.",
   "The receiving party shall use proprietary information solely for the purposes of this Agreement.",
   "Confidentiality obligations survive termination of this Agreement for five years."],
 "Liability": ["In no event shall either party be liable for indirect, incidental or consequential damages.",
   "Total aggregate liability shall not exceed the fees paid in the preceding twelve months.",
   "Provider's liability is limited and excludes loss of profits."],
 "Indemnification": ["Customer shall indemnify, defend and hold harmless Provider from any claims, losses and expenses.",
   "The indemnifying party shall bear all legal costs arising from third party claims."],
 "Intellectual Property": ["All intellectual property rights in the work product shall vest exclusively in the Company.",
   "Contractor hereby assigns all copyrights, patents and inventions created under this Agreement.",
   "Licensor grants a non-exclusive, non-transferable license to use the software."],
 "Non-Compete / Exclusivity": ["Employee shall not engage in any competing business for two years after termination.",
   "Vendor agrees to work exclusively with the Company and shall not solicit its customers or employees."],
 "Renewal / Term": ["This Agreement shall automatically renew for successive one year terms unless notice is given.",
   "The initial term of this Agreement shall be three years commencing on the Effective Date."],
 "Governing Law": ["This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
   "The courts of Mumbai shall have exclusive jurisdiction over any matter arising hereunder."],
 "Dispute Resolution": ["Any dispute shall be finally resolved by binding arbitration under the rules of the arbitration association.",
   "The parties waive the right to a jury trial and to participate in any class action."],
 "Warranty": ["Provider warrants that the services will be performed in a professional and workmanlike manner.",
   "The software is provided as is without warranty of any kind, express or implied."],
 "Force Majeure": ["Neither party shall be liable for delay caused by acts of God, war, pandemic, or events beyond reasonable control."],
 "Data Privacy": ["Provider shall process personal data in compliance with applicable data protection laws.",
   "Customer data may be shared with affiliates and third party partners for analytics."],
 "Assignment": ["Neither party may assign this Agreement without prior written consent.",
   "Company may assign or transfer its rights to any affiliate or successor without notice."],
}
KEYWORDS = {
 "Termination": ["terminate", "termination", "cancel"], "Payment": ["pay", "fee", "invoice", "payment", "refund", "interest"],
 "Confidentiality": ["confidential", "non-disclosure", "proprietary"], "Liability": ["liable", "liability", "damages"],
 "Indemnification": ["indemnif", "hold harmless"], "Intellectual Property": ["intellectual property", "copyright", "patent", "license", "trademark"],
 "Non-Compete / Exclusivity": ["non-compete", "compete", "exclusive", "solicit"], "Renewal / Term": ["renew", "initial term", "extension"],
 "Governing Law": ["governed by", "jurisdiction", "laws of"], "Dispute Resolution": ["arbitration", "dispute", "jury", "mediation"],
 "Warranty": ["warrant", "as is", "guarantee"], "Force Majeure": ["force majeure", "act of god", "beyond reasonable control"],
 "Data Privacy": ["personal data", "privacy", "data protection", "gdpr"], "Assignment": ["assign", "transfer", "novation"],
}

# ---------- risk library (semantic + regex) ----------
RISKS = [
 dict(id="unlimited_liability", title="Unlimited / uncapped liability", sev=9, rx=r"unlimited liability|without limit(ation)?|no (cap|limit) on liability|liable for all",
      ex="Party shall be fully liable for all damages without any limitation or cap.",
      why="You could owe an unlimited amount if something goes wrong.", fix="Ask for a liability cap (e.g. fees paid in the last 12 months)."),
 dict(id="one_sided_indemnity", title="One-sided indemnification", sev=8, rx=r"indemnif\w+.{0,80}(sole|any and all|all claims)|hold harmless",
      ex="Customer shall indemnify and hold harmless Provider from any and all claims, losses and expenses.",
      why="You cover the other side's legal costs and losses, even in cases that may not be your fault.", fix="Make indemnity mutual and limited to your own negligence or breach."),
 dict(id="unilateral_termination", title="Unilateral termination without cause", sev=7, rx=r"terminate.{0,60}(at any time|sole discretion|without (cause|notice|reason))|immediate(ly)? terminat",
      ex="Company may terminate this Agreement at any time, for any reason, without notice.",
      why="The other side can end the deal suddenly while you may be locked in.", fix="Require written notice (30+ days) and give both sides the same right."),
 dict(id="auto_renewal", title="Automatic renewal / lock-in", sev=5, rx=r"auto(matic(ally)?)?[- ]renew|renew(s|ed)? automatically|successive .{0,20}terms?",
      ex="This Agreement automatically renews for successive terms unless cancelled 90 days in advance.",
      why="You may be bound for another full term if you miss a notice deadline.", fix="Calendar the cancellation date or negotiate a shorter notice window."),
 dict(id="harsh_penalty", title="Heavy penalties / late fees", sev=6, rx=r"penalt(y|ies)|liquidated damages|(\d{1,2}(\.\d+)?)\s?%\s*(per|a)\s*(month|week|day)",
      ex="A penalty of 5% per month shall be charged on all overdue amounts.",
      why="Small delays can snowball into large charges.", fix="Cap late fees and add a grace period."),
 dict(id="broad_noncompete", title="Broad non-compete / exclusivity", sev=7, rx=r"shall not (engage|work|compete)|non-?compet|exclusively (with|for)|exclusive (supplier|dealing)|shall not solicit",
      ex="Contractor shall not work for any competitor anywhere in the world for five years.",
      why="Could restrict your ability to earn a living or take other clients.", fix="Limit by duration, geography and scope."),
 dict(id="ip_assignment", title="Broad IP assignment", sev=7, rx=r"hereby assigns?|all (right, title|intellectual property).{0,60}(vest|assign)|work made for hire",
      ex="All inventions and work product, whether or not related to this project, belong to the Company.",
      why="You may lose ownership of work, including work done outside the project.", fix="Restrict assignment to deliverables created for this project."),
 dict(id="as_is", title="No warranty / 'as is'", sev=4, rx=r"\bas[- ]is\b|without warrant(y|ies)|disclaims? all warrant",
      ex="The services are provided as is without warranty of any kind.",
      why="You have little recourse if the product or service fails.", fix="Negotiate minimum performance warranties."),
 dict(id="unilateral_change", title="Unilateral amendment rights", sev=7, rx=r"(may|reserves the right to) (modify|amend|change|revise).{0,60}(at any time|sole discretion|without notice)",
      ex="Provider may modify these terms at any time in its sole discretion without notice.",
      why="The other side can change the rules after you've signed.", fix="Require written, mutually-signed amendments."),
 dict(id="arbitration_waiver", title="Arbitration / waiver of court rights", sev=6, rx=r"binding arbitration|waive.{0,40}(jury|class action|right to)",
      ex="You waive your right to a jury trial and agree to binding arbitration in a venue chosen by us.",
      why="You may give up access to courts and group claims.", fix="Check the venue, costs and whether arbitration is mutual."),
 dict(id="data_sharing", title="Broad data sharing", sev=6, rx=r"share.{0,60}(third part|affiliate|partners)|sell.{0,30}data",
      ex="Customer data may be shared with affiliates and third parties for any purpose.",
      why="Your or your customers' data may be used or sold without control.", fix="Limit use to service delivery and require consent for sharing."),
 dict(id="free_assignment", title="One-sided assignment", sev=4, rx=r"may assign.{0,60}without (notice|consent)",
      ex="Company may assign this Agreement to any third party without notice or consent.",
      why="Your counterparty could change to someone you did not choose.", fix="Require consent for assignment by either side."),
 dict(id="vague_terms", title="Vague / subjective wording", sev=3, rx=r"sole discretion|reasonable efforts|as (it )?deems? (necessary|appropriate)|from time to time",
      ex="Provider shall perform the services as it deems appropriate in its sole discretion.",
      why="Open-ended wording is easy to dispute later.", fix="Replace with measurable obligations and deadlines."),
]

GLOSSARY = {
 r"\bhereinafter\b": "from now on", r"\bhereby\b": "by this document", r"\bherein\b": "in this document",
 r"\bthereof\b": "of it", r"\bpursuant to\b": "under", r"\bnotwithstanding\b": "despite", r"\bindemnif(y|ies)\b": "compensate for losses",
 r"\bhold harmless\b": "not blame or sue", r"\bterminate\b": "end", r"\bcommence(s|d)?\b": "start", r"\bprior to\b": "before",
 r"\bin the event that\b": "if", r"\bshall\b": "must", r"\bsubsequent to\b": "after", r"\bforthwith\b": "immediately",
 r"\bwarrant(s)?\b": "promise(s)", r"\bremuneration\b": "payment", r"\bbreach\b": "violation", r"\bsurvive(s)?\b": "stay(s) in effect after the end",
 r"\bconsequential damages\b": "indirect losses", r"\bcovenant(s)?\b": "promise(s)", r"\bLessee\b": "tenant", r"\bLessor\b": "landlord",
}
CAT_EXPLAIN = {
 "Termination": "Explains how and when the agreement can be ended.", "Payment": "Describes what must be paid, when, and what happens if late.",
 "Confidentiality": "Requires keeping shared information secret.", "Liability": "Sets limits on who pays for damages.",
 "Indemnification": "One side agrees to cover the other's losses or legal costs.", "Intellectual Property": "Decides who owns work, ideas and software.",
 "Non-Compete / Exclusivity": "Restricts working with others or competing.", "Renewal / Term": "Sets how long the deal lasts and if it renews.",
 "Governing Law": "Decides which country's/state's laws and courts apply.", "Dispute Resolution": "Sets how disagreements are settled.",
 "Warranty": "States what quality or performance is promised.", "Force Majeure": "Excuses delays caused by events beyond anyone's control.",
 "Data Privacy": "Covers how personal data is used and protected.", "Assignment": "Controls whether rights can be transferred to someone else.",
 "General": "General or administrative provision.",
}

# ---------- model setup ----------
_CATS = list(SEEDS)
_X = [s for c in _CATS for s in SEEDS[c]]
_y = [c for c in _CATS for s in SEEDS[c]]
_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit(_X + [r["ex"] for r in RISKS])
_clf = LogisticRegression(max_iter=2000, C=8).fit(_vec.transform(_X), _y)
_risk_mat = _vec.transform([r["ex"] for r in RISKS])
_risk_emb = _st.encode([r["ex"] for r in RISKS], normalize_embeddings=True) if _st else None

def engine_info():
    return {"ner": "spaCy + regex" if _nlp else "regex rules", "similarity": "SBERT embeddings" if _st else "TF-IDF cosine",
            "classifier": "TF-IDF + Logistic Regression + keyword boost", "summarizer": "extractive TF-IDF ranking"}

# ---------- segmentation ----------
HEAD = re.compile(r"^\s*((article|section|clause)\s+[\dIVXivx]+[.:)]?|\d+(\.\d+)*[.)]\s|\([a-z0-9]+\)\s|[A-Z][A-Z \-&/]{4,}:?\s*$)", re.I | re.M)

def segment(text):
    text = re.sub(r"\r", "", text).strip()
    parts, cur = [], []
    for line in text.split("\n"):
        if HEAD.match(line) and cur and len(" ".join(cur)) > 40:
            parts.append(" ".join(cur)); cur = []
        if line.strip():
            cur.append(line.strip())
    if cur: parts.append(" ".join(cur))
    if len(parts) < 3:  # fallback: sentence grouping
        sents = re.split(r"(?<=[.;])\s+(?=[A-Z0-9(])", re.sub(r"\s+", " ", text))
        parts = [s for s in sents]
    return [re.sub(r"\s+", " ", p).strip() for p in parts if len(p.strip()) > 25]

# ---------- NER ----------
NER_RX = {
 "MONEY": r"(?:(?:USD|INR|EUR|GBP|Rs\.?|₹|\$|€|£)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|lakh|crore|k))?)",
 "DATE": r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
 "DURATION": r"\b(?:\(?\d+\)?|one|two|three|four|five|six|seven|eight|nine|ten|thirty|sixty|ninety)[\s-]+(?:calendar\s+)?(?:day|week|month|year)s?\b",
 "PERCENT": r"\b\d+(?:\.\d+)?\s?%",
 "ORG": r"\b(?:[A-Z][\w&.]+\s){1,4}(?:Inc\.?|LLC|Ltd\.?|Limited|Pvt\.?|Corp(?:oration)?\.?|LLP|GmbH|Company)\b",
 "LAW/JURISDICTION": r"\b(?:laws? of (?:the )?(?:State of )?[A-Z][a-z]+(?: [A-Z][a-z]+)?|[A-Z][a-z]+ (?:Act|Code),? ?\d{0,4}|courts? of [A-Z][a-z]+)",
 "DEFINED TERM": r"[\"“]([A-Z][\w ]{1,30})[\"”]",
}
def ner(text):
    out, seen = [], set()
    for lab, rx in NER_RX.items():
        for m in re.finditer(rx, text):
            val = (m.group(1) if lab == "DEFINED TERM" else m.group(0)).strip(" ,.")
            if (lab, val.lower()) not in seen:
                seen.add((lab, val.lower())); out.append({"text": val, "label": lab})
    if _nlp:
        for e in _nlp(text[:100000]).ents:
            lab = {"ORG": "ORG", "PERSON": "PERSON", "GPE": "LOCATION", "LAW": "LAW/JURISDICTION"}.get(e.label_)
            if lab and (lab, e.text.lower()) not in seen:
                seen.add((lab, e.text.lower())); out.append({"text": e.text, "label": lab})
    return out

# ---------- classification ----------
def classify(clauses):
    P = _clf.predict_proba(_vec.transform(clauses))
    res = []
    for c, p in zip(clauses, P):
        score = dict(zip(_clf.classes_, p))
        low = c.lower()
        for cat, kws in KEYWORDS.items():
            score[cat] += 0.35 * sum(k in low for k in kws)
        cat = max(score, key=score.get)
        conf = score[cat] / (sum(score.values()) or 1)
        res.append(("General", round(conf, 2)) if score[cat] < 0.18 else (cat, round(min(conf * 1.6, 0.99), 2)))
    return res

# ---------- risk detection ----------
def similarity(clauses):
    if _st:
        return np.asarray(_st.encode(clauses, normalize_embeddings=True) @ _risk_emb.T)
    return cosine_similarity(_vec.transform(clauses), _risk_mat)

def detect_risk(clauses):
    sim = similarity(clauses)
    out = []
    for i, c in enumerate(clauses):
        hits = []
        for j, r in enumerate(RISKS):
            rx = bool(re.search(r["rx"], c, re.I))
            s = float(sim[i, j])
            thr = 0.55 if _st else 0.28
            if rx or s >= thr:
                strength = 1.0 if rx and s >= thr else (0.8 if rx else 0.6)
                hits.append(dict(id=r["id"], title=r["title"], severity=r["sev"], similarity=round(s, 2), matched_by="pattern+semantic" if rx and s >= thr else "pattern" if rx else "semantic",
                                 why=r["why"], suggestion=r["fix"], weight=r["sev"] * strength))
        hits.sort(key=lambda h: -h["weight"])
        score = min(10, round(max([h["weight"] for h in hits], default=0) + 0.5 * sum(h["weight"] for h in hits[1:3]) / 3, 1))
        out.append((hits, score))
    return out

# ---------- summarization ----------
def summarize(clauses, k=5):
    if len(clauses) <= k: return clauses
    M = TfidfVectorizer(stop_words="english").fit_transform(clauses)
    cent = np.asarray(M.mean(axis=0))
    sc = np.asarray(M @ cent.T).ravel() + 0.05 * np.array([1 / (1 + i) for i in range(len(clauses))])
    idx = sorted(np.argsort(-sc)[:k])
    return [clauses[i] for i in idx]

# ---------- simplification ----------
def simplify(text, category, hits):
    s = text
    for rx, rep in GLOSSARY.items():
        s = re.sub(rx, rep, s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    s = (s[:260].rsplit(" ", 1)[0] + "…") if len(s) > 280 else s
    note = CAT_EXPLAIN.get(category, "")
    if hits: note += " Watch out: " + hits[0]["why"]
    return {"plain": s, "meaning": note}

OBL = re.compile(r"\b(shall|must|agrees? to|is required to|undertakes?|will)\b", re.I)
def obligations(clauses):
    out = []
    for c in clauses:
        for s in re.split(r"(?<=[.;])\s+", c):
            if OBL.search(s) and len(s) > 30:
                m = re.match(r"(.{3,60}?)\s+(?:shall|must|agrees? to|will|undertakes?)\b", s, re.I)
                out.append({"who": m.group(1).strip(" ,.") if m else "Party", "text": s[:220]})
    return out[:25]

QUERY_SYN = {"fire": "terminate termination notice", "quit": "terminate termination notice", "pay": "payment fee invoice", "sue": "dispute arbitration jurisdiction court",
 "own": "intellectual property assign", "secret": "confidential", "cancel": "terminate renew notice", "money": "payment fee penalty", "compete": "non-compete exclusive solicit"}

def level(score):
    return "High" if score >= 7 else "Medium" if score >= 4 else "Low"

# ---------- main pipeline ----------
def analyze(text, query=None):
    clauses = segment(text)
    if not clauses: return {"error": "No analysable text found."}
    cats = classify(clauses); risks = detect_risk(clauses)
    items = []
    for i, c in enumerate(clauses):
        cat, conf = cats[i]; hits, sc = risks[i]
        items.append(dict(id=i + 1, text=c, category=cat, confidence=conf, risk_score=sc, risk_level=level(sc),
                          risks=hits, entities=ner(c), simple=simplify(c, cat, hits)))
    scores = [x["risk_score"] for x in items]
    overall = round(min(10, 0.6 * max(scores) + 0.4 * (sum(sorted(scores)[-5:]) / min(5, len(scores)))), 1)
    cat_counts = Counter(x["category"] for x in items)
    all_risk = Counter(h["title"] for x in items for h in x["risks"])
    missing = [c for c in ["Termination", "Payment", "Confidentiality", "Liability", "Governing Law", "Dispute Resolution"] if c not in cat_counts]
    words = len(text.split())
    result = dict(stats=dict(words=words, clauses=len(items), read_minutes=max(1, round(words / 220)), high=sum(x["risk_level"] == "High" for x in items),
                             medium=sum(x["risk_level"] == "Medium" for x in items), low=sum(x["risk_level"] == "Low" for x in items)),
                  overall_score=overall, overall_level=level(overall), categories=dict(cat_counts), top_risks=all_risk.most_common(6),
                  missing_clauses=missing, summary=summarize(clauses), obligations=obligations(clauses), entities=ner(text), clauses=items,
                  engine=engine_info())
    if query:  # semantic search over clauses (with legal query expansion)
        for k, v in QUERY_SYN.items():
            if k in query.lower(): query += " " + v
        sims = cosine_similarity(_vec.transform([query]), _vec.transform(clauses)).ravel() if not _st else (_st.encode([query], normalize_embeddings=True) @ _st.encode(clauses, normalize_embeddings=True).T).ravel()
        result["search"] = [dict(id=int(i) + 1, score=round(float(sims[i]), 2)) for i in np.argsort(-sims)[:3] if sims[i] > 0.05]
    return result
