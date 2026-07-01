"""
JobMatch — Semantic job-title recommender.

Embeds the user's free-text interests with the trained BiLSTM encoder and ranks
job titles by cosine similarity against the saved per-title embeddings.

Run:  streamlit run app.py
Needs (produced by day_13.ipynb):
    job_embedding_best.pt
    artifacts/title_embeddings.pt
    artifacts/vocab.json
"""
import json
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

ARTIFACTS = "artifacts"
CKPT = "job_embedding_best.pt"


# ------------------------------------------------------------------
# Model  (identical architecture to the notebook — must match state_dict)
# ------------------------------------------------------------------
class JobEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx, hidden_size=128,
                 num_layers=2, proj_hidden=256, out_dim=128, dropout=0.3):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, proj_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(proj_hidden, out_dim),
        )

    def forward(self, x, lengths=None):
        mask = (x != self.pad_idx).unsqueeze(-1).float()
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        z = self.proj(pooled)
        return F.normalize(z, p=2, dim=1)


# ------------------------------------------------------------------
# Loading (cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_everything():
    with open(f"{ARTIFACTS}/vocab.json") as f:
        cfg = json.load(f)

    model = JobEmbeddingModel(
        vocab_size=cfg["vocab_size"], emb_dim=cfg["emb_dim"], pad_idx=cfg["pad_idx"],
        hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        proj_hidden=cfg["proj_hidden"], out_dim=cfg["out_dim"],
    )
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    bundle = torch.load(f"{ARTIFACTS}/title_embeddings.pt", map_location="cpu")
    title_emb = bundle["title_embeddings"].float()          # (T, 128) unit vectors
    return model, cfg, title_emb, bundle["titles"], bundle["counts"]


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


@torch.no_grad()
def embed_query(text, model, cfg):
    toks = tokenize(text)
    stoi, unk = cfg["stoi"], cfg["unk_idx"]
    ids = [stoi.get(t, unk) for t in toks][: cfg["max_len"]]
    known = sum(1 for t in toks if t in stoi)
    if not ids:
        ids = [unk]
    x = torch.tensor([ids], dtype=torch.long)
    return model(x)[0], known                              # (128,), #known tokens


def rank_titles(q_vec, title_emb, titles, counts, k=5, temperature=0.05):
    sims = title_emb @ q_vec                                # cosine (unit vectors)
    order = torch.argsort(sims, descending=True)[:k]
    top_sims = sims[order]

    # The raw cosine scores all cluster very close to 1.0 (the embeddings live in
    # a narrow cone), so `cosine * 100` made every match read ~100%. Convert the
    # top-k similarities into a relative match score with a temperature-scaled
    # softmax: it amplifies the small gaps into distinct percentages that sum to
    # 100% across the shown results.
    match = torch.softmax(top_sims / temperature, dim=0) * 100.0

    return [
        {
            "title": titles[idx],
            "cosine": float(sims[idx]),
            "match": float(match[rank]),                    # relative match -> %
            "count": counts[idx],
        }
        for rank, idx in enumerate(order.tolist())
    ]


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(page_title="JobMatch · Semantic Recommender",
                   page_icon="◆", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500&display=swap');

:root{
  --bg:#05060d; --panel:rgba(18,22,40,.55); --line:rgba(120,140,255,.18);
  --cyan:#3df2ff; --violet:#8b5cff; --pink:#ff6ad5; --txt:#e7ecff; --dim:#8b93c4;
}
.stApp{
  background:
    radial-gradient(900px 600px at 80% -10%, rgba(139,92,255,.20), transparent 60%),
    radial-gradient(800px 600px at 0% 110%, rgba(61,242,255,.16), transparent 55%),
    #05060d;
  color:var(--txt);
  font-family:'Space Grotesk',sans-serif;
}
#MainMenu, header, footer{visibility:hidden;}
.block-container{padding-top:3.2rem; max-width:840px;}

.hero{ text-align:center; margin-bottom:1.6rem; }
.hero h1{
  font-family:'Orbitron',sans-serif; font-weight:800; font-size:2.7rem;
  letter-spacing:2px; margin:0;
  background:linear-gradient(90deg,var(--cyan),var(--violet) 55%,#ff6ad5);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 26px rgba(139,92,255,.45));
}
.hero p{ color:var(--dim); font-size:.98rem; letter-spacing:.5px; margin-top:.5rem; }
.badge{
  display:inline-block; margin-top:.7rem; padding:.28rem .8rem; border-radius:999px;
  border:1px solid var(--line); background:rgba(61,242,255,.06);
  font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--cyan);
  letter-spacing:1px;
}

/* input */
.stTextArea textarea{
  background:var(--panel)!important; color:var(--txt)!important;
  border:1px solid var(--line)!important; border-radius:16px!important;
  font-family:'Space Grotesk',sans-serif!important; font-size:1rem!important;
  box-shadow:inset 0 0 40px rgba(80,110,255,.06); backdrop-filter:blur(12px);
}
.stTextArea textarea:focus{
  border-color:var(--cyan)!important;
  box-shadow:0 0 0 1px var(--cyan), 0 0 26px rgba(61,242,255,.25)!important;
}
.stButton>button{
  width:100%; border:none; border-radius:14px; padding:.75rem 0;
  font-family:'Orbitron',sans-serif; font-weight:600; letter-spacing:2px;
  color:#05060d; background:linear-gradient(90deg,var(--cyan),var(--violet));
  box-shadow:0 0 30px rgba(139,92,255,.45); transition:.2s; cursor:pointer;
}
.stButton>button:hover{ filter:brightness(1.12); box-shadow:0 0 44px rgba(61,242,255,.55); }

/* result card */
.card{
  position:relative; margin:.8rem 0; padding:1.05rem 1.25rem;
  border:1px solid var(--line); border-radius:18px; background:var(--panel);
  backdrop-filter:blur(14px); overflow:hidden;
  animation:rise .5s cubic-bezier(.2,.7,.2,1) both;
}
@keyframes rise{from{opacity:0; transform:translateY(14px);}to{opacity:1;transform:none;}}
.card .row{ display:flex; align-items:center; justify-content:space-between; gap:1rem; }
.rank{
  font-family:'Orbitron',sans-serif; font-size:.8rem; color:var(--cyan);
  border:1px solid var(--line); border-radius:9px; padding:.15rem .5rem; margin-right:.7rem;
}
.tname{ font-weight:700; font-size:1.18rem; letter-spacing:.3px; }
.tmeta{ color:var(--dim); font-size:.76rem; font-family:'JetBrains Mono',monospace; margin-top:.15rem; }
.pct{
  font-family:'Orbitron',sans-serif; font-weight:800; font-size:1.7rem;
  background:linear-gradient(90deg,var(--cyan),#ff6ad5);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.bar{ margin-top:.75rem; height:8px; border-radius:99px; background:rgba(255,255,255,.06); overflow:hidden; }
.fill{
  height:100%; border-radius:99px;
  background:linear-gradient(90deg,var(--cyan),var(--violet),#ff6ad5);
  box-shadow:0 0 16px rgba(61,242,255,.5); animation:grow 1s ease-out both;
}
@keyframes grow{from{width:0;}}
.hint{ color:var(--dim); font-size:.8rem; text-align:center; margin-top:2rem;
  font-family:'JetBrains Mono',monospace; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>JOBMATCH</h1>
      <p>Describe what you love building. The encoder maps it into a 128-D semantic space
      and finds your closest job titles.</p>
      <span class="badge">METRIC · COSINE SIMILARITY · BiLSTM EMBEDDINGS</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, cfg, title_emb, titles, counts = load_everything()
except FileNotFoundError:
    st.error("Artifacts not found. Run the day_13.ipynb cells first to create "
             "`job_embedding_best.pt` and the `artifacts/` folder.")
    st.stop()

query = st.text_area(
    "interests",
    height=150, label_visibility="collapsed",
    placeholder="e.g. I love building data pipelines in Python and SQL, training ML "
                "models with PyTorch, deploying on AWS. Interested in NLP and "
                "recommendation systems. Target salary 90k–120k.",
)
go = st.button("◆  FIND MY MATCHES")

if go:
    if not query.strip():
        st.warning("Type a few sentences about your interests first.")
        st.stop()

    q_vec, known = embed_query(query, model, cfg)
    if known == 0:
        st.warning("None of your words are in the model's vocabulary — "
                   "try describing your skills in more detail.")
        st.stop()

    results = rank_titles(q_vec, title_emb, titles, counts, k=5)

    st.markdown(f"<div class='hint'>{known} recognized tokens · "
                f"searched {len(titles)} job titles</div>", unsafe_allow_html=True)

    for rank, r in enumerate(results, 1):
        st.markdown(
            f"""
            <div class="card">
              <div class="row">
                <div>
                  <span class="rank">#{rank}</span>
                  <span class="tname">{r['title']}</span>
                  <div class="tmeta">cosine {r['cosine']:.3f} · {r['count']} descriptions in corpus</div>
                </div>
                <div class="pct">{r['match']:.1f}%</div>
              </div>
              <div class="bar"><div class="fill" style="width:{min(r['match'],100):.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown("<div class='hint'>▁▁▁ awaiting input ▁▁▁</div>", unsafe_allow_html=True)
