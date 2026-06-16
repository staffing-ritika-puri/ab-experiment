import os

# ---------------------------------------------------------------------------
# Force HF / DeepEval offline + telemetry-off BEFORE the backend is imported.
# Without this the first "Connect & Load Models" click can hang ~50 s on
# huggingface.co SSL retries on networks with intercepted certs.
# (The backend module sets the same vars too, but they must be in place
# before its top-level `from transformers import ...` and `from bert_score
# import BERTScorer` lines execute.)
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("CONFIDENT_AI_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import importlib.util
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
import streamlit as st

st.set_page_config(
    page_title="A/B Experiment Runner",
    page_icon="x",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Base ---------- */
html, body, [class*="css"]  { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }
.main .block-container      { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1280px; }
[data-testid="stHeader"]    { background: transparent; }
hr, .stDivider              { margin: 1.4rem 0; border-color: rgba(148,163,184,.18); }

/* ---------- Hero ---------- */
.hero-banner{
  position:relative; overflow:hidden;
  background:
    radial-gradient(1200px 300px at 0% 0%,  rgba(96,192,255,.18), transparent 60%),
    radial-gradient(1000px 300px at 100% 100%, rgba(192,96,255,.18), transparent 60%),
    linear-gradient(135deg,#0b0a1f 0%,#1a1740 50%,#0b0a1f 100%);
  border:1px solid rgba(148,163,184,.18);
  border-radius:18px; padding:1.6rem 2rem; margin-bottom:1.25rem; color:#e8eaf6;
}
.hero-banner h1{ font-size:1.9rem; font-weight:800; margin:0 0 .25rem 0; letter-spacing:-.01em;
  background:linear-gradient(90deg,#9ec5ff,#d2a3ff,#9ec5ff); -webkit-background-clip:text; background-clip:text; color:transparent;
  background-size:200% 100%; animation: hueShift 10s ease-in-out infinite; }
.hero-banner p{ font-size:.96rem; opacity:.78; margin:0 0 .85rem 0; }
.hero-pills{ display:flex; gap:.5rem; flex-wrap:wrap; }
.hero-pill{ display:inline-flex; align-items:center; gap:.4rem; padding:.32rem .7rem; border-radius:99px;
  font-size:.78rem; font-weight:600; background:rgba(148,163,184,.10); border:1px solid rgba(148,163,184,.18); color:#cbd5e1; }
.hero-pill.ok{ background:rgba(34,197,94,.12); border-color:rgba(34,197,94,.35); color:#86efac; }
.hero-pill.warn{ background:rgba(234,179,8,.12); border-color:rgba(234,179,8,.35); color:#fcd34d; }
.hero-pill .dot{ width:.5rem; height:.5rem; border-radius:99px; background:currentColor; box-shadow:0 0 6px currentColor; }
@keyframes hueShift { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }

/* ---------- Stepper ---------- */
.stepper{ display:flex; gap:.6rem; margin:0 0 1.1rem 0; flex-wrap:wrap; }
.step{
  flex:1 1 0; min-width:160px;
  background:#11122a; border:1px solid rgba(148,163,184,.14); border-radius:12px;
  padding:.7rem .9rem; color:#94a3b8; position:relative; overflow:hidden;
}
.step .num{
  display:inline-flex; align-items:center; justify-content:center;
  width:1.4rem; height:1.4rem; border-radius:99px; font-size:.78rem; font-weight:700;
  background:#1f2547; color:#94a3b8; margin-right:.55rem;
}
.step .label{ font-size:.95rem; font-weight:600; color:#e2e8f0; }
.step .sub{ font-size:.77rem; opacity:.7; margin-top:.15rem; }
.step.active{ border-color:rgba(96,192,255,.45); box-shadow:0 0 0 1px rgba(96,192,255,.25), 0 8px 30px -10px rgba(96,192,255,.35); background:linear-gradient(180deg,#13214a 0%,#0f1530 100%); }
.step.active .num{ background:linear-gradient(135deg,#60c0ff,#c060ff); color:#0b0a1f; }
.step.done{ border-color:rgba(34,197,94,.35); }
.step.done .num{ background:#14532d; color:#86efac; }

/* ---------- Section headers ---------- */
.section-head{
  background:linear-gradient(180deg,#101127 0%,#0c0d20 100%);
  border:1px solid rgba(148,163,184,.14); border-left:3px solid #60c0ff;
  border-radius:10px; padding:.75rem 1rem; margin:1.1rem 0 .75rem 0;
}
.section-head h3.section-title{
  font-size:1.02rem; font-weight:700; color:#e2e8f0; margin:0;
  display:flex; align-items:center; gap:.55rem;
}
.section-head h3.section-title::before{
  content:""; width:.45rem; height:.45rem; border-radius:99px;
  background:linear-gradient(135deg,#60c0ff,#c060ff); box-shadow:0 0 8px rgba(150,150,255,.6);
}
.section-head .helper{ font-size:.82rem; color:#94a3b8; margin:.2rem 0 0 0; }

/* ---------- Model headers ---------- */
.model-header{
  font-size:.95rem; font-weight:700; padding:.55rem .9rem; border-radius:10px;
  margin-bottom:.85rem; letter-spacing:.01em; display:flex; align-items:center; gap:.55rem;
}
.model-header .pill{ font-size:.7rem; padding:.1rem .5rem; border-radius:99px; background:rgba(255,255,255,.08); }
.model-a{ background:linear-gradient(90deg,#0e2a47,#143a66); color:#9ed7ff; border:1px solid rgba(96,192,255,.25); }
.model-b{ background:linear-gradient(90deg,#2c1352,#43206e); color:#dab8ff; border:1px solid rgba(192,96,255,.25); }

/* ---------- Score badges ---------- */
.score-badge{
  display:inline-block; padding:.28rem .7rem; border-radius:99px;
  font-size:.78rem; font-weight:700; margin:0 .35rem .35rem 0;
  border:1px solid transparent;
}
.score-good { background:rgba(34,197,94,.12);  color:#86efac; border-color:rgba(34,197,94,.35); }
.score-ok   { background:rgba(234,179,8,.12);  color:#fcd34d; border-color:rgba(234,179,8,.35); }
.score-poor { background:rgba(239,68,68,.12);  color:#fca5a5; border-color:rgba(239,68,68,.35); }

/* ---------- Winner ---------- */
.winner-box{
  position:relative; overflow:hidden;
  background:linear-gradient(90deg,rgba(20,83,45,.55),rgba(22,101,52,.45));
  border:1px solid rgba(34,197,94,.45); border-radius:14px;
  padding:1.2rem 1.4rem; text-align:center; color:#bbf7d0;
  font-size:1.15rem; font-weight:700; margin:.4rem 0 1rem 0;
  box-shadow:0 12px 40px -16px rgba(34,197,94,.45);
}
.winner-box .crown{
  display:inline-block; padding:.15rem .55rem; border-radius:99px;
  background:rgba(255,255,255,.10); color:#f0fdf4; font-size:.72rem; letter-spacing:.08em;
  margin-right:.6rem; text-transform:uppercase;
}
.winner-box .margin{ display:block; font-size:.82rem; opacity:.85; margin-top:.3rem; font-weight:500; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{ background:#0a0b1c; border-right:1px solid rgba(148,163,184,.10); }
section[data-testid="stSidebar"] .sb-title{ font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; color:#94a3b8; font-weight:700; margin:.6rem 0 .35rem; }
.sb-pill{ display:inline-flex; align-items:center; gap:.35rem; padding:.22rem .55rem; border-radius:99px;
  font-size:.72rem; font-weight:600; background:rgba(148,163,184,.10); border:1px solid rgba(148,163,184,.18); color:#cbd5e1; }
.sb-pill.ok  { background:rgba(34,197,94,.12);  color:#86efac; border-color:rgba(34,197,94,.35); }
.sb-pill.warn{ background:rgba(234,179,8,.12);  color:#fcd34d; border-color:rgba(234,179,8,.35); }
.sb-pill .dot{ width:.45rem; height:.45rem; border-radius:99px; background:currentColor; box-shadow:0 0 5px currentColor; }
.sb-model-row{ font-family:"JetBrains Mono","Consolas",monospace; font-size:.78rem; color:#cbd5e1; padding:.12rem 0; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:.25rem; }
.stTabs [data-baseweb="tab"]{ background:#11122a; border-radius:10px 10px 0 0; padding:.55rem 1rem; }
.stTabs [aria-selected="true"]{ background:linear-gradient(180deg,#19204a,#11122a); border-bottom:2px solid #9ec5ff !important; }

/* ---------- Buttons ---------- */
.stButton > button[kind="primary"]{
  background:linear-gradient(135deg,#3b82f6,#8b5cf6); border:0; font-weight:700; letter-spacing:.01em;
  box-shadow:0 10px 28px -10px rgba(99,102,241,.55);
}
.stButton > button[kind="primary"]:hover{ filter:brightness(1.08); transform:translateY(-1px); }

/* ---------- Metrics ---------- */
[data-testid="stMetric"]{ background:#0e0f25; border:1px solid rgba(148,163,184,.12); border-radius:10px; padding:.6rem .8rem; }
[data-testid="stMetricLabel"]{ color:#94a3b8 !important; font-size:.78rem !important; font-weight:600 !important; }
[data-testid="stMetricValue"]{ color:#e2e8f0 !important; font-size:1.15rem !important; font-weight:700 !important; }

/* ---------- Status banners ---------- */
.ready-banner{ display:flex; align-items:center; gap:.65rem; padding:.85rem 1.1rem; border-radius:12px;
  background:rgba(34,197,94,.1); border:1px solid rgba(34,197,94,.35); color:#bbf7d0; font-weight:600; }
.notready-banner{ display:flex; align-items:center; gap:.65rem; padding:.85rem 1.1rem; border-radius:12px;
  background:rgba(234,179,8,.10); border:1px solid rgba(234,179,8,.35); color:#fde68a; font-weight:600; }
.ready-banner .dot, .notready-banner .dot{ width:.55rem; height:.55rem; border-radius:99px; background:currentColor; box-shadow:0 0 7px currentColor; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Backend loader
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

@st.cache_resource(show_spinner="Loading NLP models...")
def load_backend():
    """Import the backend module ONCE and load the heavy NLP models.

    The provider/API-key/URL are NOT baked in here — the backend's module-level
    init is resilient and we (re)configure the LLM client afterwards via
    `configure_llm_provider()`. This keeps the expensive NLP models cached even
    when the user switches provider, key, or URL.
    """
    # A placeholder so the resilient backend init never blocks on input().
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "") or "sk-ui-placeholder")
    backend_path = (SCRIPT_DIR / "ABExperimentFixes.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "ab_backend", str(backend_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load backend module from {backend_path}")

    mod = importlib.util.module_from_spec(spec)
    import builtins
    _real = builtins.input
    builtins.input = lambda *a, **k: ""
    old_cwd = os.getcwd()
    try:
        os.chdir(str(SCRIPT_DIR.parent.resolve()))
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old_cwd)
        builtins.input = _real
    return mod

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _colour(v):
    if v is None:
        return "score-ok"
    return "score-good" if float(v) >= 4 else ("score-ok" if float(v) >= 3 else "score-poor")

def badge(lbl, v):
    if v is None:
        return f"<span class='score-badge score-ok'>{lbl}: N/A</span>"
    return f"<span class='score-badge {_colour(v)}'>{lbl}: {float(v):.2f}/5</span>"

def section_header(title, helper=None):
    """Render a styled section header (title + optional helper text)."""
    helper_html = f"<p class='helper'>{helper}</p>" if helper else ""
    st.markdown(
        f"<div class='section-head'><h3 class='section-title'>{title}</h3>{helper_html}</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_defaults = dict(
    api_key=os.getenv("OPENAI_API_KEY",""),
    # ----- LLM provider selection (OpenAI vs Portkey) -----
    provider="OpenAI models",                              # dropdown label
    openai_base_url=os.getenv("OPENAI_BASE_URL", ""),      # optional custom OpenAI URL
    portkey_base_url=os.getenv("PORTKEY_BASE_URL", "https://api.portkey.ai/v1"),
    backend=None, valid_models=[], results=[],
    log_lines=[], running=False,
    dashboard_html=None, last_json_path=None,
    task_type="summarization",
    source="", topic="", taxonomy_text="",
    cfg_task=None, cfg_model1=None, cfg_model2=None,
    cfg_temp1=0.7, cfg_top_p1=1.0,
    cfg_temp2=0.7, cfg_top_p2=1.0,
    cfg_length="150-200 words",
    cfg_num_bullets=None, cfg_max_tokens=1500,
    # ----- Compare-with-reference-LLM-output (summarization only) -----
    compare_with_llm=False,
    llm_ref_input_method="Paste LLM Output",   # or "Upload LLM Output"
    llm_ref_text="",                            # parsed reference summary text
    llm_ref_source="",                          # filename or "pasted"
    llm_ref_parse_note="",                      # human-readable parse note
    llm_compare_mode="Both selected models",    # or "Only one selected model"
    llm_selected_model_role="Model A",
    llm_compare_depth="Fast",
    llm_judge_enabled=False,                    # toggle the (paid) LLM-judge call
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sb-title'>Connection</div>", unsafe_allow_html=True)
    if st.session_state.backend:
        st.markdown(
            f"<span class='sb-pill ok'><span class='dot'></span>Connected · {len(st.session_state.valid_models)} models</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span class='sb-pill warn'><span class='dot'></span>Not connected</span>",
            unsafe_allow_html=True,
        )

    # ---- Provider selector ----
    st.markdown("<div class='sb-title' style='margin-top:1rem'>Provider</div>", unsafe_allow_html=True)
    provider_label = st.selectbox(
        "Provider",
        ["OpenAI models", "Portkey models"],
        index=1 if st.session_state.get("provider") == "Portkey models" else 0,
        label_visibility="collapsed",
        help="Choose where models are served from. Both use the OpenAI-compatible API.",
    )
    st.session_state["provider"] = provider_label
    is_portkey = provider_label == "Portkey models"

    # ---- API key (label adapts to provider) ----
    key_title = "Portkey API Key" if is_portkey else "OpenAI API Key"
    st.markdown(f"<div class='sb-title' style='margin-top:1rem'>{key_title}</div>", unsafe_allow_html=True)
    key_in = st.text_input(
        key_title, value=st.session_state.api_key,
        type="password", label_visibility="collapsed",
        placeholder="pk-..." if is_portkey else "sk-...",
        help=(
            "Your Portkey API key (sent as the gateway token)."
            if is_portkey else
            "Your OpenAI secret key."
        ),
    )

    # ---- Endpoint URL (Portkey URL or optional OpenAI URL) ----
    if is_portkey:
        st.markdown("<div class='sb-title' style='margin-top:1rem'>Portkey URL</div>", unsafe_allow_html=True)
        url_in = st.text_input(
            "Portkey URL", value=st.session_state.get("portkey_base_url", ""),
            label_visibility="collapsed",
            placeholder="https://api.portkey.ai/v1",
            help="Portkey gateway base URL. Models are fetched from this endpoint.",
        )
        st.session_state["portkey_base_url"] = url_in
    else:
        st.markdown("<div class='sb-title' style='margin-top:1rem'>OpenAI URL (optional)</div>", unsafe_allow_html=True)
        url_in = st.text_input(
            "OpenAI URL", value=st.session_state.get("openai_base_url", ""),
            label_visibility="collapsed",
            placeholder="https://api.openai.com/v1 (leave blank for default)",
            help="Override only if you use an OpenAI-compatible proxy/gateway.",
        )
        st.session_state["openai_base_url"] = url_in

    if st.button("Connect & Load Models", type="primary", use_container_width=True):
        provider_id = "portkey" if is_portkey else "openai"
        with st.spinner("Connecting..."):
            try:
                st.session_state.api_key = key_in
                # NLP models load once and stay cached; only the lightweight
                # LLM client + model list is (re)configured per connect.
                be = load_backend()
                if provider_id == "portkey":
                    ok, err, models = be.configure_llm_provider(
                        provider="portkey",
                        api_key=key_in,
                        base_url=(url_in or None),
                        portkey_api_key=key_in,
                    )
                else:
                    ok, err, models = be.configure_llm_provider(
                        provider="openai",
                        api_key=key_in,
                        base_url=(url_in or None),
                    )
                if ok:
                    st.session_state.backend = be
                    st.session_state.valid_models = models
                    st.success(
                        f"Connected to {provider_label} · {len(models)} models loaded"
                    )
                    st.rerun()
                else:
                    st.session_state.backend = None
                    st.session_state.valid_models = models or []
                    st.error(f"Could not connect to {provider_label}: {err}")
            except Exception as e:
                st.error(f"Failed: {e}")
                with st.expander("Show technical details", expanded=False):
                    st.code(traceback.format_exc(), language="text")

    if st.session_state.valid_models:
        st.markdown("<div class='sb-title' style='margin-top:1rem'>Available Models</div>", unsafe_allow_html=True)
        with st.expander(f"Show all ({len(st.session_state.valid_models)})", expanded=False):
            for m in st.session_state.valid_models:
                st.markdown(f"<div class='sb-model-row'>· {m}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-title' style='margin-top:1rem'>NLP Components</div>", unsafe_allow_html=True)
    if st.session_state.backend:
        mo = st.session_state.backend.MODELS
        def _pill(name, ok):
            cls = "ok" if ok else "warn"
            txt = "Loaded" if ok else "Fallback"
            return f"<div style='margin:.18rem 0'><span class='sb-pill {cls}'><span class='dot'></span>{name}: {txt}</span></div>"
        st.markdown(
            _pill("SpaCy",    bool(mo.get("nlp"))) +
            _pill("BERT",     bool(mo.get("bert_model"))) +
            _pill("NLI",      bool(mo.get("nli_model"))) +
            _pill("BERTScore",bool(mo.get("bert_scorer"))),
            unsafe_allow_html=True,
        )
    else:
        st.caption("Connect to load NLP component status.")

    st.markdown("<div class='sb-title' style='margin-top:1rem'>Tip</div>", unsafe_allow_html=True)
    st.caption("Use Configure → Run → Results → Dashboard. Each step lights up as you progress.")

# ---------------------------------------------------------------------------
# Header + Stepper
# ---------------------------------------------------------------------------
connected      = bool(st.session_state.backend)
configured     = bool(st.session_state.get("cfg_task") and st.session_state.get("cfg_model1") and st.session_state.get("cfg_model2"))
has_results    = bool(st.session_state.get("results"))

conn_pill_cls  = "ok"   if connected else "warn"
conn_pill_txt  = f"Connected · {len(st.session_state.valid_models)} models" if connected else "Not connected"

cfg_pill_cls   = "ok"   if configured else "warn"
cfg_pill_txt   = "Configured" if configured else "Not configured"

res_pill_cls   = "ok"   if has_results else "warn"
res_pill_txt   = "Results ready" if has_results else "No run yet"

st.markdown(f"""
<div class="hero-banner">
  <h1>A/B Model Experiment Runner</h1>
  <p>Compare two OpenAI models head-to-head across Summarization, Generation, and Entity Extraction with rich NLP evaluation.</p>
  <div class="hero-pills">
    <span class="hero-pill {conn_pill_cls}"><span class="dot"></span>{conn_pill_txt}</span>
    <span class="hero-pill {cfg_pill_cls}"><span class="dot"></span>{cfg_pill_txt}</span>
    <span class="hero-pill {res_pill_cls}"><span class="dot"></span>{res_pill_txt}</span>
  </div>
</div>
""", unsafe_allow_html=True)

def _step_state(active, done):
    cls = ""
    if done:   cls += " done"
    if active: cls += " active"
    return cls

# Decide which step is "active": the next incomplete one.
if not connected:        active_step = 1
elif not configured:     active_step = 2
elif not has_results:    active_step = 3
else:                    active_step = 4

steps = [
    (1, "Connect",   "Add OpenAI API key",        connected),
    (2, "Configure", "Pick task, models, inputs", configured),
    (3, "Run",       "Execute the comparison",    has_results),
    (4, "Review",    "Inspect scores & winner",   False),
]
step_html = "<div class='stepper'>"
for n, label, sub, done in steps:
    cls = _step_state(active_step == n, done)
    step_html += (
        f"<div class='step{cls}'>"
        f"<span class='num'>{n}</span>"
        f"<span class='label'>{label}</span>"
        f"<div class='sub'>{sub}</div>"
        f"</div>"
    )
step_html += "</div>"
st.markdown(step_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
t_cfg, t_run, t_res, t_dash = st.tabs(["Configure", "Run Experiment", "Results", "Dashboard"])

# ===========================================================================
# TAB: Configure
# ===========================================================================
with t_cfg:
    if not st.session_state.backend:
        st.info("Enter your OpenAI API key in the sidebar and click **Connect & Load Models** to get started.")
        st.stop()

    be = st.session_state.backend
    vm = st.session_state.valid_models
    tc = be.TASK_CONFIGS

    # ---- Task type ----
    section_header("1 · Choose Task Type", "Pick what you want the models to do.")
    task_type = st.radio(
        "Task",
        options=list(tc.keys()),
        format_func=lambda t: {
            "summarization":     "Summarization",
            "generation":        "Generation",
            "entity_extraction": "Entity Extraction",
        }.get(t, t.title()),
        horizontal=True,
        label_visibility="collapsed",
        key="task_radio",
    )
    st.session_state["task_type"] = task_type

    notes = tc[task_type].get("evaluation_notes", {})
    w     = tc[task_type].get("effectiveness_weights", {})
    with st.expander("How this task is evaluated", expanded=False):
        c1, c2 = st.columns(2)
        c1.markdown(f"**Accuracy** &nbsp;·&nbsp; {notes.get('accuracy','N/A')}")
        c1.markdown(f"**Relevancy** &nbsp;·&nbsp; {notes.get('relevancy','N/A')}")
        c2.markdown(f"**Special** &nbsp;·&nbsp; {notes.get('special','N/A')}")
        c2.markdown(
            f"**Effectiveness Weights** &nbsp;·&nbsp; "
            f"Accuracy × {w.get('accuracy',0.5)}  +  Relevancy × {w.get('relevancy',0.5)}"
        )

    # ---- Models ----
    section_header("2 · Select Models", "Two different OpenAI models will be compared side-by-side.")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(
            '<div class="model-header model-a">Model A <span class="pill">Baseline</span></div>',
            unsafe_allow_html=True,
        )
        idx_a = vm.index("gpt-4o") if "gpt-4o" in vm else 0
        model1 = st.selectbox("Model A", vm, index=idx_a, key="sel_model1", label_visibility="collapsed")
    with mc2:
        st.markdown(
            '<div class="model-header model-b">Model B <span class="pill">Challenger</span></div>',
            unsafe_allow_html=True,
        )
        dflt_b = "gpt-4o-mini" if "gpt-4o-mini" in vm else (vm[1] if len(vm) > 1 else vm[0])
        idx_b = vm.index(dflt_b)
        model2 = st.selectbox("Model B", vm, index=idx_b, key="sel_model2", label_visibility="collapsed")

    if model1 == model2:
        st.warning("Pick two **different** models — Model A and Model B are currently the same.")

    # ---- Parameters ----
    section_header("3 · Tune Model Parameters",
                 "Higher temperature = more creative. Top-P narrows the sampling pool to the most likely tokens.")
    pa1, pa2 = st.columns(2)
    with pa1:
        st.markdown(f"<div style='font-weight:600;color:#9ed7ff;margin-bottom:.3rem'>Model A · {model1}</div>", unsafe_allow_html=True)
        temp1  = st.slider("Temperature A", 0.0, 2.0, 0.7, 0.05, key="sl_t1")
        top_p1 = st.slider("Top-P A",       0.0, 1.0, 1.0, 0.05, key="sl_p1")
    with pa2:
        st.markdown(f"<div style='font-weight:600;color:#dab8ff;margin-bottom:.3rem'>Model B · {model2}</div>", unsafe_allow_html=True)
        temp2  = st.slider("Temperature B", 0.0, 2.0, 0.7, 0.05, key="sl_t2")
        top_p2 = st.slider("Top-P B",       0.0, 1.0, 1.0, 0.05, key="sl_p2")

    # ---- Output length ----
    section_header("4 · Output Length & Tokens", "Controls how long each generated answer is allowed to be.")
    if task_type == "entity_extraction":
        st.info("Entity Extraction uses a fixed structured JSON format — `max_tokens` is locked at 4000.")
        length_str  = "entities only"
        num_bullets = None
        max_tokens  = 4000
    else:
        lc1, lc2, lc3 = st.columns([2, 1, 1])
        with lc1:
            lmode = st.radio(
                "Length mode",
                ["Bullet points", "Word count", "Custom tokens"],
                horizontal=True,
                key="lmode",
            )
        with lc2:
            if lmode == "Bullet points":
                nb = int(st.number_input("Bullets", 2, 20, 5, key="nb_input"))
                max_tokens  = min(4000, max(200, nb * 80))
                length_str  = f"{nb} bullet points"
                num_bullets = nb
            elif lmode == "Word count":
                wc = int(st.number_input("Words", 50, 2000, 200, key="wc_input"))
                max_tokens  = min(4000, max(300, int(wc * 1.3)))
                length_str  = f"{wc} words"
                num_bullets = None
            else:
                max_tokens  = int(st.number_input("Max tokens", 50, 4000, 1500, key="mt_input"))
                length_str  = f"~{int(max_tokens/1.3):.0f} words"
                num_bullets = None
        with lc3:
            if lmode != "Custom tokens":
                ov = int(st.number_input("Override tokens", 50, 4000, int(max_tokens), key="ov_input"))
                max_tokens = ov
            st.metric("Max tokens", max_tokens)

    st.session_state.update(dict(length=length_str, num_bullets=num_bullets, max_tokens=max_tokens))

    # ---- Task input ----
    section_header("5 · Provide Task Input", "Paste the source content (or upload a file) the models will work on.")

    if task_type == "summarization":
        src_m = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True, key="src_m")
        if src_m == "Paste text":
            src_txt = st.text_area(
                "Source text", height=240,
                placeholder="Paste the document you want to summarise...",
                key="src_paste",
            )
        else:
            up = st.file_uploader("Upload .txt", type=["txt"], key="src_up")
            src_txt = up.read().decode("utf-8") if up else ""
            if src_txt:
                st.text_area("Preview", src_txt[:500], height=130, disabled=True)
        st.session_state["source"] = src_txt
        st.session_state["topic"]  = None

        # ---- 6 · Compare with reference LLM output (optional) ----
        section_header(
            "6 · Compare With Reference LLM Output (optional)",
            "Provide a reference summary (e.g. your in-house LLM's output). Each "
            "model's summary will be scored against it using a multi-signal "
            "comparison (lexical + semantic + NLI + DeepEval + LLM-as-judge).",
        )
        compare_choice = st.radio(
            "Compare model output with a reference LLM output?",
            ["No", "Yes"],
            horizontal=True,
            index=1 if st.session_state.get("compare_with_llm") else 0,
            key="compare_choice",
        )
        st.session_state["compare_with_llm"] = (compare_choice == "Yes")

        if st.session_state["compare_with_llm"]:
            st.session_state["llm_compare_mode"] = st.radio(
                "Which model output should be compared with the reference LLM output?",
                ["Both selected models", "Only one selected model"],
                horizontal=True,
                index=1 if st.session_state.get("llm_compare_mode") == "Only one selected model" else 0,
                key="llm_compare_mode_radio",
                help=(
                    "Model output is always generated during this run. "
                    "No pasted model output is used for comparison."
                ),
            )

            if st.session_state["llm_compare_mode"] == "Only one selected model":
                st.session_state["llm_selected_model_role"] = st.radio(
                    "Choose the selected model to run and compare",
                    ["Model A", "Model B"],
                    horizontal=True,
                    index=1 if st.session_state.get("llm_selected_model_role") == "Model B" else 0,
                    key="llm_selected_model_role_radio",
                    format_func=lambda role: (
                        f"{role} · {model1}" if role == "Model A" else f"{role} · {model2}"
                    ),
                )
            else:
                st.session_state["llm_selected_model_role"] = "Both"

            st.session_state["llm_compare_depth"] = st.radio(
                "Reference comparison depth",
                ["Fast", "Full audit"],
                horizontal=True,
                index=1 if st.session_state.get("llm_compare_depth") == "Full audit" else 0,
                key="llm_compare_depth_radio",
                help=(
                    "Fast is recommended for normal runs: lexical, semantic, "
                    "embedding, and NLI signals only. Full audit adds extra "
                    "DeepEval/source/judge calls and can take much longer."
                ),
            )
            if st.session_state["llm_compare_depth"] == "Fast":
                st.caption(
                    "Fast mode skips extra DeepEval pair/source audit calls for "
                    "the reference comparison. Per-model Accuracy, Relevancy, "
                    "and Effectiveness are still computed."
                )

            llm_ref_method = st.radio(
                "How would you like to provide the reference output?",
                ["Paste LLM Output", "Upload LLM Output"],
                horizontal=True,
                index=1 if st.session_state.get("llm_ref_input_method") == "Upload LLM Output" else 0,
                key="llm_ref_method",
            )
            st.session_state["llm_ref_input_method"] = llm_ref_method

            if llm_ref_method == "Paste LLM Output":
                llm_ref_pasted = st.text_area(
                    "Reference LLM output",
                    value=st.session_state.get("llm_ref_text", "")
                    if st.session_state.get("llm_ref_source") == "pasted" else "",
                    height=200,
                    placeholder="Paste the reference summary produced by your LLM...",
                    key="llm_ref_paste",
                )
                if llm_ref_pasted and llm_ref_pasted.strip():
                    st.session_state["llm_ref_text"] = llm_ref_pasted.strip()
                    st.session_state["llm_ref_source"] = "pasted"
                    st.session_state["llm_ref_parse_note"] = "Pasted text"
                else:
                    if st.session_state.get("llm_ref_source") == "pasted":
                        st.session_state["llm_ref_text"] = ""
                        st.session_state["llm_ref_parse_note"] = ""
            else:  # Upload
                llm_ref_up = st.file_uploader(
                    "Upload reference LLM output (.txt, .json, .docx)",
                    type=["txt", "json", "docx"],
                    key="llm_ref_up",
                )
                if llm_ref_up is not None:
                    raw = llm_ref_up.read()
                    parsed_text, note = be.parse_reference_summary_file(
                        llm_ref_up.name, raw
                    )
                    if parsed_text:
                        st.session_state["llm_ref_text"] = parsed_text
                        st.session_state["llm_ref_source"] = llm_ref_up.name
                        st.session_state["llm_ref_parse_note"] = note
                    else:
                        st.session_state["llm_ref_text"] = ""
                        st.session_state["llm_ref_source"] = ""
                        st.session_state["llm_ref_parse_note"] = note
                        st.error(f"Could not parse `{llm_ref_up.name}` — {note}")

            ref_preview = st.session_state.get("llm_ref_text", "")
            if ref_preview:
                wc = len(ref_preview.split())
                cc = len(ref_preview)
                src_lbl = st.session_state.get("llm_ref_source") or "—"
                note_lbl = st.session_state.get("llm_ref_parse_note") or ""
                st.caption(
                    f"Reference loaded · {wc} words · {cc:,} chars · "
                    f"source: {src_lbl}{(' · ' + note_lbl) if note_lbl else ''}"
                )
                with st.expander("Preview reference summary", expanded=False):
                    st.text_area(
                        "Reference preview", ref_preview[:2000], height=160,
                        disabled=True, label_visibility="collapsed",
                    )

            judge_col1, judge_col2 = st.columns([3, 2])
            with judge_col1:
                st.session_state["llm_judge_enabled"] = st.checkbox(
                    "Also run LLM-as-judge pairwise comparison "
                    "(uses your judge model, slightly more cost)",
                    value=(
                        st.session_state.get("llm_judge_enabled", False)
                        and st.session_state.get("llm_compare_depth") == "Full audit"
                    ),
                    key="llm_judge_chk",
                    disabled=st.session_state.get("llm_compare_depth") == "Fast",
                )
                if st.session_state.get("llm_compare_depth") == "Fast":
                    st.session_state["llm_judge_enabled"] = False
            with judge_col2:
                judge_model_default = (
                    getattr(be, "_LLM_JUDGE_DEFAULT_MODEL", None)
                    or os.getenv("LLM_JUDGE_MODEL")
                    or "gpt-4o-mini"
                )
                st.caption(f"Judge model: `{judge_model_default}` "
                           "(set `LLM_JUDGE_MODEL` env var to override)")

    elif task_type == "generation":
        tp_txt = st.text_area(
            "Topic / Prompt", height=160,
            placeholder="e.g. Write a balanced overview of how AI is reshaping healthcare...",
            key="tp_txt",
        )
        st.session_state["source"] = None
        st.session_state["topic"]  = tp_txt

    elif task_type == "entity_extraction":
        ee1, ee2 = st.columns(2)
        with ee1:
            st.markdown("**Taxonomy** &nbsp; *(ground-truth entities)*", unsafe_allow_html=True)
            tax_m = st.radio("Taxonomy input", ["Paste", "Upload"], horizontal=True, key="tax_m")
            if tax_m == "Paste":
                tax_txt = st.text_area(
                    "Taxonomy (one per line)", height=240,
                    placeholder="AI\nMachine Learning\nNeural Network\n...",
                    key="tax_paste",
                )
            else:
                tax_up = st.file_uploader("Upload taxonomy .txt", type=["txt"], key="tax_up")
                tax_txt = tax_up.read().decode("utf-8") if tax_up else ""
                if tax_txt:
                    st.text_area("Preview", tax_txt[:300], height=100, disabled=True)
        with ee2:
            st.markdown("**Document** &nbsp; *(text to analyse)*", unsafe_allow_html=True)
            doc_m = st.radio("Document input", ["Paste", "Upload"], horizontal=True, key="doc_m")
            if doc_m == "Paste":
                doc_txt = st.text_area(
                    "Document text", height=240,
                    placeholder="Paste the document the models should extract entities from...",
                    key="doc_paste",
                )
            else:
                doc_up = st.file_uploader("Upload document .txt", type=["txt"], key="doc_up")
                doc_txt = doc_up.read().decode("utf-8") if doc_up else ""
                if doc_txt:
                    st.text_area("Preview", doc_txt[:300], height=100, disabled=True)
        st.session_state["taxonomy_text"] = tax_txt
        st.session_state["source"]        = doc_txt
        st.session_state["topic"]         = None

    # ---- Validate ----
    errs = []
    if task_type == "summarization" and not (st.session_state.get("source") or "").strip():
        errs.append("Source text is required.")
    if task_type == "generation" and not (st.session_state.get("topic") or "").strip():
        errs.append("Topic / prompt is required.")
    if task_type == "entity_extraction":
        if not st.session_state.get("taxonomy_text", "").strip():
            errs.append("Taxonomy is required.")
        if not (st.session_state.get("source") or "").strip():
            errs.append("Document is required.")
    if model1 == model2:
        errs.append("Model A and Model B must be different.")
    if (
        task_type == "summarization"
        and st.session_state.get("compare_with_llm")
        and not (st.session_state.get("llm_ref_text") or "").strip()
    ):
        errs.append(
            "Reference LLM output is required when "
            "'Compare with reference LLM output' is set to Yes "
            "(paste it or upload a .txt/.json/.docx file)."
        )

    if errs:
        bullet_html = "".join(f"<li>{e}</li>" for e in errs)
        st.markdown(
            f"<div class='notready-banner'><span class='dot'></span>"
            f"<div><b>Configuration incomplete</b>"
            f"<ul style='margin:.3rem 0 0 1.1rem;font-weight:500;opacity:.95'>{bullet_html}</ul></div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.session_state.update(dict(
            cfg_task=task_type,
            cfg_model1=model1, cfg_model2=model2,
            cfg_temp1=temp1, cfg_top_p1=top_p1,
            cfg_temp2=temp2, cfg_top_p2=top_p2,
            cfg_length=length_str,
            cfg_num_bullets=num_bullets,
            cfg_max_tokens=max_tokens,
        ))
        st.markdown(
            "<div class='ready-banner'><span class='dot'></span>"
            "<div><b>Configuration is ready.</b> Switch to the <b>Run Experiment</b> tab to start.</div></div>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB: Run
# ===========================================================================
with t_run:
    if not st.session_state.backend:
        st.info("Connect your OpenAI API key first.")
        st.stop()

    cfg = dict(
        task=st.session_state.get("cfg_task"),
        model1=st.session_state.get("cfg_model1"),
        model2=st.session_state.get("cfg_model2"),
        temp1=st.session_state.get("cfg_temp1", 0.7),
        top_p1=st.session_state.get("cfg_top_p1", 1.0),
        temp2=st.session_state.get("cfg_temp2", 0.7),
        top_p2=st.session_state.get("cfg_top_p2", 1.0),
        length=st.session_state.get("cfg_length", "150-200 words"),
        num_bullets=st.session_state.get("cfg_num_bullets"),
        max_tokens=st.session_state.get("cfg_max_tokens", 1500),
        source=st.session_state.get("source"),
        topic=st.session_state.get("topic"),
        taxonomy=st.session_state.get("taxonomy_text", ""),
        compare_with_llm=bool(st.session_state.get("compare_with_llm")),
        llm_compare_mode=st.session_state.get("llm_compare_mode", "Both selected models"),
        llm_selected_model_role=st.session_state.get("llm_selected_model_role", "Model A"),
        llm_compare_depth=st.session_state.get("llm_compare_depth", "Fast"),
        llm_ref_text=st.session_state.get("llm_ref_text", "") or "",
        llm_ref_source=st.session_state.get("llm_ref_source", "") or "",
        llm_judge_enabled=bool(st.session_state.get("llm_judge_enabled", True)),
    )

    if not cfg["task"] or not cfg["model1"] or not cfg["model2"]:
        st.info("Complete the **Configure** tab before running.")
        st.stop()

    # ---- Summary ----
    section_header("Experiment Summary", "Confirm everything looks right before launching.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task",       cfg["task"].title())
    c2.metric("Model A",    cfg["model1"])
    c3.metric("Model B",    cfg["model2"])
    c4.metric("Max Tokens", cfg["max_tokens"])
    c5, c6, c7 = st.columns(3)
    c5.metric("Temp / Top-P · A", f"{cfg['temp1']} / {cfg['top_p1']}")
    c6.metric("Temp / Top-P · B", f"{cfg['temp2']} / {cfg['top_p2']}")
    c7.metric("Length",            cfg["length"])

    if cfg["task"] == "summarization" and cfg["compare_with_llm"]:
        ref_words = len((cfg["llm_ref_text"] or "").split())
        src_lbl = cfg["llm_ref_source"] or "—"
        judge_lbl = "on" if cfg["llm_judge_enabled"] else "off"
        depth_lbl = cfg["llm_compare_depth"]
        if cfg["llm_compare_mode"] == "Only one selected model":
            selected_label = (
                cfg["model1"]
                if cfg["llm_selected_model_role"] == "Model A"
                else cfg["model2"]
            )
            compare_scope = f"Only {cfg['llm_selected_model_role']} ({selected_label})"
        else:
            compare_scope = f"Both selected models ({cfg['model1']} and {cfg['model2']})"
        st.markdown(
            f"<div class='ready-banner'><span class='dot'></span><div>"
            f"<b>Reference LLM output active</b> — {ref_words} words "
            f"(source: {src_lbl}). Compare scope: <b>{compare_scope}</b>. "
            f"Model output will be generated at run time. "
            f"Depth: <b>{depth_lbl}</b>. LLM-as-judge: <b>{judge_lbl}</b>."
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ---- Launch ----
    launch_helper = "Both models run in parallel — comparison is fully automated."
    if cfg["task"] == "summarization" and cfg["compare_with_llm"] and cfg["llm_compare_mode"] == "Only one selected model":
        launch_helper = "Only the selected model is run; its runtime output is compared with the reference LLM output."
    section_header("Launch", launch_helper)
    run_btn = st.button(
        "Run A/B Experiment",
        disabled=st.session_state.running,
        type="primary",
        use_container_width=True,
    )
    prog       = st.progress(0)
    status_box = st.empty()
    log_box    = st.empty()

    def run_experiment():
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        import threading

        be = st.session_state.backend
        st.session_state.running        = True
        st.session_state.results        = []
        st.session_state.log_lines      = []
        st.session_state.dashboard_html = None

        # Thread-safe log list — workers append here; the main thread flushes to UI
        _thread_logs = []
        _log_lock    = threading.Lock()

        def _tlog(msg):
            """Thread-safe log — never touches Streamlit from a worker thread."""
            with _log_lock:
                _thread_logs.append(msg)

        def _flush_logs():
            """Call only from the main Streamlit thread to push logs to UI."""
            with _log_lock:
                st.session_state.log_lines = list(_thread_logs)
            log_box.code("\n".join(st.session_state.log_lines), language="text")

        try:
            task   = cfg["task"]
            m1, m2 = cfg["model1"], cfg["model2"]
            t1, p1 = float(cfg["temp1"]), float(cfg["top_p1"])
            t2, p2 = float(cfg["temp2"]), float(cfg["top_p2"])
            ln     = cfg["length"]
            nb     = cfg["num_bullets"]
            mt     = int(cfg["max_tokens"])
            src    = cfg["source"]   or ""
            top    = cfg["topic"]    or ""
            tax    = cfg["taxonomy"] or ""

            if task == "entity_extraction":
                be._ENTITY_TAXONOMY_TEXT = tax
                be._parse_taxonomy_from_text(tax)

            ref = tax if task == "entity_extraction" else (src if src else top)
            tv  = dict(length=ln, source=src, topic=top)
            if task == "entity_extraction":
                tv["taxonomy"] = tax
            prompt = be.generate_task_prompt(task, **tv)
            _tlog(f"Prompt ready ({len(prompt)} chars)")
            prog.progress(10)
            _flush_logs()

            def _call_and_eval(model, temp, top_p):
                """Runs entirely in a worker thread — NO Streamlit calls allowed here."""
                _tlog(f"Calling {model} (temp={temp}, top_p={top_p})...")
                output, lat, toks, ptoks, ctoks = be.call_openai(
                    model, prompt, max_tokens=mt, temperature=temp, top_p=top_p, num_bullets=nb
                )
                if not output or not output.strip():
                    _tlog(f"ERROR: {model} returned empty output")
                    return None
                _tlog(f"{model} -> {lat:.1f}s, {toks} tokens")
                ent_det = None
                if task == "entity_extraction":
                    rec, prec, f1, det = be._evaluate_entity_extraction_core(output, tax)
                    ent_det = det
                    _tlog(f"  TP={det['tp']} FP={det['fp']} FN={det['fn']} F1={f1:.3f}")
                _tlog(f"Evaluating {model}...")
                try:
                    acc, rel, eff = be.evaluate_quality_improved(output, ref, task)
                except Exception as ex:
                    _tlog(f"Eval warning: {ex}")
                    acc, rel, eff = 2.5, 2.5, 2.5
                cost = be.estimate_cost(model, ptoks or 0, ctoks or 0)
                _tlog(f"  Acc={acc:.2f} Rel={rel:.2f} Eff={eff:.2f} Cost=${cost:.4f}")
                entry = dict(
                    Task=task, Model=model, Output=output,
                    Latency_s=round(lat,2) if lat else None,
                    Token_Usage=toks, Prompt_Tokens=ptoks, Completion_Tokens=ctoks,
                    Cost_USD=cost, Accuracy=acc, Relevance=rel, Effectiveness=eff,
                    Temperature=temp, Top_P=top_p,
                    Timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                )
                if ent_det:
                    entry.update(dict(
                        Entity_Recall=ent_det.get("recall"),
                        Entity_Precision=ent_det.get("precision"),
                        Entity_F1=ent_det.get("f1"),
                        Entity_Gold_Count=ent_det.get("gold_count"),
                        Entity_Pred_Count=ent_det.get("pred_count"),
                        Entity_TP=ent_det.get("tp"), Entity_FP=ent_det.get("fp"), Entity_FN=ent_det.get("fn"),
                        Extracted_Entities_List=ent_det.get("predicted_entities"),
                        Matching_Entities=ent_det.get("true_positive_entities"),
                        Missing_Entities=ent_det.get("missing_entities"),
                        Extra_Entities=ent_det.get("extra_entities"),
                    ))
                if task == "summarization":
                    try:
                        entry["Eval_Details"] = be.get_summarization_evaluation_details(output, ref)
                        ed = entry["Eval_Details"]
                        if ed.get("deepeval_enabled"):
                            _tlog(
                                "  DeepEval | metric.score=%.3f alignment=%.3f "
                                "coverage=%.3f (raw=%.3f, robust=%.3f, src=%s, bucket=%s)"
                                % (
                                    ed.get("deepeval_score") or 0.0,
                                    ed.get("alignment_score") or 0.0,
                                    ed.get("coverage_score") or 0.0,
                                    ed.get("deepeval_raw_coverage") or 0.0,
                                    ed.get("robust_coverage") or 0.0,
                                    ed.get("coverage_source") or "n/a",
                                    ed.get("summary_length_bucket") or "n/a",
                                )
                            )
                    except Exception as ex:
                        _tlog(f"Eval_Details warning: {ex}")
                return entry

            if (
                task == "summarization"
                and cfg["compare_with_llm"]
                and cfg["llm_compare_mode"] == "Only one selected model"
            ):
                if cfg["llm_selected_model_role"] == "Model B":
                    run_specs = [(m2, t2, p2)]
                else:
                    run_specs = [(m1, t1, p1)]
            else:
                run_specs = [(m1, t1, p1), (m2, t2, p2)]

            if len(run_specs) == 1:
                _tlog(
                    f"Running selected model only: {run_specs[0][0]} "
                    "(runtime output will be compared with reference LLM output)..."
                )
            else:
                _tlog("Running both selected models in parallel...")
            prog.progress(20)
            _flush_logs()

            results     = []
            model_order = [m for m, _, _ in run_specs]
            with ThreadPoolExecutor(max_workers=len(run_specs)) as _ex:
                _futs = {_ex.submit(_call_and_eval, m, t, p): m for m, t, p in run_specs}
                for _f in _as_completed(_futs):
                    _e = _f.result()   # re-raises any worker exception in main thread
                    if _e is not None:
                        results.append(_e)

            _flush_logs()
            results.sort(key=lambda r: model_order.index(r["Model"]) if r["Model"] in model_order else 99)

            # ---- Optional: compare each model's output against the user's
            # reference LLM output (summarization only).
            # PERF: Both per-model comparisons run in parallel. Internally
            # `compare_two_summaries` ALSO parallelizes its 5 LLM-bound
            # stages, so one comparison takes wall-time ~= max(slowest
            # stage) instead of sum(stages). Net speedup vs the original
            # sequential implementation: ~5-7x.
            if (
                task == "summarization"
                and cfg["compare_with_llm"]
                and (cfg["llm_ref_text"] or "").strip()
                and results
            ):
                _tlog("")
                _tlog(
                    f"Comparing {len(results)} model output(s) vs the reference "
                    "LLM output (running in parallel)..."
                )
                _flush_logs()
                ref_text = cfg["llm_ref_text"]
                ref_label = f"Reference ({cfg['llm_ref_source'] or 'pasted'})"

                def _compare_one(entry):
                    try:
                        return entry, be.compare_two_summaries(
                            summary_a=entry.get("Output", ""),
                            summary_b=ref_text,
                            source_text=src or None,
                            label_a=entry["Model"],
                            label_b=ref_label,
                            run_llm_judge=(
                                cfg["llm_judge_enabled"]
                                and cfg["llm_compare_depth"] == "Full audit"
                            ),
                            run_deepeval_pair=cfg["llm_compare_depth"] == "Full audit",
                            run_source_score=cfg["llm_compare_depth"] == "Full audit",
                        ), None
                    except Exception as cmp_err:
                        return entry, None, cmp_err

                with ThreadPoolExecutor(max_workers=max(2, len(results))) as _cex:
                    cmp_futs = [_cex.submit(_compare_one, e) for e in results]
                    for _cf in _as_completed(cmp_futs):
                        entry, cmp_dict, cmp_err = _cf.result()
                        if cmp_err is not None:
                            _tlog(f"  {entry['Model']} vs Reference FAILED: {cmp_err}")
                            continue
                        cmp_dict["model_output_source"] = "runtime_generated_output"
                        cmp_dict["comparison_mode"] = (
                            "single_model_vs_llm"
                            if cfg["llm_compare_mode"] == "Only one selected model"
                            else "both_models_vs_llm"
                        )
                        cmp_dict["comparison_depth"] = cfg["llm_compare_depth"]
                        entry["LLM_Comparison"] = cmp_dict
                        entry["Reference_Comparison"] = cmp_dict
                        overall = cmp_dict.get("overall_score_100")
                        wc = cmp_dict.get("winner_card") or {}
                        win = wc.get("winner") or cmp_dict.get("winner", "Tie")
                        confidence = wc.get("confidence") or "—"
                        agg = cmp_dict.get("agreement_score_0_1")
                        lbl = cmp_dict.get("agreement_label", "n/a")
                        _tlog(
                            f"  {entry['Model']} vs Reference: "
                            f"overall={overall if overall is not None else 'n/a'}/100  "
                            f"winner={win} (confidence: {confidence})  "
                            f"[legacy: agreement={agg if agg is not None else 'n/a'}, "
                            f"label={lbl}]"
                        )
                        for w in cmp_dict.get("error_warnings") or []:
                            _tlog(f"    warn: {w}")
                _flush_logs()

            st.session_state.results = results
            prog.progress(90)

            if results:
                ts = results[0]["Timestamp"]
                comparison_config = None
                if task == "summarization" and cfg["compare_with_llm"]:
                    selected_models = model_order
                    comparison_config = {
                        "mode": (
                            "single_model_vs_llm"
                            if cfg["llm_compare_mode"] == "Only one selected model"
                            else "both_models_vs_llm"
                        ),
                        "label": cfg["llm_compare_mode"],
                        "selected_models": selected_models,
                        "model_output_source": "runtime_generated_output",
                        "uses_reference_llm": True,
                        "comparison_depth": cfg["llm_compare_depth"],
                    }
                reference_meta = None
                if task == "summarization" and cfg["compare_with_llm"] and cfg["llm_ref_text"]:
                    # Cap stored text so very large references don't bloat the
                    # JSON export, but keep enough to render meaningfully in
                    # both the Results tab and the standalone HTML dashboard.
                    _ref_full = cfg["llm_ref_text"] or ""
                    _REF_TEXT_LIMIT = 20000
                    reference_meta = {
                        "source": cfg["llm_ref_source"] or "pasted",
                        "word_count": len(_ref_full.split()),
                        "char_count": len(_ref_full),
                        "parser_note": st.session_state.get("llm_ref_parse_note", ""),
                        "comparison_mode": (
                            "single_model_vs_llm"
                            if cfg["llm_compare_mode"] == "Only one selected model"
                            else "both_models_vs_llm"
                        ),
                        "text": _ref_full[:_REF_TEXT_LIMIT],
                        "text_truncated": len(_ref_full) > _REF_TEXT_LIMIT,
                    }
                    if hasattr(be, "_short_text_hash"):
                        reference_meta["sha256_16"] = be._short_text_hash(cfg["llm_ref_text"])

                jp = be.save_results_to_json(
                    results,
                    ts,
                    extra_blocks={
                        "Reference_Meta": reference_meta,
                        "Comparison_Config": comparison_config,
                    },
                )
                st.session_state.last_json_path = jp
                if jp:
                    hp = be.generate_dashboard(jp)
                    if hp and os.path.exists(hp):
                        with open(hp, encoding="utf-8") as f:
                            st.session_state.dashboard_html = f.read()
                        _tlog(f"Dashboard: {hp}")

            prog.progress(100)
            _tlog("Done!")
            _flush_logs()

        except Exception as ex:
            import traceback
            _tlog(f"ERROR: {ex}\n{traceback.format_exc()}")
            _flush_logs()
        finally:
            st.session_state.running = False

    if run_btn:
        run_experiment()
        st.rerun()

    if st.session_state.log_lines:
        log_box.code("\n".join(st.session_state.log_lines), language="text")
    if not st.session_state.running and st.session_state.results:
        status_box.success("Experiment complete — head over to the **Results** tab!")

# ===========================================================================
# TAB: Results
# ===========================================================================
with t_res:
    res = st.session_state.results
    if not res:
        st.info("Run an experiment first to see results here.")
        st.stop()

    # ---- Winner banner ----
    if len(res) >= 2:
        win = max(res, key=lambda r: r.get("Effectiveness", 0))
        los = [r for r in res if r["Model"] != win["Model"]][0]
        mg  = win["Effectiveness"] - los["Effectiveness"]
        st.markdown(
            f"<div class='winner-box'>"
            f"<span class='crown'>Winner</span>"
            f"<span style='font-size:1.25rem'>{win['Model']}</span>"
            f"<span style='opacity:.8;margin:0 .55rem'>·</span>"
            f"<span>Effectiveness {win['Effectiveness']:.2f}/5</span>"
            f"<span class='margin'>+{mg:.2f} over {los['Model']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ---- Per-model side-by-side ----
    section_header("Side-by-Side Comparison", "Each card shows scores, costs, and the full output for one model.")
    cols = st.columns(len(res))
    for col, r in zip(cols, res):
        with col:
            is_w  = len(res) >= 2 and r["Model"] == max(res, key=lambda x: x.get("Effectiveness", 0))["Model"]
            hc    = "model-a" if res.index(r) == 0 else "model-b"
            badge_pill = "<span class='pill' style='background:rgba(34,197,94,.18);color:#86efac'>Winner</span>" if is_w else ""
            st.markdown(
                f'<div class="model-header {hc}">{r["Model"]} {badge_pill}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                badge("Accuracy",      r.get("Accuracy")) +
                badge("Relevancy",     r.get("Relevance")) +
                badge("Effectiveness", r.get("Effectiveness")),
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Latency", f"{r.get('Latency_s','N/A')}s")
            m2.metric("Tokens",  r.get("Token_Usage", "N/A"))
            m3.metric("Cost",    f"${r.get('Cost_USD',0):.4f}")
            mc1, mc2 = st.columns(2)
            mc1.metric("Temp",  r.get("Temperature","N/A"))
            mc2.metric("Top-P", r.get("Top_P","N/A"))

            if r.get("Entity_F1") is not None:
                st.markdown("**Entity Metrics**")
                a, b2, c = st.columns(3)
                a.metric("Recall",     f"{r.get('Entity_Recall',0):.3f}")
                b2.metric("Precision", f"{r.get('Entity_Precision',0):.3f}")
                c.metric("F1",         f"{r.get('Entity_F1',0):.3f}")
                st.markdown("**Counts**")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Gold",  r.get("Entity_Gold_Count",0))
                e2.metric("Pred",  r.get("Entity_Pred_Count",0))
                e3.metric("TP",    r.get("Entity_TP",0))
                e4.metric("FP/FN", f"{r.get('Entity_FP',0)}/{r.get('Entity_FN',0)}")
                with st.expander("Matched entities"):
                    st.write(", ".join(str(e) for e in (r.get("Matching_Entities") or [])) or "None")
                with st.expander("Missing (false negatives)"):
                    st.write(", ".join(str(e) for e in (r.get("Missing_Entities") or [])) or "None")
                with st.expander("Extra (false positives)"):
                    st.write(", ".join(str(e) for e in (r.get("Extra_Entities") or [])) or "None")

            with st.expander("Model Output", expanded=True):
                st.markdown(r.get("Output",""))

    # ---- Optional: per-model comparison vs the user's reference LLM output ----
    has_ref_cmp = any(r.get("LLM_Comparison") for r in res)
    if has_ref_cmp:
        section_header(
            "Comparison vs Reference LLM Output",
            "Each model scored against your reference on the four signals that "
            "matter for LLM A/B testing: LLM-as-Judge quality, factual "
            "consistency, semantic similarity, and length sanity — folded into "
            "a single Overall Score (0-100) and a Winner verdict.",
        )

        # ---- Reference LLM output preview: show what's being compared against ----
        _ref_text = st.session_state.get("llm_ref_text", "") or ""
        if _ref_text.strip():
            _ref_words = len(_ref_text.split())
            _ref_chars = len(_ref_text)
            _ref_source = st.session_state.get("llm_ref_source") or "pasted"
            _ref_filename = st.session_state.get("llm_ref_filename") or ""
            _meta_chips = (
                f"<span class='score-badge score-ok'>Source: {_ref_source}</span>"
                f"{f'<span class=\"score-badge score-ok\">File: {_ref_filename}</span>' if _ref_filename else ''}"
                f"<span class='score-badge score-ok'>{_ref_words} words</span>"
                f"<span class='score-badge score-ok'>{_ref_chars:,} chars</span>"
            )
            st.markdown(
                f"""
<div style="border:1px solid rgba(245,158,11,.35); border-radius:12px;
            padding:14px 16px; margin:6px 0 14px 0;
            background:linear-gradient(180deg, rgba(245,158,11,.08), rgba(245,158,11,.02));">
  <div style="display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin-bottom:.55rem;">
    <span style="font-weight:700; opacity:.95;">📎 Reference LLM Output (used for comparison)</span>
    {_meta_chips}
  </div>
  <div style="font-size:.82rem; opacity:.75; margin-bottom:.4rem;">
    Every model's runtime output below is scored against this text.
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("View reference text", expanded=False):
                st.text_area(
                    "Reference text",
                    _ref_text,
                    height=260,
                    disabled=True,
                    label_visibility="collapsed",
                )

        # ---- Metric guide (plain English): pull from compute layer's
        # plain_english dict so the wording is identical in Results + Dashboard ----
        # We grab the dict from the first available comparison; it's the same
        # for every model because it describes the metric stack, not the data.
        _first_cmp = next(
            (r["LLM_Comparison"] for r in res if r.get("LLM_Comparison")),
            None,
        )
        _plain_english = (
            ((_first_cmp or {}).get("metric_details") or {}).get("plain_english")
            or {}
        )

        with st.expander(
            "How to read these metrics · plain-English guide",
            expanded=True,
        ):
            st.markdown(
                "Each metric below answers a different question about the "
                "model's output. **Read this once** — it explains every number "
                "you'll see in the cards below."
            )
            metric_order = [
                "overall_score", "winner_and_confidence", "llm_judge",
                "factual_consistency", "semantic_similarity",
                "faithfulness_source", "length_ratio", "rouge_l",
            ]
            for key in metric_order:
                m = _plain_english.get(key)
                if not m:
                    continue
                st.markdown(
                    f"""
<div style="border-left:3px solid rgba(99,102,241,.55);
            padding:8px 12px; margin:8px 0;
            background:rgba(99,102,241,.04); border-radius:6px;">
  <div style="font-weight:700; margin-bottom:.25rem;">{m['label']}</div>
  <div style="font-size:.88rem; line-height:1.55; opacity:.92;">
    <b>What it measures:</b> {m['what']}<br>
    <b>How it's computed:</b> {m['how']}<br>
    <b>What a good score looks like:</b> {m['good']}<br>
    <b>What it catches:</b> {m['catches']}
  </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

        def _fmt_num(v, places=3):
            if v is None:
                return "—"
            try:
                return f"{float(v):.{places}f}"
            except (TypeError, ValueError):
                return str(v)

        def _badge_class_for_0_1(v, good=0.75, ok=0.5):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return "score-ok"
            if f >= good:
                return "score-good"
            if f >= ok:
                return "score-ok"
            return "score-poor"

        def _badge_class_for_length_ratio(ratio):
            try:
                r = float(ratio)
            except (TypeError, ValueError):
                return "score-ok"
            if 0.7 <= r <= 1.4:
                return "score-good"
            if 0.5 <= r <= 2.0:
                return "score-ok"
            return "score-poor"

        def _winner_pill(confidence):
            if confidence == "Tie":
                return "score-ok"
            if confidence == "High":
                return "score-good"
            if confidence == "Medium":
                return "score-good"
            return "score-ok"

        cmp_models = [r for r in res if r.get("LLM_Comparison")]
        cmp_cols = st.columns(len(cmp_models))
        for col, r in zip(cmp_cols, cmp_models):
            cmp = r["LLM_Comparison"]
            with col:
                hc = "model-a" if res.index(r) == 0 else "model-b"

                wc = cmp.get("winner_card") or {}
                overall = cmp.get("overall_score_100")
                winner = wc.get("winner") or cmp.get("winner") or "Tie"
                confidence = wc.get("confidence") or "Low"
                rationale = wc.get("rationale") or ""

                # Header pill shows model name + vs-reference
                st.markdown(
                    f'<div class="model-header {hc}">{r["Model"]} '
                    f'<span class="pill">vs Reference</span></div>',
                    unsafe_allow_html=True,
                )

                # ---- Tier 1: Winner Card (headline verdict) -------------
                if overall is not None:
                    pill_cls = _winner_pill(confidence)
                    score_cls = _badge_class_for_0_1(overall / 100.0, good=0.7, ok=0.5)
                    delta = wc.get("delta_100")
                    delta_txt = (
                        f"Δ {delta:+.1f}" if isinstance(delta, (int, float)) else ""
                    )
                    st.markdown(
                        f"""
<div style="border:1px solid rgba(255,255,255,.08); border-radius:14px;
            padding:14px 16px; margin:6px 0 10px 0;
            background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0));">
  <div style="display:flex; align-items:center; gap:.5rem; flex-wrap:wrap;">
    <span class="score-badge {score_cls}" style="font-size:1.05rem;">
      Overall Score · {overall:.1f}/100
    </span>
    <span class="score-badge {pill_cls}">Winner: {winner}</span>
    <span class="score-badge score-ok">Confidence: {confidence}</span>
    {f'<span class="score-badge score-ok">{delta_txt}</span>' if delta_txt else ''}
  </div>
  <div style="margin-top:.5rem; opacity:.85; font-size:.92rem;">{rationale}</div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Overall Score unavailable — judge or NLI did not run.")

                # ---- Tier 2: LLM-as-Judge headline metric ---------------
                judge = cmp.get("llm_judge")
                judge_0_1 = cmp.get("judge_score_0_1")
                top1, top2 = st.columns(2)
                if judge_0_1 is not None:
                    top1.metric(
                        "LLM-as-Judge",
                        f"{judge_0_1 * 5:.2f} / 5",
                        help="Mean of 5 rubric dimensions (faithfulness, coverage, "
                             "conciseness, coherence, style), position-bias controlled "
                             "via A/B swap. The single highest-signal metric.",
                    )
                else:
                    top1.metric("LLM-as-Judge", "—",
                                help="Judge not run or returned an error.")
                top2.metric(
                    "Factual Consistency",
                    _fmt_num(cmp.get("factual_score_0_1")),
                    help="Bidirectional NLI mutual entailment with a contradiction "
                         "penalty (each contradicting pair multiplies the score "
                         "by 0.7). 0 = unrelated/contradictory, 1 = fully entailed.",
                )

                # ---- Tier 2 (cont.): supporting signals ----------------
                emb = cmp.get("embedding") or {}
                ivs = cmp.get("independent_vs_source") or {}
                sa = (ivs.get("a") or {}) if isinstance(ivs, dict) else {}
                source_score = sa.get("alignment_0_1") or sa.get("deepeval_score_0_1")
                sup1, sup2 = st.columns(2)
                sup1.metric(
                    "Semantic Similarity",
                    _fmt_num(emb.get("cosine_0_1")),
                    help="cos(embed(model), embed(reference)), clamped to [0,1]. "
                         "Captures meaning-level agreement; replaces BERTScore "
                         "and ROUGE for the headline view.",
                )
                if source_score is not None:
                    sup2.metric(
                        "Faithfulness · Source",
                        _fmt_num(source_score),
                        help="How well the model's output preserves facts from "
                             "the original source document (when provided). "
                             "Catches hallucinations the reference comparison misses.",
                    )
                else:
                    sup2.metric("Faithfulness · Source", "—",
                                help="No source document provided for this run.")

                # ---- LLM-as-Judge rationale + hallucinations ------------
                if judge and "error" not in judge:
                    rationale_text = (judge.get("overall_rationale") or "").strip()
                    halls = judge.get("hallucinations") or {}
                    miss = judge.get("missing_keypoints") or {}
                    if rationale_text:
                        st.markdown(
                            f"<div style='font-size:.88rem; opacity:.8; "
                            f"margin:.25rem 0 .5rem 0;'>"
                            f"<b>Judge says:</b> {rationale_text}</div>",
                            unsafe_allow_html=True,
                        )

                    with st.expander("Judge breakdown · per-dimension scores", expanded=False):
                        rows = []
                        for dim, vals in (judge.get("per_dimension") or {}).items():
                            rows.append({
                                "Dimension": dim,
                                "Model (A)": vals.get("a"),
                                "Reference (B)": vals.get("b"),
                                "Winner": vals.get("winner"),
                            })
                        if rows:
                            st.dataframe(
                                pd.DataFrame(rows),
                                use_container_width=True,
                                hide_index=True,
                            )
                        st.caption(
                            f"Judge model: `{judge.get('_judge_model','?')}` · "
                            f"position-bias controlled: "
                            f"{judge.get('_position_bias_controlled', False)}"
                        )
                        if halls.get("a") or miss.get("a"):
                            hc1, hc2 = st.columns(2)
                            with hc1:
                                st.markdown("**Hallucinations · Model**")
                                st.write(halls.get("a") or "None")
                            with hc2:
                                st.markdown("**Missing key points · Model**")
                                st.write(miss.get("a") or "None")
                elif judge and "error" in judge:
                    st.caption(f"LLM-judge unavailable: {judge['error']}")

                # ---- Tier 3: collapsed diagnostics ----------------------
                length = cmp.get("length") or {}
                lex = cmp.get("lexical") or {}
                ratio = length.get("length_ratio")
                ratio_badge = _badge_class_for_length_ratio(ratio)
                ratio_txt = _fmt_num(ratio)
                with st.expander(
                    "Diagnostics · length sanity, ROUGE (lexical baseline)",
                    expanded=False,
                ):
                    d1, d2 = st.columns(2)
                    d1.markdown(
                        f"**Length ratio**<br>"
                        f"<span class='score-badge {ratio_badge}'>"
                        f"{ratio_txt}</span><br>"
                        f"<span style='font-size:.78rem; opacity:.7;'>"
                        f"&lt;0.5 truncated · 0.7–1.4 healthy · &gt;2.0 verbose"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                    d2.metric(
                        "ROUGE-L (lexical)",
                        _fmt_num(lex.get("rouge_0_1")),
                        help="Surface-level word overlap. Kept as a familiar "
                             "baseline for NLP stakeholders; not used in the "
                             "Overall Score because semantic similarity "
                             "subsumes it for modern LLM outputs.",
                    )
                    st.caption(
                        f"Model words: {length.get('words_a','—')} · "
                        f"Reference words: {length.get('words_b','—')}"
                    )

                metric_details = cmp.get("metric_details") or {}

                overall_components = cmp.get("overall_components") or {}
                overall_weights = cmp.get("overall_weights") or {}
                overall_score = cmp.get("overall_score_100")
                if metric_details or overall_components:
                    with st.expander(
                        "How the Overall Score was built · step-by-step with this run's numbers",
                        expanded=False,
                    ):
                        st.markdown(
                            "Below is the **exact math** that produced this "
                            "model's Overall Score. Every number you see here "
                            "comes from this run."
                        )

                        if overall_components:
                            # Step 1 — show each signal in plain English
                            signal_friendly = {
                                "judge":         "LLM-as-Judge quality",
                                "factual":       "Factual Consistency",
                                "semantic":      "Semantic Similarity",
                                "length_sanity": "Length Sanity",
                            }
                            st.markdown("**Step 1 — Score each signal (0–1)**")
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {
                                            "Signal": signal_friendly.get(k, k),
                                            "Score (0–1)": round(v, 3),
                                            "Meaning": (
                                                "Strong" if v >= 0.75
                                                else "Acceptable" if v >= 0.5
                                                else "Weak"
                                            ),
                                        }
                                        for k, v in overall_components.items()
                                    ]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # Step 2 — multiply each by its weight
                            st.markdown(
                                "**Step 2 — Multiply each score by its weight, "
                                "then add up the contributions**"
                            )
                            sum_contrib = 0.0
                            sum_weight = 0.0
                            rows = []
                            for k, v in overall_components.items():
                                w = overall_weights.get(k) or 0
                                contrib = round(w * v, 4)
                                sum_contrib += contrib
                                sum_weight += w
                                rows.append({
                                    "Signal": signal_friendly.get(k, k),
                                    "Score": round(v, 3),
                                    "× Weight": round(w, 3),
                                    "= Contribution": contrib,
                                })
                            rows.append({
                                "Signal": "TOTAL",
                                "Score": "",
                                "× Weight": round(sum_weight, 3),
                                "= Contribution": round(sum_contrib, 4),
                            })
                            st.dataframe(
                                pd.DataFrame(rows),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # Step 3 — divide and scale to 0-100
                            avg = sum_contrib / sum_weight if sum_weight else 0
                            st.markdown(
                                f"**Step 3 — Divide by the total weight and "
                                f"multiply by 100**"
                            )
                            st.code(
                                f"Overall Score "
                                f"= {sum_contrib:.4f} ÷ {sum_weight:.3f} × 100 "
                                f"= {avg:.4f} × 100 "
                                f"= {overall_score if overall_score is not None else round(avg * 100, 1)}",
                                language="text",
                            )

                            st.caption(
                                "Why divide by total weight? If a signal is "
                                "missing this run (e.g. the judge was skipped), "
                                "we don't want the score to drop just because "
                                "fewer signals contributed. Dividing renormalizes "
                                "so the score always lives on a 0–100 scale."
                            )

                            # Step 4 — show the default weights for reference
                            st.markdown("**Default weights (used when every signal runs)**")
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {"Signal": "LLM-as-Judge quality", "Weight": "45%"},
                                        {"Signal": "Factual Consistency",  "Weight": "30%"},
                                        {"Signal": "Semantic Similarity",  "Weight": "20%"},
                                        {"Signal": "Length Sanity",        "Weight": "5%"},
                                    ]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.caption(
                                "Why these weights? LLM-as-Judge correlates "
                                "best with human preference, so it carries the "
                                "most weight. Factual Consistency is a hard "
                                "guardrail against hallucinations. Semantic "
                                "Similarity catches meaning-level agreement. "
                                "Length Sanity is a small tie-breaker."
                            )

                        skipped = metric_details.get("skipped_signals") or []
                        if skipped:
                            st.info(
                                "**Skipped in this run:** "
                                + ", ".join(skipped)
                                + ". Weights were renormalized over the remaining "
                                "signals so the score still lives on a 0–100 scale."
                            )

                        # Engineer-facing details, hidden by default so they
                        # don't clutter the friendly view above.
                        with st.expander(
                            "Advanced · raw formulas and legacy agreement view",
                            expanded=False,
                        ):
                            formulas = metric_details.get("formulas") or {}
                            if formulas:
                                st.markdown("**Per-signal formulas (technical)**")
                                for name, formula in formulas.items():
                                    st.markdown(f"- **{name}:** `{formula}`")

                            legacy_used = (
                                metric_details.get("used_components")
                                or cmp.get("agreement_components")
                                or {}
                            )
                            if legacy_used:
                                st.markdown(
                                    "**Legacy similarity-agreement view** "
                                    "(kept for backward compatibility with older "
                                    "exports — the Overall Score above is the "
                                    "current headline)."
                                )
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "Component": k,
                                                "Score": v,
                                                "Weight": (metric_details.get("agreement_weights") or {}).get(k),
                                            }
                                            for k, v in legacy_used.items()
                                        ]
                                    ),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                applied = metric_details.get("agreement_formula_applied")
                                if applied:
                                    st.code(applied, language="text")

                        for note in metric_details.get("notes") or []:
                            st.caption(note)

                if cmp.get("error_warnings"):
                    with st.expander("Warnings", expanded=False):
                        for w in cmp["error_warnings"]:
                            st.write("·", w)

        # ---- Final Comparison Summary: cross-model winner with reasoning ----
        # Picks the model with the highest Overall Score, then explains the
        # gap by naming the signal where the lead is largest. Falls back to
        # the per-row winner_card if Overall Score is unavailable for any side.
        scored = []
        for r in cmp_models:
            cmpx = r.get("LLM_Comparison") or {}
            scored.append({
                "model":       r.get("Model", "?"),
                "overall":     cmpx.get("overall_score_100"),
                "judge":       cmpx.get("judge_score_0_1"),
                "factual":     cmpx.get("factual_score_0_1"),
                "semantic":    (cmpx.get("embedding") or {}).get("cosine_0_1"),
                "length":      cmpx.get("length_sanity_0_1"),
                "wc":          cmpx.get("winner_card") or {},
                "halls":       ((cmpx.get("llm_judge") or {}).get("hallucinations") or {}).get("a") or [],
                "missing":     ((cmpx.get("llm_judge") or {}).get("missing_keypoints") or {}).get("a") or [],
            })

        # Determine the cross-model winner
        ranked = [s for s in scored if isinstance(s["overall"], (int, float))]
        ranked.sort(key=lambda s: s["overall"], reverse=True)

        if not ranked:
            summary_winner = "Tie"
            summary_confidence = "—"
            summary_reason = ("Overall Score could not be computed for any "
                              "model — judge and NLI signals were missing.")
            summary_delta = 0.0
        elif len(ranked) == 1:
            sole = ranked[0]
            # Single-model vs reference: defer to the row-level winner_card
            summary_winner = sole["wc"].get("winner") or sole["model"]
            summary_confidence = sole["wc"].get("confidence") or "—"
            summary_reason = sole["wc"].get("rationale") or (
                f"{sole['model']} scored {sole['overall']:.1f}/100 against the reference."
            )
            summary_delta = sole["wc"].get("delta_100") or 0.0
        else:
            top, runner = ranked[0], ranked[1]
            summary_delta = round(top["overall"] - runner["overall"], 1)
            summary_confidence = _confidence_label_for_delta = (
                "Tie"    if abs(summary_delta) < 3 else
                "Low"    if abs(summary_delta) < 8 else
                "Medium" if abs(summary_delta) < 15 else "High"
            )
            if summary_confidence == "Tie":
                summary_winner = "Tie"
            else:
                summary_winner = top["model"]

            # Find the signal where the gap is largest among the top two
            signal_labels = {
                "judge":    "LLM-as-Judge quality",
                "factual":  "factual consistency",
                "semantic": "semantic similarity",
                "length":   "length sanity",
            }
            gaps = {}
            for k in signal_labels:
                tv, rv = top.get(k), runner.get(k)
                if isinstance(tv, (int, float)) and isinstance(rv, (int, float)):
                    gaps[k] = tv - rv
            if gaps:
                lead_signal = max(gaps, key=lambda k: abs(gaps[k]))
                lead_sign = "ahead on" if gaps[lead_signal] >= 0 else "behind on"
                if summary_winner == "Tie":
                    summary_reason = (
                        f"{top['model']} and {runner['model']} are within "
                        f"{abs(summary_delta):.1f} points "
                        f"({top['overall']:.1f} vs {runner['overall']:.1f}). "
                        f"Largest single-signal gap: "
                        f"{signal_labels[lead_signal]} ({gaps[lead_signal]:+.2f})."
                    )
                else:
                    summary_reason = (
                        f"{top['model']} leads {runner['model']} by "
                        f"{abs(summary_delta):.1f} points "
                        f"({top['overall']:.1f} vs {runner['overall']:.1f}), "
                        f"driven primarily by {signal_labels[lead_signal]} "
                        f"(Δ {gaps[lead_signal]:+.2f}, "
                        f"{top['model']} {lead_sign} this dimension)."
                    )
            else:
                summary_reason = (
                    f"{top['model']} scored {top['overall']:.1f}/100 vs "
                    f"{runner['model']} at {runner['overall']:.1f}/100. "
                    "Component-level deltas unavailable."
                )

        # Render the summary card
        winner_badge_cls = (
            "score-good" if summary_confidence in ("High", "Medium")
            else "score-ok"
        )
        confidence_badge_cls = (
            "score-good" if summary_confidence in ("High", "Medium")
            else "score-poor" if summary_confidence == "Tie"
            else "score-ok"
        )
        st.markdown(
            f"""
<div style="border:1px solid rgba(99,102,241,.35); border-radius:14px;
            padding:18px 20px; margin:18px 0 8px 0;
            background:linear-gradient(180deg, rgba(99,102,241,.10), rgba(99,102,241,.02));">
  <div style="font-weight:700; font-size:1.05rem; margin-bottom:.6rem;">
    Final Comparison Summary
  </div>
  <div style="display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin-bottom:.6rem;">
    <span class="score-badge {winner_badge_cls}" style="font-size:1.0rem;">
      Winner: {summary_winner}
    </span>
    <span class="score-badge {confidence_badge_cls}">
      Confidence: {summary_confidence}
    </span>
    {f'<span class="score-badge score-ok">Lead: {abs(summary_delta):.1f} pts</span>' if summary_delta else ''}
  </div>
  <div style="opacity:.9; font-size:.95rem; line-height:1.55;">
    {summary_reason}
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        # Per-model scorecard table for the summary
        summary_rows = []
        for s in scored:
            summary_rows.append({
                "Model": s["model"],
                "Overall (0–100)": s["overall"] if s["overall"] is not None else None,
                "LLM Judge (/5)":   round(s["judge"] * 5, 2) if isinstance(s["judge"], (int, float)) else None,
                "Factual (0–1)":    s["factual"],
                "Semantic (0–1)":   s["semantic"],
                "Length sanity":    s["length"],
                "Confidence":       (s["wc"].get("confidence") or "—"),
            })
        st.markdown("**Per-model scorecard**")
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )

        # Surface hallucinations / missing key points across all models
        any_hallucinations = any(s["halls"] for s in scored)
        any_missing = any(s["missing"] for s in scored)
        if any_hallucinations or any_missing:
            with st.expander(
                "Issues detected by the judge across all models",
                expanded=False,
            ):
                for s in scored:
                    if not (s["halls"] or s["missing"]):
                        continue
                    st.markdown(f"**{s['model']}**")
                    if s["halls"]:
                        st.markdown("• Hallucinations:")
                        for h in s["halls"]:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;– {h}", unsafe_allow_html=True)
                    if s["missing"]:
                        st.markdown("• Missing key points:")
                        for m in s["missing"]:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;– {m}", unsafe_allow_html=True)

    # ---- Comparison table ----
    section_header("Comparison Table", "All numeric metrics side-by-side.")
    keys = ["Model","Accuracy","Relevance","Effectiveness","Latency_s","Token_Usage",
            "Prompt_Tokens","Completion_Tokens","Cost_USD","Temperature","Top_P"]
    df = pd.DataFrame([{k: r.get(k) for k in keys} for r in res])
    st.dataframe(df.set_index("Model").T, use_container_width=True)

    jp = st.session_state.last_json_path
    if jp and os.path.exists(jp):
        with open(jp, "rb") as f:
            st.download_button(
                "Download Results JSON", f.read(),
                file_name=os.path.basename(jp), mime="application/json",
                use_container_width=True,
            )

# ===========================================================================
# TAB: Dashboard
# ===========================================================================
with t_dash:
    html = st.session_state.dashboard_html
    if not html:
        st.info("Run an experiment to generate the rich HTML dashboard.")
        st.stop()
    section_header("HTML Dashboard", "Standalone interactive dashboard generated from the latest experiment.")
    st.download_button(
        "Download Dashboard HTML",
        html, file_name="ab_dashboard.html", mime="text/html",
        use_container_width=True,
    )
    st.components.v1.html(html, height=920, scrolling=True)
