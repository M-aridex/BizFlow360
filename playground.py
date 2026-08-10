import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import base64
import os
import json
import re
import html as _html
import warnings
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ============================================================================
# SECTION 1: SUPPRESS WARNINGS & PAGE CONFIG
# ============================================================================

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

st.set_page_config(page_title="BizFlow360 | AI Financial Advisor", layout="wide", page_icon="📊")

# ============================================================================
# SECTION 2: CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
.main .block-container{padding-top:2.2rem;max-width:1150px;}
.hero{text-align:center;padding:1.2rem 0 .6rem;}
.hero-title{font-size:2.8rem;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(92deg,#0f766e,#0891b2,#0284c7);
  -webkit-background-clip:text;background-clip:text;color:transparent;margin:0;}
.hero-sub{opacity:.72;font-size:1.05rem;margin-top:.3rem;}
.eyebrow{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;opacity:.6;margin:.8rem 0 .3rem;}
.card{background:rgba(148,163,184,.10);border:1px solid rgba(148,163,184,.22);
  border-radius:16px;padding:1.05rem 1.2rem;margin:.35rem 0;}
.card b,.card h3{margin:0 0 .3rem 0;}
.pill{display:inline-block;padding:.3rem .95rem;border-radius:999px;font-weight:700;font-size:.9rem;}
.pill-low{background:rgba(34,197,94,.16);color:#16a34a;border:1px solid rgba(34,197,94,.45);}
.pill-med{background:rgba(234,179,8,.16);color:#ca8a04;border:1px solid rgba(234,179,8,.45);}
.pill-high{background:rgba(239,68,68,.16);color:#dc2626;border:1px solid rgba(239,68,68,.45);}
.muted{opacity:.6;}
.tier{padding:.6rem .9rem;border-radius:10px;margin:.35rem 0;border:1px solid rgba(148,163,184,.22);}
.bz-row{display:flex;margin:.4rem 0;width:100%;}
.bz-bubble{max-width:75%;padding:.65rem .9rem;border-radius:12px;font-size:.95rem;line-height:1.5;
  box-shadow:0 1px 1px rgba(0,0,0,.08);word-wrap:break-word;}
.bz-ai{background:#ffffff;color:#111b21;border-top-left-radius:4px;border:1px solid #e9edef;}
.bz-user{background:#d9fdd3;color:#111b21;border-top-right-radius:4px;margin-left:auto;border:1px solid #d1f0cc;}
@media (prefers-color-scheme: dark) {
  .bz-ai{background:#202c33;color:#e9edef;border-color:#2a3942;}
  .bz-user{background:#005c4b;color:#e9edef;border-color:#005c4b;}
}
.bz-bubble img{border-radius:8px;margin-top:.4rem;max-width:240px;display:block;}
div[data-testid="stPopover"]>button{border-radius:999px !important;}
div[data-testid="stChatInput"]>div{border-radius:999px !important;}
@media (max-width:768px){
  .hero-title{font-size:1.8rem;}
  .main .block-container{padding:1rem .9rem;}
  .bz-bubble{max-width:90%;}
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 3: CONFIGURATION & CONSTANTS
# ============================================================================

TEXT_MODEL = "z-ai/glm-5.2"
VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
BASE = Path(__file__).resolve().parent

# Model artifact paths (v2.1 production engine)
ENGINE_PATH = BASE / "ml_models/models/trained/on_real_data/bizflow_engine_v2.1.joblib"
BASE_ENGINE_PATH = BASE / "ml_models/models/trained/on_real_data/bizflow_engine_v2.1_base.joblib"
CARD_PATH = BASE / "ml_models/models/trained/on_real_data/model_card_v2.1.json"

# Local storage files (user history)
HISTORY_FILE = BASE / "bizflow_history.json"
CHAT_SESSIONS_FILE = BASE / "chat_sessions.json"

# Feature display names (must match model's 23-feature input order)
DISPLAY_NAMES = [
    "County", "Sector", "Male Owners", "Female Owners",
    "Total Monthly Expenses", "Monthly Rent", "Monthly Electricity",
    "Monthly Credit Payments", "Monthly Social Responsibility",
    "Revenue Last Month", "Normal Monthly Revenue", "Stock (Beginning)",
    "Stock (End)", "Annual Turnover", "Revenue Change Ratio",
    "Closed Any Establishment", "Number Closed", "Revenue Declined",
    "Low Revenue Flag", "Expense-to-Revenue Ratio", "Stock Change",
    "Turnover Missing", "Revenue Missing"
]

# User-facing disclaimer (shown on multiple pages)
DISCLAIMER = ("⚖️ **Prediction vs advice:** the risk score is a calibrated statistical prediction from "
              "historical KNBS survey data. Recommendations are AI-generated suggestions for discussion — "
              "not guarantees, and not professional financial, legal or tax advice.")

# AI Advisor system prompt (enforces evidence-based, domain-locked responses)
FINANCIAL_SYSTEM_PROMPT = """You are BizFlow Advisor, the AI financial assistant inside BizFlow360, an early-warning system for Kenyan MSMEs.

STRICT DOMAIN LOCK - discuss only MSME/business finance and the BizFlow360 risk output.

EVIDENCE RULES (mandatory):
1. Use ONLY the numbers provided in the user context. Never invent external facts about counties, competition, markets, or timelines.
2. Never promise outcomes or timeframes.
3. Separate prediction from advice: the score is a statistical prediction; suggestions are recommendations, not guarantees.
4. Be simple, warm, practical. Use KES. Explain jargon briefly."""

# ============================================================================
# SECTION 4: API & CLIENT HELPERS
# ============================================================================

def get_api_key():
    """Retrieve NVIDIA API key from secrets, env, or fallback."""
    try:
        k = st.secrets.get("NVIDIA_API_KEY", None)
    except Exception:
        k = None
    return k or os.getenv("NVIDIA_API_KEY")

@st.cache_resource
def get_client(api_key):
    """Initialize OpenAI client for NVIDIA NIM (cached to avoid re-init)."""
    try:
        return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    except Exception:
        return None

def stream_chat(client, model, messages, max_tokens=2048):
    """Stream AI chat completion chunks (yields text tokens)."""
    completion = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, top_p=1,
        max_tokens=max_tokens, seed=42, stream=True)
    for chunk in completion:
        if not getattr(chunk, "choices", None) or len(chunk.choices) == 0: continue
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None): yield delta.content

# ============================================================================
# SECTION 5: UTILITY FUNCTIONS (Images, Markdown, Text Cleaning)
# ============================================================================

def image_to_data_url(data, mime="image/jpeg"):
    """Convert binary image data to base64 data URL for HTML embedding."""
    return f"data:{mime};base64," + base64.b64encode(data).decode()

def extract_document_text(file):
    """Extract text from uploaded PDF or plain text file."""
    try:
        if file.name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(file).pages[:10])
        return file.getvalue().decode("utf-8", errors="ignore")[:8000]
    except Exception:
        return None

def md_to_html(text):
    """Convert simple markdown to HTML for chat bubble rendering."""
    if not text: return ""
    t = _html.escape(str(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.S)
    t = re.sub(r"^#{1,4}\s*(.+)$", r"<b style='font-size:1.02em;'>\1</b>", t, flags=re.M)
    t = re.sub(r"^\s*(\d+)\.\s+(.+)$", r"<span style='display:block;padding-left:1.5em;text-indent:-1.5em;'>\1. \2</span>", t, flags=re.M)
    t = re.sub(r"^\s*[-•]\s+(.+)$", r"<span style='display:block;padding-left:1em;text-indent:-.9em;'>• \1</span>", t, flags=re.M)
    return t.replace("\n", "<br>")

def render_bubble(role, text, img_urls=None):
    """Render a WhatsApp-style chat bubble (user or AI)."""
    cls = "bz-user" if role == "user" else "bz-ai"
    imgs = "".join([f'<img src="{u}">' for u in (img_urls or [])])
    st.markdown(f'<div class="bz-row"><div class="bz-bubble {cls}">{md_to_html(text)}{imgs}</div></div>',
                unsafe_allow_html=True)

# ============================================================================
# SECTION 6: HISTORY & CHAT SESSION PERSISTENCE
# ============================================================================

def load_history():
    """Load analysis history from local JSON file."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f: return json.load(f)
    return []

def save_history(record):
    """Append a new analysis record to history file."""
    history = load_history(); history.append(record)
    with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=2)

def load_chat_sessions():
    """Load archived chat sessions from local JSON file."""
    if CHAT_SESSIONS_FILE.exists():
        with open(CHAT_SESSIONS_FILE, "r") as f: return json.load(f)
    return []

def save_chat_sessions(sessions):
    """Save chat sessions to local JSON file."""
    with open(CHAT_SESSIONS_FILE, "w") as f: json.dump(sessions, f, indent=2)

def save_current_chat_to_history():
    """Archive the current chat session (called on new chat or page switch)."""
    if "chat" in st.session_state and len(st.session_state["chat"]) > 1:
        sessions = load_chat_sessions()
        current_id = st.session_state.get("current_chat_id")
        if current_id:
            for s in sessions:
                if s["id"] == current_id:
                    s["messages"] = st.session_state["chat"]
                    for msg in st.session_state["chat"]:
                        if msg["role"] == "user":
                            s["title"] = msg["display"][:40]; break
                    save_chat_sessions(sessions); return
        title = "New Chat"
        for msg in st.session_state["chat"]:
            if msg["role"] == "user":
                title = msg["display"][:40]; break
        new_id = str(datetime.now().timestamp())
        sessions.insert(0, {"id": new_id, "title": title,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "messages": st.session_state["chat"]})
        save_chat_sessions(sessions)
        st.session_state["current_chat_id"] = new_id

# ============================================================================
# SECTION 7: FINANCIAL CALCULATIONS & ACTION PLAN BUILDER
# ============================================================================

def compute_financial_math(revenue, expenses, net_income, revenue_change_ratio):
    """
    Compute key financial metrics from user inputs.
    Returns dict with deficit, break_even, rev_gap, exp_cut, margin, pct_change.
    """
    effective_net = net_income if net_income != 0 else (revenue - expenses)
    deficit = max(0.0, expenses - revenue)
    break_even = expenses
    rev_gap = max(0.0, break_even - revenue)
    exp_cut = max(0.0, expenses - revenue)
    margin = (effective_net / revenue) if revenue > 0 else 0.0
    pct_change = (revenue_change_ratio - 1) * 100
    return {"deficit": deficit, "break_even": break_even, "rev_gap": rev_gap,
            "exp_cut": exp_cut, "margin": margin, "pct_change": pct_change}

def build_action_plan(m):
    """
    Build instant rule-based quick wins (kept for backward compatibility / PDF summary).
    """
    a = []
    if m["deficit"] > 0:
        a.append(("🔴 Critical", "Stop the monthly bleed",
                  f"Expenses exceed revenue by KES {m['deficit']:,.0f} per month."))
    if m["pct_change"] <= -10:
        a.append(("🟠 High", "Stabilise falling revenue",
                  f"Revenue is {abs(m['pct_change']):.0f}% below your normal month."))
    if m["deficit"] <= 0 and 0 <= m["margin"] < 0.15:
        a.append(("🟡 Medium", "Thicken your margin",
                  f"Your margin is {m['margin']*100:.0f}%. Aim for 20%+."))
    if m["deficit"] <= 0 and m["margin"] >= 0.15:
        a.append(("🟢 Growth", "Build a buffer",
                  "Set aside about 10% of monthly profit into a separate account."))
    if not a:
        a.append(("🟡 Medium", "Keep records current",
                  "Update revenue and expenses weekly."))
    return a

# ============================================================================
# SECTION 8: PDF GENERATION (with text sanitization for fpdf2)
# ============================================================================

def strip_emojis(text):
    """Remove emoji characters from text."""
    if not text: return ""
    emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF" "\U00002702-\U000027B0" "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF" "\U0001FA70-\U0001FAFF" "\U00002600-\U000026FF"
        "\U0001F3FB-\U0001F3FF" "\u200d" "\ufe0f" "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))

def clean_for_pdf(text):
    """Sanitize text for fpdf2 (remove emojis, smart quotes, markdown)."""
    if not text: return ""
    t = strip_emojis(text)
    for k, v in {"—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
                 "•": "-", "…": "...", "→": "->", "⚖": ""}.items():
        t = t.replace(k, v)
    for md in ["**", "##", "__"]:
        t = t.replace(md, "")
    return t.encode("latin-1", "ignore").decode("latin-1")

def generate_pdf(prob, label, top_factors, math, actions, ai_text, county, sector, card):
    """
    Generate branded PDF report with risk score, SHAP drivers, action plan, and AI explanation.
    Returns bytes or None if fpdf2 not installed.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    class BizPDF(FPDF):
        def footer(self):
            self.set_y(-12); self.set_font("Arial", "I", 8); self.set_text_color(140, 140, 140)
            self.cell(0, 8, "BizFlow360  |  Financial clarity for every MSME  |  Page {nb}", align="C")

    pdf = BizPDF(); pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page(); pdf.set_left_margin(15); pdf.set_right_margin(15)

    # Header band (branded blue)
    pdf.set_fill_color(30, 58, 138); pdf.rect(0, 0, 210, 32, "F")
    pdf.set_xy(15, 9); pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 17); pdf.multi_cell(0, 9, "BizFlow360 - Financial Health Report")
    pdf.set_x(15); pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 6, "AI-Powered Early Warning System for Kenyan MSMEs")

    # Meta info (date, business)
    pdf.set_y(40); pdf.set_x(15); pdf.set_text_color(90, 90, 90); pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, f"Date: {datetime.now().strftime('%d %B %Y, %H:%M')}")
    pdf.set_x(15); pdf.multi_cell(0, 6, f"Business: {clean_for_pdf(sector)}   |   {clean_for_pdf(county)} County")
    pdf.ln(4)

    # Risk badge (color-coded)
    if prob < 0.4: dark, light = (22, 163, 74), (232, 245, 233)
    elif prob < 0.7: dark, light = (202, 138, 4), (250, 240, 220)
    else: dark, light = (220, 38, 38), (250, 230, 230)
    pdf.set_x(15); pdf.set_fill_color(*light); pdf.set_draw_color(*dark); pdf.set_text_color(*dark)
    pdf.set_font("Arial", "B", 13)
    pdf.multi_cell(0, 12, f"   Distress Risk Score (calibrated): {prob*100:.1f}%   ({label})", border=1, fill=True)
    pdf.ln(4)

    # Computed numbers section
    pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", "B", 13)
    pdf.multi_cell(0, 9, "Your Numbers, Computed"); pdf.set_font("Arial", "", 10.5)
    for ln_ in [
        f"- Revenue changed by {math['pct_change']:+.1f}% compared with your normal month.",
        f"- Current monthly deficit: KES {math['deficit']:,.0f}." if math['deficit'] > 0 else "- No monthly deficit: revenue covers expenses.",
        f"- Break-even revenue (expenses constant): KES {math['break_even']:,.0f}/month.",
        f"- Required revenue increase to break even: KES {math['rev_gap']:,.0f}.",
        f"- Expense reduction required to break even: KES {math['exp_cut']:,.0f}."]:
        pdf.set_x(15); pdf.multi_cell(0, 7, clean_for_pdf(ln_))
    pdf.ln(3)

    # SHAP drivers section
    pdf.set_font("Arial", "B", 13); pdf.multi_cell(0, 9, "Top Risk Drivers (SHAP)"); pdf.set_font("Arial", "", 10.5)
    for _, row in top_factors.iterrows():
        direction = "increases risk" if row['Impact'] > 0 else "reduces risk"
        pdf.set_x(15)
        pdf.multi_cell(0, 7, f"- {clean_for_pdf(row['Feature'])}: {direction} (impact {row['Impact']:+.3f})")
    pdf.ln(3)

    # AI explanation & 10 Actions section (combined)
    if ai_text:
        pdf.set_font("Arial", "B", 13); pdf.multi_cell(0, 9, "AI Advisor Deep Analysis & 10 Actions")
        pdf.set_font("Arial", "", 10.5); pdf.set_x(15)
        pdf.multi_cell(0, 6.5, clean_for_pdf(ai_text))
    pdf.ln(3)

    # Footer disclaimer
    pdf.set_font("Arial", "I", 9); pdf.set_text_color(90, 90, 90); pdf.set_x(15)
    pdf.multi_cell(0, 6, clean_for_pdf(
        "Prediction vs advice: the risk score is a calibrated statistical prediction from historical KNBS survey data ("
        + card.get("model", "LightGBM") + "). Recommendations are AI-generated suggestions, not guarantees, "
        "and not professional financial, legal or tax advice."))
    return bytes(pdf.output())

# ============================================================================
# SECTION 9: MODEL LOADING & SECTOR MAPPING
# ============================================================================

@st.cache_resource
def load_engine():
    """Load v2.1 model artifacts (returns tuple or None if missing)."""
    if not (ENGINE_PATH.exists() and BASE_ENGINE_PATH.exists() and CARD_PATH.exists()):
        return None
    return (joblib.load(ENGINE_PATH), joblib.load(BASE_ENGINE_PATH),
            json.loads(CARD_PATH.read_text()))

resources = load_engine()
API_KEY = get_api_key()
client = get_client(API_KEY) if API_KEY else None

# Sector code -> friendly name mapping (for UI display)
SECTOR_FRIENDLY = {
    "01": "🌾 Farming & Livestock (Shamba, Dairy, Poultry)", "02": "🌳 Forestry & Tree Farming",
    "10": "🍞 Food Processing (Bakery, Milling)", "14": "👗 Tailoring & Clothing (Boutique, Fundi)",
    "15": "👞 Leather & Shoemaking", "16": "🪚 Carpentry & Woodwork", "24": "⚙️ Basic Metal Works",
    "25": "🔧 Jua Kali Metal Fabrication (Welding, Mabati)", "28": "🏭 Machinery Making",
    "31": "🪑 Furniture Making", "45": "🏍️ Garage & Motor Vehicle Trade/Repair",
    "46": "📦 Wholesale & Distribution", "47": "🛒 Retail Shop / Kiosk (Duka, Mama Mboga)",
    "49": "🚕 Transport (Boda Boda, Taxi, Matatu)", "55": "🏠 Guest House / Lodging",
    "56": "🍽️ Eatery / Food Vendor (Kibanda, Catering)", "62": "💻 IT & Cyber Services",
    "64": "💰 Money Services (M-Pesa Agent, Lender, SACCO)", "66": "🛡️ Insurance & Financial Support",
    "68": "🏢 Real Estate & Rentals", "74": "📷 Professional Services (Photography, Design)",
    "79": "✈️ Travel & Tour Agent", "82": "🖨️ Business Support (Printing, Secretarial)",
    "85": "🎓 School / Education (Daycare, Tuition)", "86": "🏥 Clinic / Chemist",
    "90": "🎨 Arts & Entertainment", "92": "🎰 Betting & Gambling",
    "93": "⚽ Sports & Recreation (Gym, Pitch)", "95": "🔌 Repair Services (Electronics, Computers)",
    "96": "💇 Salon / Barbershop / Laundry",
}

def friendly_sector(isic_string):
    """Convert ISIC sector code to user-friendly name."""
    code = str(isic_string).split("-")[0].strip()
    return SECTOR_FRIENDLY.get(code, str(isic_string))

def short_sector_label(isic_string):
    """Short, plot-friendly sector label (no emoji, no parenthetical) for SHAP charts."""
    friendly = friendly_sector(str(isic_string))
    friendly = re.sub(r"^[^A-Za-z0-9]+", "", friendly)   # strip leading emoji/symbols
    return friendly.split("(")[0].strip()                # drop parenthetical

def build_plot_values(row_values):
    """
    Build short display values for the SHAP waterfall plot.
    """
    plot_values = []
    for name, val in zip(DISPLAY_NAMES, row_values):
        if name == "Sector":
            plot_values.append(short_sector_label(val))
        elif name == "County":
            plot_values.append(str(val).title())
        else:
            try:
                plot_values.append(round(float(val), 2))
            except (TypeError, ValueError):
                plot_values.append(str(val))
    return plot_values

# ============================================================================
# SECTION 10: SIDEBAR & NAVIGATION
# ============================================================================

logo_path = BASE / "logo-1.png"
logo_exists = logo_path.exists()

def logo_b64():
    """Load logo as base64 for HTML embedding."""
    with open(logo_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

with st.sidebar:
    if logo_exists:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;">
        <img src="data:image/png;base64,{logo_b64()}" style="height:42px;width:auto;border-radius:8px;">
        <span style="font-size:1.4rem;font-weight:800;">BizFlow360</span></div>""", unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation",
        ["🏠 Welcome", "📊 Analyze", "📜 History", "💬 Chat", "⚙️ Settings"],
        key="nav", label_visibility="collapsed")
    st.sidebar.caption("Financial clarity for every MSME.")

# ============================================================================
# SECTION 11: PAGE 1 — WELCOME (Onboarding)
# ============================================================================
if page == "🏠 Welcome":
    if logo_exists:
        st.markdown(f'<div style="display:flex;justify-content:center;margin:1rem 0 2rem 0;">'
                    f'<img src="data:image/png;base64,{logo_b64()}" style="height:160px;width:auto;border-radius:16px;"></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1 class="hero-title">Welcome to BizFlow360</h1>'
                '<div class="hero-sub">AI-Powered Financial Early Warning System for Kenyan MSMEs</div></div>',
                unsafe_allow_html=True)
    st.info(DISCLAIMER)
    st.markdown('<div class="eyebrow">How it works</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="card"><b>1 · Analyze Your Business</b><br>Enter your county, sector and monthly financials to get a calibrated distress risk score from our LightGBM engine trained on real KNBS data.</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><b>2 · Understand Your Risk</b><br>A SHAP chart plus plain-language meanings show exactly why you got your score, including how your revenue changed versus your normal month.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><b>3 · Get a Data-Driven Action Plan</b><br>We compute your deficit, break-even revenue and required changes from YOUR numbers, then generate 10 deeply analyzed AI actions tailored to your sector and download everything as a PDF.</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><b>4 · Track & Chat</b><br>Use History to track progress month over month, or Chat with the Advisor — upload receipts or take a photo of your books.</div>', unsafe_allow_html=True)

    def go_to_analyze():
        st.session_state["nav"] = "📊 Analyze"
    if st.button("🚀 Get Started", type="primary", on_click=go_to_analyze):
        pass

# ============================================================================
# SECTION 12: PAGE 2 — ANALYZE (Core Prediction Engine)
# ============================================================================
elif page == "📊 Analyze":
    # --- Check model availability ---
    if resources is None:
        st.error("Engine v2.1 not found. Run this once first:\n\n`python ml_models/scripts/train_bizflow_engine_v2_1.py`")
        st.stop()
    engine, base_engine, card = resources

    # --- Extract county/sector lists from model encoder ---
    enc = base_engine.named_steps["pre"].transformers_[0][1].named_steps["enc"]
    county_list = list(enc.categories_[0])
    sector_list = list(enc.categories_[1])
    sector_options = {friendly_sector(s): s for s in sector_list}

    # --- Sidebar input form ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📝 Business Profile")
    county = st.sidebar.selectbox("County", county_list)
    sector_friendly = st.sidebar.selectbox("Business Sector", list(sector_options.keys()))
    sector_full = sector_options[sector_friendly]

    male_owners = st.sidebar.number_input("Male Working Owners", 0, 20, 1)
    female_owners = st.sidebar.number_input("Female Working Owners", 0, 20, 0)

    st.sidebar.markdown("#### 💰 Monthly Financials (KES)")
    revenue_last_month = st.sidebar.number_input("Last Month's Revenue", 0, 5000000, 150000, step=10000)
    normal_monthly_revenue = st.sidebar.number_input("Normal Monthly Revenue", 0, 5000000, 150000, step=10000)
    net_income_last_month = st.sidebar.number_input("Last Month's Net Income", -500000, 5000000, 30000, step=5000)
    total_monthly_expenses = st.sidebar.number_input("Total Monthly Expenses", 0, 5000000, 120000, step=10000)

    st.sidebar.markdown("#### 📦 Assets & History")
    stock_beginning = st.sidebar.number_input("Stock Value (Beginning)", 0, 10000000, 100000, step=10000)
    stock_end = st.sidebar.number_input("Stock Value (End)", 0, 10000000, 120000, step=10000)
    total_turnover = st.sidebar.number_input("Annual Turnover", 0, 50000000, 1800000, step=100000)
    business_closed = st.sidebar.checkbox("Closed an establishment in last 5 yrs?")
    num_closed = st.sidebar.number_input("Number closed", 0, 10, 0) if business_closed else 0
    revenue_decline = st.sidebar.checkbox("Revenue declined recently?")

    analyze_clicked = st.sidebar.button("🔍 Analyze My Business", type="primary")

    # --- Run prediction when button clicked ---
    if analyze_clicked:
        with st.spinner("Running analysis..."):
            # Compute engineered features (must match v2.1 training script)
            rcr = float(np.clip(revenue_last_month / normal_monthly_revenue, 0.0, 20.0)) if normal_monthly_revenue > 0 else 1.0
            expense_to_revenue = float(np.clip(total_monthly_expenses / max(revenue_last_month, 1.0), 0.0, 10.0))
            stock_change = float(np.clip((stock_end - stock_beginning) / max(stock_beginning, 1.0), -1.0, 5.0)) if stock_beginning > 0 else 0.0
            turnover_missing = int(pd.isna(total_turnover) or total_turnover == 288000)
            revenue_missing = 0
            low_revenue = 1 if revenue_last_month < 10000 else 0

            # Build feature row (23 columns in model order)
            row = pd.DataFrame([{
                "county": county, "sector": sector_full,
                "male_working_owners": male_owners, "female_working_owners": female_owners,
                "total_monthly_expenses": total_monthly_expenses,
                "monthly_rent_expense": 0.0, "monthly_electricity_expense": 0.0,
                "monthly_credit_expense": 0.0, "monthly_social_responsibility_expense": 0.0,
                "revenue_last_month": revenue_last_month, "normal_monthly_revenue": normal_monthly_revenue,
                "stock_value_beginning": stock_beginning, "stock_value_end": stock_end,
                "total_turnover_2015": total_turnover, "revenue_change_ratio": rcr,
                "business_closed": 1 if business_closed else 0,
                "number_closed_establishments": num_closed,
                "revenue_decline": 1 if revenue_decline else 0, "low_revenue": low_revenue,
                "expense_to_revenue": expense_to_revenue, "stock_change": stock_change,
                "turnover_missing": turnover_missing, "revenue_missing": revenue_missing,
            }])

            # Predict probability (calibrated)
            prob = float(engine.predict_proba(row)[:, 1][0])

            # Compute SHAP values (using base engine for TreeExplainer)
            Xt = base_engine.named_steps["pre"].transform(row)
            explainer = shap.TreeExplainer(base_engine.named_steps["clf"])
            sv = explainer.shap_values(Xt)
            if isinstance(sv, list): sv = sv[1]
            bv = explainer.expected_value
            if isinstance(bv, (list, tuple)): bv = bv[1]

            # Build results
            top_factors = pd.DataFrame({"Feature": DISPLAY_NAMES, "Impact": sv[0]}) \
                            .sort_values("Impact", key=abs, ascending=False).head(5)
            math = compute_financial_math(revenue_last_month, total_monthly_expenses,
                                          net_income_last_month, rcr)
            actions = build_action_plan(math)

            # Save to session state & history
            st.session_state.update({
                "prob_real": prob, "X_row": row, "shap_values": sv, "base_value": bv,
                "top_factors": top_factors, "county": county, "sector": sector_friendly,
                "math": math, "actions": actions, 
                "ai_report_text": None, "ai_actions": None, # Added ai_actions
                "revenue": revenue_last_month, "expenses": total_monthly_expenses,
            })
            save_history({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "county": county, "sector": sector_friendly,
                "risk_score": float(prob),
                "risk_label": "Low" if prob < 0.4 else ("Medium" if prob < 0.7 else "High"),
                "top_factors": top_factors.to_dict("records"),
            })

    # --- Display results if analysis exists ---
    if "prob_real" in st.session_state and st.session_state["prob_real"] is not None:
        prob = st.session_state["prob_real"]
        math = st.session_state["math"]; actions = st.session_state["actions"]
        pill = 'pill-low' if prob < 0.4 else ('pill-med' if prob < 0.7 else 'pill-high')
        label = 'Low Risk' if prob < 0.4 else ('Medium Risk' if prob < 0.7 else 'High Risk')

        # Risk score header cards
        st.markdown('<div class="eyebrow">Financial Health Report</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns([1.2, 1, 1.4], gap="medium")
        with m1:
            st.markdown(f'<div class="card"><div class="eyebrow">Distress Risk Score (calibrated)</div>'
                        f'<h2 style="margin:0;">{prob*100:.1f}%</h2></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="card"><div class="eyebrow">Category</div>'
                        f'<span class="pill {pill}">{label}</span></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="card"><div class="eyebrow">Business</div>'
                        f'<b>{st.session_state["county"]}</b> · {st.session_state["sector"]}</div>',
                        unsafe_allow_html=True)
        st.progress(float(prob), text="Risk level")
        st.caption(f"Revenue changed by {math['pct_change']:+.1f}% vs your normal month · "
                   "the engine flags businesses for review at a score ≥ 33% (validated operating point: "
                   "detects 74.6% of distressed businesses on the held-out test set).")
        st.info(DISCLAIMER)

        # Four tabs for detailed view
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Why this score?", "🧮 Your numbers",
                                          "🎯 Action plan", "🤖 AI explanation"])

        # --- Tab 1: SHAP waterfall & top factors ---
        with tab1:
            plot_values = build_plot_values(st.session_state['X_row'].values[0])

            fig, ax = plt.subplots(figsize=(12, 7.5))
            shap.waterfall_plot(
                shap.Explanation(
                    values=st.session_state['shap_values'][0],
                    base_values=st.session_state['base_value'],
                    data=plot_values,
                    feature_names=DISPLAY_NAMES),
                show=False, max_display=10)
            plt.title("Key Drivers of Your Risk Score", fontsize=14)
            ax.set_xlabel("Impact on risk score (log-odds)", fontsize=10)
            ax.tick_params(axis="x", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown('<div class="eyebrow">Top factors & what they mean</div>', unsafe_allow_html=True)
            meanings = card.get("feature_meanings", card.get("engineered_features", {}))
            for _, r in st.session_state['top_factors'].iterrows():
                up = r['Impact'] > 0
                st.markdown(f'<div class="card" style="padding:.55rem 1rem;">'
                            f'<b>{r["Feature"]}</b> — {"⬆️ increases" if up else "⬇️ reduces"} risk '
                            f'<span class="muted">({r["Impact"]:+.3f})</span><br>'
                            f'<span class="muted">{meanings.get(r["Feature"], "")}</span></div>',
                            unsafe_allow_html=True)

        # --- Tab 2: Computed financial metrics ---
        with tab2:
            st.markdown('<div class="eyebrow">Computed from your inputs</div>', unsafe_allow_html=True)
            g1, g2 = st.columns(2, gap="medium")
            with g1:
                st.markdown(f'<div class="card"><b>Monthly deficit</b><br>KES {math["deficit"]:,.0f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card"><b>Break-even revenue</b><br>KES {math["break_even"]:,.0f}/month (expenses constant)</div>', unsafe_allow_html=True)
            with g2:
                st.markdown(f'<div class="card"><b>Required revenue increase</b><br>KES {math["rev_gap"]:,.0f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card"><b>Expense reduction required</b><br>KES {math["exp_cut"]:,.0f}</div>', unsafe_allow_html=True)
            st.caption(f"Margin: {math['margin']*100:.1f}% · Revenue vs normal: {math['pct_change']:+.1f}%")

        # --- Tab 3: Prioritized action plan (Instant Metrics + 10 AI Actions) ---
        with tab3:
            st.markdown('<div class="eyebrow">Instant Quick Wins</div>', unsafe_allow_html=True)
            for tier, title, detail in actions:
                st.markdown(f'<div class="tier"><b>{tier} {title}</b><br>{detail}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<div class="eyebrow">Deep AI Analysis (10 Tailored Actions)</div>', unsafe_allow_html=True)
            
            if not client:
                st.caption("AI service offline. Contact the development team.")
            else:
                if not st.session_state.get('ai_actions'):
                    if st.button("🎯 Generate 10 AI-Powered Actions", type="primary", key="gen_actions"):
                        st.session_state['generating_actions'] = True
                        st.rerun()

                displayed_actions_this_run = False
                if st.session_state.get('generating_actions'):
                    shap_summary = "\n".join([f"- {r['Feature']}: {r['Impact']:+.3f}"
                                              for _, r in st.session_state['top_factors'].iterrows()])
                    prompt_actions = (
                        f"User context (ONLY use these numbers): county={st.session_state['county']}; "
                        f"sector={st.session_state['sector']}; revenue=KES {st.session_state['revenue']:,.0f}; "
                        f"expenses=KES {st.session_state['expenses']:,.0f}; "
                        f"revenue change vs normal={math['pct_change']:+.1f}%; deficit=KES {math['deficit']:,.0f}; "
                        f"break-even=KES {math['break_even']:,.0f}; risk score={prob*100:.1f}% ({label}).\n"
                        f"Top SHAP drivers:\n{shap_summary}\n\n"
                        f"Task: Write EXACTLY the header '## 🎯 10 Actionable Suggestions' followed by a numbered list of 10 deeply analyzed, "
                        f"data-driven actions tailored to this specific {st.session_state['sector']} business in {st.session_state['county']}. "
                        f"Each action MUST have a short bold title and a detailed paragraph referencing the user's specific KES numbers, margins, or SHAP drivers. "
                        f"Do not write any introductions, conclusions, or other sections. Just the header and the 10 numbered steps."
                    )

                    placeholder = st.empty()
                    full_text = ""
                    try:
                        with st.spinner("🧠 Analyzing your business and writing 10 actions..."):
                            for chunk in stream_chat(client, TEXT_MODEL,
                                                    [{"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                                                     {"role": "user", "content": prompt_actions}], max_tokens=2500):
                                full_text += chunk
                                placeholder.markdown(full_text)
                        st.session_state['ai_actions'] = full_text
                    except Exception as e:
                        st.error(f"AI request failed: {e}")
                        placeholder.empty()

                    st.session_state['generating_actions'] = False
                    displayed_actions_this_run = True

                if st.session_state.get('ai_actions') and not displayed_actions_this_run:
                    st.markdown(st.session_state['ai_actions'])

        # --- Tab 4: AI explanation (streamed with spinner) ---
        with tab4:
            if not client:
                st.caption("AI service offline. Contact the development team.")
            else:
                if not st.session_state.get('ai_report_text'):
                    if st.button("🤖 Generate AI Explanation", type="primary", key="gen_explain"):
                        st.session_state['generating_report'] = True
                        st.rerun()

                displayed_this_run = False
                if st.session_state.get('generating_report'):
                    shap_summary = "\n".join([f"- {r['Feature']}: {r['Impact']:+.3f}"
                                              for _, r in st.session_state['top_factors'].iterrows()])
                    prompt_explain = (
                        f"User context (ONLY use these numbers): county={st.session_state['county']}; "
                        f"sector={st.session_state['sector']}; revenue=KES {st.session_state['revenue']:,.0f}; "
                        f"expenses=KES {st.session_state['expenses']:,.0f}; "
                        f"revenue change vs normal={math['pct_change']:+.1f}%; deficit=KES {math['deficit']:,.0f}; "
                        f"break-even=KES {math['break_even']:,.0f}; risk score={prob*100:.1f}% ({label}).\n"
                        f"Top SHAP drivers:\n{shap_summary}\n\n"
                        f"Task: Write two sections:\n"
                        f"1. '## 📖 Simple Explanation' in plain language using ONLY the numbers above.\n"
                        f"2. '## 💡 Encouragement' with a warm, practical closing.\n"
                        f"State clearly that the score is a statistical prediction, not a guarantee."
                    )

                    placeholder = st.empty()
                    full_text = ""
                    try:
                        with st.spinner("🧠 Connecting to AI Advisor and writing your explanation..."):
                            for chunk in stream_chat(client, TEXT_MODEL,
                                                    [{"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                                                     {"role": "user", "content": prompt_explain}], max_tokens=2000):
                                full_text += chunk
                                placeholder.markdown(full_text)
                        st.session_state['ai_report_text'] = full_text
                    except Exception as e:
                        st.error(f"AI request failed: {e}")
                        placeholder.empty()

                    st.session_state['generating_report'] = False
                    displayed_this_run = True

                if st.session_state.get('ai_report_text') and not displayed_this_run:
                    st.markdown(st.session_state['ai_report_text'])

                # PDF Download Button (combines both AI outputs if available)
                if st.session_state.get('ai_actions') or st.session_state.get('ai_report_text'):
                    combined_ai_text = ""
                    if st.session_state.get('ai_report_text'):
                        combined_ai_text += st.session_state['ai_report_text'] + "\n\n"
                    if st.session_state.get('ai_actions'):
                        combined_ai_text += st.session_state['ai_actions']
                    
                    pdf_bytes = generate_pdf(prob, label, st.session_state['top_factors'], math, actions,
                                             combined_ai_text,
                                             st.session_state['county'], st.session_state['sector'], card)
                    if pdf_bytes:
                        st.download_button("📥 Download Full Report as PDF", pdf_bytes,
                                           f"BizFlow_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                           "application/pdf")
    else:
        st.markdown('<div class="hero"><h2 class="hero-title" style="font-size:1.6rem;">Ready when you are</h2>'
                    '<div class="hero-sub">Fill in your business profile on the left and press <b>Analyze My Business</b>.</div></div>',
                    unsafe_allow_html=True)

# ============================================================================
# SECTION 13: PAGE 3 — HISTORY (Trend Tracking)
# ============================================================================
elif page == "📜 History":
    st.markdown('<div class="eyebrow">Your journey</div><h2 style="margin-top:0;">📜 Analysis History</h2>',
                unsafe_allow_html=True)
    history = load_history()
    if not history:
        st.info("No past analyses yet. Run your first analysis in **📊 Analyze**.")
    else:
        df_hist = pd.DataFrame(history)
        df_hist['Risk %'] = (df_hist['risk_score'] * 100).round(1)
        c1, c2 = st.columns([1.3, 1], gap="medium")
        with c1:
            st.markdown('<div class="eyebrow">Risk trend over time</div>', unsafe_allow_html=True)
            st.line_chart(df_hist.set_index(pd.to_datetime(df_hist['timestamp']))['Risk %'], height=260)
        with c2:
            best = df_hist['Risk %'].min(); latest = df_hist['Risk %'].iloc[-1]
            st.markdown(f'<div class="card"><div class="eyebrow">Latest score</div><h2 style="margin:0;">{latest}%</h2>'
                        f'<div class="eyebrow">Best score</div><h2 style="margin:0;">{best}%</h2></div>',
                        unsafe_allow_html=True)
        st.dataframe(df_hist[['timestamp', 'county', 'sector', 'Risk %', 'risk_label']],
                     use_container_width=True, hide_index=True)
        st.markdown("#### 🔄 Compare with a past analysis")
        past_dates = [f"{h['timestamp']} — {h['risk_label']} ({h['risk_score']*100:.1f}%)" for h in history]
        selected = st.selectbox("Pick a past analysis", past_dates)
        if st.button("🧠 Ask AI to Compare", type="primary"):
            if 'prob_real' not in st.session_state or st.session_state['prob_real'] is None:
                st.warning("Run a current analysis in **📊 Analyze** first.")
            elif not client:
                st.caption("AI service offline. Contact the development team.")
            else:
                past = history[past_dates.index(selected)]
                prompt = (f"Compare two BizFlow360 analyses for the same MSME using only these numbers.\n"
                          f"PAST ({past['timestamp']}): risk {past['risk_score']*100:.1f}% ({past['risk_label']}).\n"
                          f"CURRENT: risk {st.session_state['prob_real']*100:.1f}%.\n"
                          f"Give a brief, encouraging, evidence-based progress review. No invented facts or timelines.")
                with st.spinner("Comparing..."):
                    try:
                        st.write_stream(stream_chat(client, TEXT_MODEL,
                            [{"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}]))
                    except Exception as e:
                        st.error(f"AI request failed: {e}")

# ============================================================================
# SECTION 14: PAGE 4 — CHAT (WhatsApp-Style AI Advisor)
# ============================================================================
elif page == "💬 Chat":
    # --- Sidebar chat history ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 📜 Chat History")
        sessions = load_chat_sessions()
        if sessions:
            for s in sessions[:10]:
                is_current = st.session_state.get("current_chat_id") == s["id"]
                btn_type = "primary" if is_current else "secondary"
                if st.button(f"💬 {s['title']}", key=f"hist_{s['id']}", use_container_width=True, type=btn_type):
                    save_current_chat_to_history()
                    st.session_state["chat"] = s["messages"]
                    st.session_state["current_chat_id"] = s["id"]
                    st.rerun()
        else:
            st.caption("No past chats yet.")

    st.markdown("#### 💬 BizFlow Advisor")
    st.caption(DISCLAIMER)
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("current_chat_id", None)
    if not st.session_state["chat"]:
        st.session_state["chat"].append({
            "role": "assistant",
            "display": "👋 **Karibu!** I'm your BizFlow Advisor. Ask me anything about your business finances — cash flow, expenses, pricing, M-Pesa, loans — or attach a receipt/photo and I'll help you make sense of it.",
            "img_urls": [], "api": "Greeting sent."})

    # New chat button
    c1, c2 = st.columns([5, 1])
    with c2:
        if st.button("➕ New Chat", use_container_width=True):
            save_current_chat_to_history()
            st.session_state["chat"] = []
            st.session_state["current_chat_id"] = None
            st.rerun()

    # Suggestion chips (only on fresh chat)
    suggestion = None
    if len(st.session_state["chat"]) <= 1:
        s1, s2, s3 = st.columns(3)
        with s1:
            if st.button("💡 Cut expenses", use_container_width=True):
                suggestion = "How can I cut my monthly expenses without hurting sales?"
        with s2:
            if st.button("📉 Revenue dropped", use_container_width=True):
                suggestion = "My revenue dropped this month. What practical steps should I take?"
        with s3:
            if st.button("🧾 Explain risk score", use_container_width=True):
                suggestion = "Explain my current BizFlow360 risk score in simple language."

    # Chat message display area
    try:
        chat_area = st.container(height=480, border=True)
    except TypeError:
        chat_area = st.container()
    with chat_area:
        for msg in st.session_state["chat"]:
            render_bubble(msg["role"], msg["display"], msg.get("img_urls"))

    # File/camera attachment expander
    with st.expander("📎 Attach a receipt or document (Optional)", expanded=False):
        uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "pdf", "csv", "txt"], label_visibility="collapsed")
        camera_photo = st.camera_input("Or take a photo", label_visibility="collapsed") if hasattr(st, "camera_input") else None

    # Chat input (pinned to bottom)
    prompt = st.chat_input("Type a message...")
    user_text = prompt or suggestion

    # Process user input & generate AI response
    att_image_url, att_text = None, None
    source = camera_photo or uploaded_file
    if source is not None:
        if source.type.startswith("image/"):
            att_image_url = image_to_data_url(source.getvalue(), source.type)
        else:
            att_text = extract_document_text(source)

    if user_text:
        if not client:
            st.error("AI service offline. Please contact the development team.")
        else:
            text_part = user_text
            if att_text:
                text_part = f"{user_text}\n\n--- Document ({source.name}) ---\n{att_text}\n---"
            api_content = [{"type": "text", "text": text_part}]
            if att_image_url:
                api_content.append({"type": "image_url", "image_url": {"url": att_image_url}})
            st.session_state["chat"].append({"role": "user", "display": user_text,
                                             "img_urls": [att_image_url] if att_image_url else [], "api": api_content})

            # Build system prompt with live context
            system = FINANCIAL_SYSTEM_PROMPT
            if st.session_state.get('prob_real') is not None:
                system += (f"\n\nUSER'S LIVE CONTEXT (only use these): risk {st.session_state['prob_real']*100:.1f}%; "
                           f"county {st.session_state.get('county','')}; sector {st.session_state.get('sector','')}; "
                           f"revenue KES {st.session_state.get('revenue',0):,.0f}; expenses KES {st.session_state.get('expenses',0):,.0f}.")
            messages = [{"role": "system", "content": system}] + \
                       [{"role": e["role"], "content": e["api"]} for e in st.session_state["chat"][-10:]]
            model = VISION_MODEL if att_image_url else TEXT_MODEL

            # Render user bubble immediately
            with chat_area:
                render_bubble("user", user_text, [att_image_url] if att_image_url else [])
                placeholder = st.empty()

            # Stream AI response
            full = ""
            try:
                for chunk in stream_chat(client, model, messages):
                    full += chunk
                    with chat_area:
                        placeholder.markdown(
                            f'<div class="bz-row"><div class="bz-bubble bz-ai">{md_to_html(full)}</div></div>',
                            unsafe_allow_html=True)
                st.session_state["chat"].append({"role": "assistant", "display": full, "img_urls": [], "api": full})
                save_current_chat_to_history()
            except Exception as e:
                with chat_area:
                    placeholder.markdown(
                        f'<div class="bz-row"><div class="bz-bubble bz-ai">⚠️ Sorry, I hit a connection problem: {e}</div></div>',
                        unsafe_allow_html=True)
            st.rerun()

# ============================================================================
# SECTION 15: PAGE 5 — SETTINGS (Model Info & Data Management)
# ============================================================================
elif page == "⚙️ Settings":
    st.markdown('<div class="eyebrow">System</div><h2 style="margin-top:0;">⚙️ Settings & About</h2>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        if resources:
            engine, base_engine, card = resources
            m = card.get("metrics_test", {})
            st.markdown('<div class="card"><b>🤖 Model & Validation (v2.1)</b><br>'
                        f'{card.get("model", "")}<br>'
                        f'ROC-AUC {m.get("roc_auc", "-")} (CI {m.get("roc_auc_ci95", ["-","-"])}) · '
                        f'Precision {m.get("precision", "-")} · Recall {m.get("recall", "-")} · '
                        f'F1 {m.get("f1", "-")} · Brier {m.get("brier", "-")}<br>'
                        f'Operating threshold (max-F1): {card.get("operating_threshold_max_f1", "-")}<br>'
                        f'<span class="muted">{card.get("leakage_policy", "")}</span></div>',
                        unsafe_allow_html=True)
        st.markdown('<div class="card"><b>🔐 AI Service</b><br>'
                    + ("✅ Configured by the development team." if client
                       else "⚠️ Not configured (developer: set NVIDIA_API_KEY in the environment or .streamlit/secrets.toml).")
                    + '</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><b>📱 About BizFlow360</b><br>'
                    'An early-warning system for Kenyan MSMEs by Team Maridex. '
                    'Works on phone, tablet and desktop.</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><b>🗄️ Your Data</b><br>'
                    'Analysis and chat history are stored only on this device.</div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ Clear All History (Analysis & Chat)", type="secondary"):
        if HISTORY_FILE.exists(): HISTORY_FILE.unlink()
        if CHAT_SESSIONS_FILE.exists(): CHAT_SESSIONS_FILE.unlink()
        st.success("All history cleared!")
        st.rerun()