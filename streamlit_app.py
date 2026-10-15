from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAIError

from banking_assistant.emi import calculate_emi
from banking_assistant.rag import answer_question, build_knowledge_base


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
st.set_page_config(
    page_title="Bankwise | AI Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #142e35;
        --muted: #657b7c;
        --teal: #11786e;
        --mint: #e5f4ef;
        --line: #e5edeb;
        --paper: #f5f8f6;
        --white: #ffffff;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 75% 0%, rgba(214, 239, 229, .52), transparent 34%),
            var(--paper);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1260px;
        padding-top: 2.1rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: #12363a;
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #f2faf7; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #bed3ce;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: #1d4548;
        border: 1px solid #426568;
        border-radius: 10px;
        color: white;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: #1d4548;
        border: 1px dashed #62817d;
        border-radius: 12px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        color: #173c3d;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        border-color: #527271;
        color: white;
        background: #1d4548;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 11px;
        margin: 0 0 2.3rem 0;
        color: #ffffff;
        font-family: 'Manrope', sans-serif;
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .brand-mark {
        display: grid;
        width: 39px;
        height: 39px;
        place-items: center;
        border-radius: 13px;
        background: #b5dfcf;
        color: #12363a;
        font-size: 1.25rem;
    }
    .sidebar-label {
        margin: 1.5rem 0 .35rem;
        color: #82aaa3;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: clamp(2rem, 5vw, 4.3rem);
        border: 1px solid rgba(216, 232, 225, .85);
        border-radius: 28px;
        background:
            radial-gradient(circle at 83% 34%, rgba(166, 221, 197, .27), transparent 22%),
            linear-gradient(120deg, #ffffff 0%, #f6fbf8 57%, #e9f5ef 100%);
        box-shadow: 0 18px 55px rgba(25, 64, 55, .055);
    }
    .hero:after {
        position: absolute;
        top: -84px;
        right: -36px;
        width: 310px;
        height: 310px;
        border: 1px solid rgba(17, 120, 110, .11);
        border-radius: 50%;
        content: "";
        box-shadow: 0 0 0 30px rgba(17, 120, 110, .025), 0 0 0 60px rgba(17, 120, 110, .02);
        pointer-events: none;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1.1rem;
        color: var(--teal);
        font-size: .71rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
    }
    .eyebrow-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #43a98c;
        box-shadow: 0 0 0 4px rgba(67, 169, 140, .12);
    }
    .hero h1 {
        max-width: 710px;
        margin: 0;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.5rem, 5.4vw, 4.6rem);
        font-weight: 700;
        letter-spacing: -.065em;
        line-height: 1.05;
    }
    .hero h1 span { color: var(--teal); }
    .hero-copy {
        max-width: 580px;
        margin: 1.1rem 0 1.8rem;
        color: #637a79;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .meta-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 8px 12px;
        border: 1px solid #e0ebe6;
        border-radius: 999px;
        background: rgba(255,255,255,.72);
        color: #46615f;
        font-size: .77rem;
        font-weight: 600;
    }
    .section-kicker {
        margin: 2.15rem 0 .35rem;
        color: #809491;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
    }
    .section-title {
        margin: 0 0 1rem;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -.045em;
    }
    [data-testid="stBaseButton-secondary"] {
        min-height: 47px;
        border: 1px solid #e0eae5;
        border-radius: 12px;
        background: rgba(255,255,255,.9);
        color: #29494a;
        font-weight: 600;
        transition: border-color .15s ease, transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        border-color: #8bc6b0;
        color: var(--teal);
        transform: translateY(-1px);
        box-shadow: 0 7px 20px rgba(35, 91, 73, .08);
    }
    [data-testid="stBaseButton-primary"] {
        min-height: 46px;
        border: 0;
        border-radius: 11px;
        background: #b9e1d1;
        color: #143b3c;
        font-weight: 700;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background: #c8eadc;
        color: #143b3c;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 18px;
        background: #ffffff;
        box-shadow: 0 8px 26px rgba(24, 62, 53, .035);
    }
    [data-testid="stChatInput"] {
        border: 1px solid #dce9e3;
        border-radius: 17px;
        background: #ffffff;
        box-shadow: 0 8px 30px rgba(24, 62, 53, .07);
    }
    [data-testid="stChatInput"] textarea { color: var(--ink); }
    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(255,255,255,.78);
    }
    [data-testid="stMetric"] {
        padding: 1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: white;
    }
    .helper-note {
        margin: .55rem 0 0;
        color: #81918e;
        font-size: .78rem;
        line-height: 1.55;
    }
    .footer-note {
        margin-top: 2.2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--line);
        color: #879592;
        font-size: .76rem;
        text-align: center;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 3rem; }
        .hero { border-radius: 21px; }
        .hero:after { right: -160px; }
        .hero-copy { font-size: .94rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><span class="brand-mark">b</span> bankwise</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">Your knowledge base</div>', unsafe_allow_html=True)
    st.markdown("Add approved bank documents to give your assistant trusted answers.")

    api_key_override = st.text_input(
        "OpenAI API key override",
        value="",
        type="password",
        placeholder="Leave blank to use OPENAI_API_KEY",
        help="Optional override. By default, the app reads OPENAI_API_KEY from its environment or .env file.",
    )
    uploads = st.file_uploader(
        "Policy documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload approved, non-sensitive policy PDFs.",
    )
    if st.button("Build secure index", type="primary", use_container_width=True):
        if not uploads:
            st.error("Upload at least one bank policy PDF before indexing.")
        else:
            try:
                with st.spinner("Extracting pages and building your index..."):
                    index, chunk_count = build_knowledge_base(uploads, api_key_override.strip() or None)
                st.session_state.vectorstore = index
                st.session_state.indexed_files = [upload.name for upload in uploads]
                st.session_state.indexed_chunks = chunk_count
                st.session_state.chat_history = []
                st.success(f"Ready — {chunk_count} sections indexed.")
            except ValueError as error:
                st.error(str(error))
            except OpenAIError as error:
                st.error(
                    f"OpenAI could not create document embeddings ({type(error).__name__}). "
                    "Check the API key, account access, and network connection."
                )

    if st.session_state.get("indexed_files"):
        st.markdown('<div class="sidebar-label">Indexed documents</div>', unsafe_allow_html=True)
        for document_name in st.session_state.indexed_files:
            st.markdown(f"📄 {document_name}")
        st.caption(f"{st.session_state.get('indexed_chunks', 0)} searchable sections")
        st.success("Knowledge base is ready")
    else:
        st.markdown('<div class="sidebar-label">Index status</div>', unsafe_allow_html=True)
        st.info("Upload your bank PDFs and build an index to get started.")

    st.markdown('<div class="sidebar-label">Privacy first</div>', unsafe_allow_html=True)
    st.markdown(
        "Only upload documents approved for AI processing. PDF text is sent to "
        "the configured OpenAI API for embeddings and answers."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow"><span class="eyebrow-dot"></span> YOUR BANKING, MADE CLEARER</div>
      <h1>Good answers.<br><span>Grounded in your bank.</span></h1>
      <p class="hero-copy">
        Ask about loans, cards, and everyday banking. Get clear answers drawn
        from the official documents you choose to upload.
      </p>
      <div class="hero-meta">
        <span class="meta-pill">✦ Document-grounded</span>
        <span class="meta-pill">⌕ Page-level sources</span>
        <span class="meta-pill">◈ Private, in-memory index</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">LET’S GET STARTED</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What can I help you with?</div>', unsafe_allow_html=True)

suggestions = [
    ("⌂", "Home loan documents", "What documents do I need for a home loan?"),
    ("₹", "Understand your EMI", "Explain how an EMI works."),
    ("↗", "Missed a payment", "What happens if I miss an EMI?"),
]
suggestion_columns = st.columns(3)
for column, (icon, title, query) in zip(suggestion_columns, suggestions):
    with column:
        if st.button(f"{icon}  {title}  →", key=f"suggest-{title}", use_container_width=True):
            st.session_state.pending_question = query
            st.rerun()

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("📚 Sources from your documents"):
                for source in turn["sources"]:
                    st.markdown(f"- {source['label']}")

question = st.chat_input("Ask a question about your bank documents...")
if question is None:
    question = st.session_state.pop("pending_question", None)

if question:
    if not st.session_state.get("vectorstore"):
        st.warning("Upload your official bank PDFs and build an index before asking questions.")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="✦"):
            try:
                with st.spinner("Finding the right information in your documents..."):
                    answer, sources = answer_question(
                        question,
                        st.session_state.chat_history,
                        st.session_state.vectorstore,
                        api_key_override.strip() or None,
                    )
                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources from your documents"):
                        for source in sources:
                            st.markdown(f"- {source['label']}")
            except ValueError as error:
                st.error(str(error))
            except OpenAIError as error:
                st.error(
                    f"OpenAI could not answer the question ({type(error).__name__}). "
                    "Check the API key, account access, and network connection."
                )
            else:
                st.session_state.chat_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer, "sources": sources},
                ])

st.markdown('<div class="section-kicker">PLAN WITH CONFIDENCE</div>', unsafe_allow_html=True)
with st.expander("₹  Estimate a monthly loan payment"):
    st.markdown("### EMI calculator")
    st.caption("A simple arithmetic estimate — not a bank quote or loan approval.")
    col1, col2, col3 = st.columns(3)
    principal = col1.number_input(
        "Loan amount (INR)", min_value=1.0, value=1_000_000.0, step=10_000.0, format="%.0f"
    )
    annual_rate = col2.number_input(
        "Annual interest rate (%)", min_value=0.0, value=9.0, step=0.1, format="%.1f"
    )
    tenure_months = col3.number_input("Loan tenure (months)", min_value=1, value=60, step=1)
    emi = calculate_emi(principal, annual_rate, tenure_months)
    st.metric("Estimated monthly EMI", f"₹{emi:,.2f}")

st.markdown(
    '<div class="footer-note">AI answers are based on retrieved document passages and may not reflect your personal account. '
    'Confirm important decisions with your bank. &nbsp;·&nbsp; For demonstration purposes.</div>',
    unsafe_allow_html=True,
)
