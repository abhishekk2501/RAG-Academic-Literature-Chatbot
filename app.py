import tempfile
from pathlib import Path

import streamlit as st

from rag_engine import RAGEngine  # unchanged: your RAG logic stays exactly as it is

st.set_page_config(page_title="PDF Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
:root{--ac:#4f5bd5;--line:rgba(128,128,128,.28);--soft:rgba(128,128,128,.09);--hl:#f5b942}
html,body,[class*="css"],.stMarkdown,button,input,textarea{font-family:'Instrument Sans',system-ui,sans-serif!important}
h1,h2,h3,.hero h1{font-family:'Newsreader',Georgia,serif!important;font-weight:500!important;letter-spacing:-.01em}
.stApp{background-image:
 radial-gradient(900px 520px at 8% -5%,rgba(79,91,213,.20),transparent 60%),
 radial-gradient(800px 480px at 98% 2%,rgba(20,184,166,.15),transparent 60%),
 radial-gradient(900px 520px at 100% 100%,rgba(245,185,66,.14),transparent 60%),
 radial-gradient(800px 480px at 0% 100%,rgba(236,72,153,.10),transparent 60%);background-attachment:fixed}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:rgba(128,128,128,.07)!important;backdrop-filter:blur(14px)}
.docbar,.step{backdrop-filter:blur(8px)}
footer{visibility:hidden}
.block-container{padding-top:2.2rem;max-width:1100px}
.hero h1{font-size:clamp(2.4rem,6vw,4rem);line-height:1.05;margin:.4rem 0 .6rem}
.hero p{opacity:.7;max-width:520px;font-size:1.05rem}
.docbar{display:flex;gap:14px;align-items:center;border:1px solid var(--line);background:var(--soft);border-radius:16px;padding:14px 18px;margin:.4rem 0 1.2rem}
.docbar .ic{width:44px;height:44px;border-radius:12px;background:var(--hl);color:#4a3300;display:grid;place-items:center;font-weight:700;flex:none}
.docbar .nm{font-weight:600;word-break:break-all}.docbar .mt{opacity:.65;font-size:.88rem}
.pill{margin-left:auto;font-size:.8rem;font-weight:600;padding:3px 12px;border-radius:99px;background:rgba(24,160,100,.16);color:#1aa467}
.step{border:1px solid var(--line);border-radius:16px;padding:18px;height:100%;background:var(--soft)}
.step b{font-family:'Newsreader',serif;font-size:1.25rem;font-weight:500;display:block;margin:6px 0 2px}.step span{opacity:.7;font-size:.93rem}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:8px 14px}
.stTabs [aria-selected="true"]{color:var(--ac)!important;font-weight:600}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid var(--line);transition:transform .12s,border-color .15s}
.stButton>button:hover{border-color:var(--ac);color:var(--ac)}.stButton>button:active{transform:scale(.97)}
.stButton>button[kind="primary"]{background:var(--ac);color:#fff;border-color:var(--ac)}
[data-testid="stChatMessage"]{border:1px solid var(--line);border-radius:16px;padding:12px 16px;background:var(--soft)}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:12px;background:var(--soft)}
[data-testid="stSidebar"]{border-right:1px solid var(--line)}
.logo{font-family:'Newsreader',serif;font-size:1.5rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    return RAGEngine()


engine = get_engine()
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("results", {})


def human_size(n):
    return f"{n / 1048576:.1f} MB" if n > 1048576 else f"{max(1, round(n / 1024))} KB"


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown('<div class="logo">📚 PDF Assistant</div>', unsafe_allow_html=True)
    st.caption("Runs locally with Ollama. No paid API.")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("🧹 Clear chat", use_container_width=True):
        ss.messages = []
        st.rerun()
    st.divider()
    st.caption("Tip: use the ⋮ menu at the top right → Settings to switch light / dark theme.")

# ---------------- index the PDF (same logic as before) ----------------
if uploaded_file and ss.get("loaded_file") != uploaded_file.name:
    with st.spinner("Reading and indexing PDF..."):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            pdf_path = tmp.name
        try:
            ss.chunk_count = engine.index_pdf(pdf_path)
            ss.loaded_file = uploaded_file.name
            ss.messages, ss.results = [], {}
        except Exception as e:
            st.error(f"Could not process the PDF: {e}")


def ask(prompt):
    """Calls your existing engine.ask(); returns (answer, sources)."""
    try:
        with st.spinner("Searching the PDF and generating an answer..."):
            return engine.ask(prompt)
    except Exception as e:
        st.error(f"Error: {e}\n\nMake sure Ollama is running and the model is installed.")
        return None, []


def show_sources(sources):
    if sources:
        with st.expander("Retrieved sources"):
            for i, s in enumerate(sources, 1):
                st.markdown(f"**Source {i}, page {s['page']}**\n\n{s['text'][:700]}...")


def tool(key, label, prompt, collapsible=False):
    """One-click tool: runs once, caches the result for this document."""
    if st.button(label, key=f"btn_{key}", type="primary"):
        ans, src = ask(prompt)
        if ans:
            ss.results[key] = (ans, src)
    if key in ss.results:
        ans, src = ss.results[key]
        if collapsible:
            with st.expander("Result", expanded=True):
                st.markdown(ans)
        else:
            st.markdown(ans)
        st.download_button("⬇️ Download as text", ans, file_name=f"{key}.txt", key=f"dl_{key}")
        show_sources(src)


# ---------------- landing ----------------
if not (uploaded_file and ss.get("loaded_file") == uploaded_file.name):
    st.markdown('<div class="hero"><h1>Your documents, understood.</h1>'
                '<p>Upload a PDF in the sidebar to chat with it, summarize it, or turn it into study notes.</p></div>',
                unsafe_allow_html=True)
    c = st.columns(3)
    steps = [("⬆️", "Upload", "Drop a text-based PDF into the sidebar."),
             ("⚙️", "Index", "It is read and indexed locally on your machine."),
             ("💬", "Explore", "Ask questions, summarize, or make study notes.")]
    for col, (e, t, d) in zip(c, steps):
        col.markdown(f'<div class="step">{e}<b>{t}</b><span>{d}</span></div>', unsafe_allow_html=True)
    st.caption("Privacy: your PDF and questions are processed locally by this application.")
    st.stop()

# ---------------- workspace ----------------
st.markdown(
    f'<div class="docbar"><div class="ic">PDF</div><div><div class="nm">{uploaded_file.name}</div>'
    f'<div class="mt">{human_size(uploaded_file.size)} · {ss.get("chunk_count", "?")} passages indexed</div></div>'
    f'<span class="pill">Ready</span></div>', unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat", "📝 Summarize", "📒 Study Notes"])

with tabs[0]:
    if not ss.messages:
        st.info("Ask a question below. Answers come from the PDF and show the pages they used.")
    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_sources(m.get("sources"))
    q = st.chat_input("Ask a question about the PDF...")
    if q:
        ss.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            ans, src = ask(q)
            if ans:
                st.markdown(ans)
                show_sources(src)
                ss.messages.append({"role": "assistant", "content": ans, "sources": src})

with tabs[1]:
    modes = {"Concise": "Write a concise summary of the document in one short paragraph.",
             "Detailed": "Write a detailed summary of the document with headings and short paragraphs.",
             "By section": "Summarize each major section or chapter of the document separately under its own heading.",
             "Takeaways": "List the 6 to 10 key takeaways of the document as bullet points."}
    mode = st.radio("Style", list(modes), horizontal=True, label_visibility="collapsed")
    tool(f"summary_{mode}", f"Generate {mode.lower()} summary", modes[mode])

with tabs[2]:
    tool("notes", "Create study notes",
         "Convert the document into structured study notes in markdown with headings, definitions, important concepts, examples and a short revision checklist.")