import streamlit as st


GITHUB_REPOSITORY_URL = (
    "https://github.com/siddharthapramanik771/Customer-Churn-Prediction-ML-"
)


def apply_page_styles() -> None:
    st.markdown(
        """
<style>
    :root {
        color-scheme: light dark;
        --app-bg-start: #f7fafc;
        --app-bg-mid: #eef4f6;
        --app-bg-end: #f8faf9;
        --app-ink: #172033;
        --app-muted: #5c667a;
        --app-line: rgba(23, 32, 51, 0.12);
        --app-panel: rgba(255, 255, 255, 0.88);
        --app-panel-solid: #ffffff;
        --app-shadow: rgba(23, 32, 51, 0.08);
        --app-accent: #0f9f8f;
        --app-accent-strong: #0a6f86;
        --app-warm: #f4b860;
        --sidebar-bg-start: #102033;
        --sidebar-bg-end: #17374b;
        --sidebar-ink: #f8fbfc;
        --sidebar-muted: #d8e7ef;
        --button-bg: #ffffff;
        --button-ink: #102033;
        --tab-active-bg: #10334a;
        --tab-active-ink: #ffffff;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --app-bg-start: #0f1722;
            --app-bg-mid: #101d28;
            --app-bg-end: #0b111a;
            --app-ink: #edf4f8;
            --app-muted: #b8c5cf;
            --app-line: rgba(237, 244, 248, 0.16);
            --app-panel: rgba(21, 34, 47, 0.9);
            --app-panel-solid: #15222f;
            --app-shadow: rgba(0, 0, 0, 0.28);
            --sidebar-bg-start: #08131f;
            --sidebar-bg-end: #10283a;
            --sidebar-ink: #f8fbfc;
            --sidebar-muted: #cfe0e8;
            --button-bg: #f8fbfc;
            --button-ink: #102033;
            --tab-active-bg: #0f9f8f;
            --tab-active-ink: #06161b;
        }
    }

    html[data-theme="dark"],
    body[data-theme="dark"],
    [data-base-theme="dark"] {
        --app-bg-start: #0f1722;
        --app-bg-mid: #101d28;
        --app-bg-end: #0b111a;
        --app-ink: #edf4f8;
        --app-muted: #b8c5cf;
        --app-line: rgba(237, 244, 248, 0.16);
        --app-panel: rgba(21, 34, 47, 0.9);
        --app-panel-solid: #15222f;
        --app-shadow: rgba(0, 0, 0, 0.28);
        --sidebar-bg-start: #08131f;
        --sidebar-bg-end: #10283a;
        --sidebar-ink: #f8fbfc;
        --sidebar-muted: #cfe0e8;
        --button-bg: #f8fbfc;
        --button-ink: #102033;
        --tab-active-bg: #0f9f8f;
        --tab-active-ink: #06161b;
    }

    .stApp {
        background:
            radial-gradient(circle at 16% 18%, rgba(15, 159, 143, 0.13), transparent 28%),
            radial-gradient(circle at 86% 10%, rgba(244, 184, 96, 0.16), transparent 26%),
            linear-gradient(180deg, var(--app-bg-start) 0%, var(--app-bg-mid) 48%, var(--app-bg-end) 100%);
        color: var(--app-ink);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar-bg-start) 0%, var(--sidebar-bg-end) 100%);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: var(--sidebar-ink);
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    [data-testid="stSidebar"] small {
        color: var(--sidebar-muted);
    }

    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button,
    [data-testid="stSidebar"] [data-testid="stLinkButton"] a,
    [data-testid="stSidebar"] .stButton > a,
    [data-testid="stSidebar"] .stButton > button {
        background: var(--button-bg);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: var(--button-ink) !important;
        width: 100%;
    }

    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button *,
    [data-testid="stSidebar"] [data-testid="stLinkButton"] a *,
    [data-testid="stSidebar"] .stButton > a *,
    [data-testid="stSidebar"] .stButton > button * {
        color: var(--button-ink) !important;
    }

    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover,
    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:focus,
    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:active,
    [data-testid="stSidebar"] [data-testid="stLinkButton"] a:hover,
    [data-testid="stSidebar"] [data-testid="stLinkButton"] a:focus,
    [data-testid="stSidebar"] [data-testid="stLinkButton"] a:active,
    [data-testid="stSidebar"] .stButton > a:hover,
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--button-bg);
        color: var(--button-ink) !important;
    }

    .hero {
        border: 1px solid rgba(255, 255, 255, 0.26);
        border-radius: 8px;
        padding: 2.15rem 2.35rem;
        background:
            linear-gradient(135deg, rgba(18, 34, 53, 0.96), rgba(10, 111, 134, 0.9)),
            linear-gradient(135deg, #102033, #0f9f8f);
        box-shadow: 0 18px 45px var(--app-shadow);
        color: #ffffff;
        margin-bottom: 1.1rem;
    }

    .hero__eyebrow {
        color: #bceee7;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(2.1rem, 4vw, 4.1rem);
        line-height: 1.02;
        margin: 0 0 0.85rem;
    }

    .hero p {
        color: rgba(255, 255, 255, 0.86);
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 760px;
        margin: 0 0 1.25rem;
    }

    .hero__actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: center;
    }

    .hero__link {
        background: #ffffff;
        border-radius: 8px;
        color: #102033 !important;
        display: inline-flex;
        font-weight: 800;
        padding: 0.72rem 1rem;
        text-decoration: none;
    }

    .hero__note {
        color: #d8faf5;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .status-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.9rem 0 1.35rem;
    }

    .status-tile,
    div[data-testid="stMetric"],
    div[data-testid="stForm"] {
        background: var(--app-panel);
        border: 1px solid var(--app-line);
        border-radius: 8px;
        box-shadow: 0 10px 28px var(--app-shadow);
    }

    .status-tile {
        padding: 1rem 1.05rem;
    }

    .status-tile span {
        color: var(--app-muted);
        display: block;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }

    .status-tile strong {
        color: var(--app-ink);
        display: block;
        font-size: 1.35rem;
        line-height: 1.2;
    }

    .status-tile small {
        color: var(--app-muted);
        display: block;
        margin-top: 0.3rem;
    }

    div[data-testid="stMetric"] {
        padding: 0.9rem 1rem;
    }

    div[data-testid="stMetric"] *,
    div[data-testid="stMetricValue"] * {
        color: var(--app-ink) !important;
    }

    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] label {
        color: var(--app-muted) !important;
    }

    div[data-testid="stForm"] {
        padding: 1.15rem;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    div[data-baseweb="select"] *,
    div[data-testid="stNumberInput"] *,
    div[data-testid="stTextInput"] *,
    textarea {
        color: var(--app-ink) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--app-panel);
        border: 1px solid var(--app-line);
        border-radius: 8px;
        color: var(--app-ink) !important;
        font-weight: 700;
        padding: 0.65rem 1rem;
    }

    .stTabs [data-baseweb="tab"] * {
        color: var(--app-ink) !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--tab-active-bg);
        color: var(--tab-active-ink) !important;
    }

    .stTabs [aria-selected="true"] * {
        color: var(--tab-active-ink) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--app-accent), var(--app-accent-strong));
        border: 0;
        border-radius: 8px;
        color: #ffffff !important;
        font-weight: 800;
        min-height: 2.85rem;
        box-shadow: 0 10px 22px rgba(10, 111, 134, 0.24);
    }

    .stButton > button *,
    .stFormSubmitButton > button * {
        color: #ffffff !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border: 0;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    @media (max-width: 780px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.45rem;
        }

        .status-strip {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
        unsafe_allow_html=True,
    )
