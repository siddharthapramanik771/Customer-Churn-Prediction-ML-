import streamlit as st


GITHUB_REPOSITORY_URL = (
    "https://github.com/siddharthapramanik771/Customer-Churn-Prediction-ML-"
)


def apply_page_styles() -> None:
    st.markdown(
        """
<style>
    :root {
        --app-panel: rgba(127, 127, 127, 0.10);
        --app-panel-hover: rgba(127, 127, 127, 0.16);
        --app-line: rgba(127, 127, 127, 0.24);
        --app-shadow: rgba(0, 0, 0, 0.10);
        --app-accent: #0f9f8f;
        --app-accent-strong: #0a6f86;
        --sidebar-bg-start: #102033;
        --sidebar-bg-end: #17374b;
        --sidebar-ink: #f8fbfc;
        --sidebar-muted: #d8e7ef;
        --button-bg: #ffffff;
        --button-ink: #102033;
        --tab-active-bg: #10334a;
        --tab-active-ink: #ffffff;
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
        color: inherit;
    }

    .status-tile {
        padding: 1rem 1.05rem;
    }

    .status-tile span {
        display: block;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        opacity: 0.72;
        text-transform: uppercase;
    }

    .status-tile strong {
        color: inherit;
        display: block;
        font-size: 1.35rem;
        line-height: 1.2;
    }

    .status-tile small {
        display: block;
        margin-top: 0.3rem;
        opacity: 0.68;
    }

    div[data-testid="stMetric"] {
        padding: 0.9rem 1rem;
    }

    div[data-testid="stMetric"] *,
    div[data-testid="stMetricValue"] * {
        color: inherit !important;
    }

    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] label {
        opacity: 0.72;
    }

    div[data-testid="stForm"] {
        padding: 1.15rem;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    .model-structure-diagram {
        align-items: stretch;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.8rem 0 1.1rem;
    }

    .model-structure-node {
        background: var(--app-panel);
        border: 1px solid var(--app-line);
        border-radius: 8px;
        box-shadow: 0 8px 22px var(--app-shadow);
        flex: 1 1 145px;
        min-width: 0;
        padding: 0.85rem;
    }

    .model-structure-node strong,
    .model-structure-node span {
        display: block;
    }

    .model-structure-node strong {
        font-size: 0.92rem;
        line-height: 1.25;
    }

    .model-structure-node span {
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 0.28rem;
        opacity: 0.72;
    }

    .model-structure-arrow {
        align-items: center;
        display: flex;
        font-weight: 800;
        justify-content: center;
        min-width: 1.15rem;
        opacity: 0.62;
    }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 0 !important;
        gap: 0.45rem;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--app-panel);
        border: 1px solid var(--app-line);
        border-radius: 8px;
        color: inherit !important;
        font-weight: 700;
        padding: 0.65rem 1rem;
    }

    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
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

        .model-structure-node {
            flex-basis: 100%;
        }

        .model-structure-arrow {
            display: none;
        }
    }
</style>
""",
        unsafe_allow_html=True,
    )
