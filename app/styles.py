import streamlit as st


GITHUB_REPOSITORY_URL = (
    "https://github.com/siddharthapramanik771/Customer-Churn-Prediction-ML-"
)


def apply_page_styles() -> None:
    st.markdown(
        """
<style>
    :root {
        --ink: #172033;
        --muted: #5c667a;
        --line: rgba(23, 32, 51, 0.12);
        --panel: rgba(255, 255, 255, 0.88);
        --accent: #0f9f8f;
        --accent-strong: #0a6f86;
        --warm: #f4b860;
    }

    .stApp {
        background:
            radial-gradient(circle at 16% 18%, rgba(15, 159, 143, 0.13), transparent 28%),
            radial-gradient(circle at 86% 10%, rgba(244, 184, 96, 0.16), transparent 26%),
            linear-gradient(180deg, #f7fafc 0%, #eef4f6 48%, #f8faf9 100%);
        color: var(--ink);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102033 0%, #17374b 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f8fbfc;
    }

    [data-testid="stSidebar"] [data-testid="stDataFrame"] {
        background: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.22);
    }

    [data-testid="stSidebar"] .stDownloadButton > button {
        background: #ffffff;
        color: #102033;
        width: 100%;
    }

    [data-testid="stSidebar"] .stDownloadButton > button:hover {
        color: #102033;
    }

    .hero {
        border: 1px solid rgba(255, 255, 255, 0.7);
        border-radius: 8px;
        padding: 2.15rem 2.35rem;
        background:
            linear-gradient(135deg, rgba(18, 34, 53, 0.96), rgba(10, 111, 134, 0.9)),
            linear-gradient(135deg, #102033, #0f9f8f);
        box-shadow: 0 18px 45px rgba(23, 32, 51, 0.16);
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

    .status-tile {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 10px 28px rgba(23, 32, 51, 0.08);
        padding: 1rem 1.05rem;
    }

    .status-tile span {
        color: var(--muted);
        display: block;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }

    .status-tile strong {
        color: var(--ink);
        display: block;
        font-size: 1.35rem;
        line-height: 1.2;
    }

    .status-tile small {
        color: var(--muted);
        display: block;
        margin-top: 0.3rem;
    }

    div[data-testid="stMetric"],
    div[data-testid="stForm"],
    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid var(--line);
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 22px rgba(23, 32, 51, 0.06);
    }

    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        padding: 1.15rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid var(--line);
        border-radius: 8px;
        color: var(--ink);
        font-weight: 700;
        padding: 0.65rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: #10334a;
        color: #ffffff;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        border: 0;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 800;
        min-height: 2.85rem;
        box-shadow: 0 10px 22px rgba(10, 111, 134, 0.24);
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border: 0;
        color: #ffffff;
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
