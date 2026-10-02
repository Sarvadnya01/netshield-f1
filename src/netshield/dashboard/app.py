"""NetShield-FL Streamlit Dashboard — entry point with page navigation."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="NetShield-FL",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page imports (deferred to avoid circular)
from netshield.dashboard.pages import live_soc, federated_lab, experiments, architecture  # noqa: E402

PAGES = {
    "Live SOC": live_soc,
    "Federated Lab": federated_lab,
    "Experiments": experiments,
    "Architecture": architecture,
}

st.sidebar.title("NetShield-FL")
page = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

PAGES[page].render()
