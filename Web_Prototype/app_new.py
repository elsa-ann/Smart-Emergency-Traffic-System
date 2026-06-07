import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="SECS — Smart Emergency Corridor System",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        section[data-testid="stSidebar"] {display:none;}
        .css-1lcbmhc.e1fqkh3o3 {padding-top: 0rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

html_path = Path(__file__).with_name('app.html')
if html_path.exists():
    html_content = html_path.read_text(encoding='utf-8')
    components.html(html_content, width=1200, height=920, scrolling=True)
else:
    st.error(f'File not found: {html_path}')
