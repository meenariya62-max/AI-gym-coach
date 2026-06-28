import os
import streamlit as st
import base64
 
 
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
 
def apply_theme(theme: str):
    """Apply light/dark theme via JS + CSS override."""
    # JS sets data-theme attribute on <html>
    st.markdown(
        f"""<script>
        (function applyTheme() {{
            var t = '{theme}';
            document.documentElement.setAttribute('data-theme', t);
            // retry after Streamlit re-renders
            [100, 300, 700, 1500].forEach(function(d) {{
                setTimeout(function() {{
                    document.documentElement.setAttribute('data-theme', t);
                }}, d);
            }});
        }})();
        </script>""",
        unsafe_allow_html=True,
    )
 
    # CSS fallback — force background/color for Streamlit's own elements
    if theme == "light":
        st.markdown(
            """<style>
            .stApp { background-color: #F0F2FA !important; color: #1A1D2E !important; }
            [data-testid="stSidebar"] { background-color: #FFFFFF !important; }
            [data-testid="stSidebar"] * { color: #1A1D2E !important; }
            .stMarkdown, .stMarkdown p, .stMarkdown span { color: #1A1D2E !important; }
            [data-testid="stMetric"] { background: #FFFFFF !important; border-color: rgba(0,0,0,0.10) !important; }
            [data-testid="stMetricValue"] { color: #1A1D2E !important; }
            [data-testid="stMetricLabel"] { color: rgba(26,29,46,0.55) !important; }
            input, textarea { background-color: #F0F2FA !important; color: #1A1D2E !important; }
            [data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #1A1D2E !important; }
            [data-baseweb="menu"], [data-baseweb="menu"] li { background-color: #FFFFFF !important; color: #1A1D2E !important; }
            button, .stButton button { background: #FFFFFF !important; color: #1A1D2E !important; border-color: rgba(0,0,0,0.10) !important; }
            [data-testid="stExpander"] { background: #FFFFFF !important; }
            .stDataFrame, .stDataFrame * { background-color: #FFFFFF !important; color: #1A1D2E !important; }
            h1,h2,h3,h4,h5,h6,p,span,label,div { color: #1A1D2E !important; }
            </style>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<style>
            .stApp { background-color: #0A0D14 !important; color: #FFFFFF !important; }
            [data-testid="stSidebar"] { background-color: #111520 !important; }
            </style>""",
            unsafe_allow_html=True,
        )
 
 
def inject_local_font(font_path, font_name):
    if not os.path.exists(font_path):
        return
    with open(font_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(font_path)[1].lstrip(".")
    fmt = {"otf": "opentype"}.get(ext, ext)
    mime = {"otf": "font/otf"}.get(ext, f"font/{ext}")
    st.markdown(f"""<style>
        @font-face {{
            font-family: '{font_name}';
            src: url('data:{mime};base64,{encoded}') format('{fmt}');
            font-weight: 100 900;
            font-style: normal;
        }}
        </style>""", unsafe_allow_html=True)
 
 
def inject_webrtc_styles():
    st.markdown(
        """<style>
        iframe[title="streamlit_component"] {{
            pointer-events: none !important;
            border-radius: 12px !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )
 

