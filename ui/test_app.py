"""
Test Streamlit App - Simplified version to test rendering
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Quasar Test",
    page_icon="🔭",
    layout="wide"
)

# Simple CSS to test
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom, #0a0e27 0%, #1a1e3a 100%);
    }
    
    /* Ensure content is visible */
    .main .block-container {
        position: relative;
        z-index: 10;
        padding-top: 3rem;
        color: white !important;
    }
    
    h1, h2, h3, p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 QUASAR TEST")
st.header("Radio Astronomy Intelligence System")

st.write("If you can see this text, the app is working!")

# Test columns
col1, col2, col3 = st.columns(3)

with col1:
    st.button("Test Button 1")
    
with col2:
    st.button("Test Button 2")
    
with col3:
    st.button("Test Button 3")

# Test chat
st.subheader("Chat Test")
prompt = st.chat_input("Type a message...")
if prompt:
    st.write(f"You said: {prompt}")
