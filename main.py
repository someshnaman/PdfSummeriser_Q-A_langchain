import streamlit as st
from src.streamlit_app import summerise
from streamlit_extras.add_vertical_space import add_vertical_space

st.title("Research Paper Summarizer")

with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown("""
    ## Hey This Somesh ! Connect me on Linkedin
    - [Linkedin](https://www.linkedin.com/in/somesh-naman/) 
    """)
    add_vertical_space(5)
    st.write('Made with ❤️ by [Somesh](https://github.com/someshnaman)')
st.markdown(
    """
    ## Summarizer

    This is a summarizer for research papers."""
)
if __name__ == "__main__":
    summerise()
