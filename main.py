import streamlit as st
from src.streamlit_app import summerise
from streamlit_extras.add_vertical_space import add_vertical_space


st.markdown("#### Pdf Summarizer & Q&A #################################")

with st.sidebar:
    st.title('🤗💬 LLM Q&A App')
    st.markdown("""
    ## Hey This Somesh ! Connect me on Linkedin
    - [Linkedin](https://www.linkedin.com/in/somesh-naman/) 
    """)
    add_vertical_space(5)
    st.write('Made with ❤️ by [Somesh](https://github.com/someshnaman)')
st.markdown("""
    

    This is a summarizer for pdf."""
)
if __name__ == "__main__":
    summerise()
