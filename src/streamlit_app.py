import streamlit as st
import langchain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np


# import pdfminer.six


def summerise():
    user_input_pdf = st.file_uploader("Upload your research paper:", type='pdf')
    if user_input_pdf is not None:
        pdf_file = PdfReader(user_input_pdf)
        text = ""
        for page in pdf_file.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=200,
                                                       length_function=len
                                                       )
        chunks = text_splitter.split_text(text)
        st.write(chunks)
