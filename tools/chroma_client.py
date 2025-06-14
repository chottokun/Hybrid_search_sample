import streamlit as st
import chromadb

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="chroma_data")
