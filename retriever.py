
import streamlit as st

import requests
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

PAGES = [
    "Intelligence_artificielle_générative",
    "Transformeur_génératif_préentraîné",
    "Google_Gemini",
    "Grand_modèle_de_langage",
    "ChatGPT",
    "LLaMA",
    "Réseaux_antagonistes_génératifs",
    "Apprentissage_auto-supervisé",
    "Apprentissage_par_renforcement",
    "DALL-E",
    "Midjourney",
    "Stable_Diffusion"
]


def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://fr.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAG_project/0.0.1 (contact@datascientist.fr)"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


def get_documents():

    # Récupère les documents
    docs = []
    for page_title in PAGES:
        content = get_wikipedia_page(page_title)
        docs.append(Document(
            page_content=content,
            metadata={
                "title": page_title,
                "url": f"https://fr.wikipedia.org/wiki/{page_title}",
                "source": "Wikipedia"
            }
        ))

    # Découpage en chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=250
    )
    return text_splitter.split_documents(docs)


def get_retriever():

    # Construction du vector store
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=OpenAIEmbeddings()
    )

    # Check if the vector store is empty
    if len(vectorstore.get(limit=1)['ids']) == 0:
        with st.spinner('Loading documents...'):
            docs = get_documents()
            vectorstore.add_documents(docs)

    # Document retriever
    return vectorstore.as_retriever(top_k=10)
