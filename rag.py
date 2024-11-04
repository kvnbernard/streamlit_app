
from typing import List
from pydantic import BaseModel, Field
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_variant_queries(original_query):
    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    class QueriesStructure(BaseModel):
        queries: List[str] = Field(
            ...,
            description="List of queries to generate different perspectives on the original query."
        )

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = model.with_structured_output(QueriesStructure)

    generate_queries = (
        prompt_perspectives
        | structured_llm
    )

    return generate_queries.invoke(original_query).queries


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """

    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    sorted_docs = sorted(fused_scores.items(),
                         key=lambda x: x[1], reverse=True)
    return [loads(doc) for doc, _ in sorted_docs]


def format_documents(docs):
    result = ""
    for doc in docs:
        result += "---\n"
        result += f"Lien source : {doc.metadata['url']}\n"
        result += f"Contenu : {doc.page_content}\n"
        result += "---\n"
    return result


def rag_stream(query, retriever):
    # Prompt
    template = """Réponds à la question en utilisant uniquement le contexte donné ci-dessous.
    Indique à la fin de ta réponse, les liens vers les documents utilisés.
    \"\"\"
    {context}
    \"\"\"
    
    Exemple de réponse :
    \"\"\"
    Gemini est un LLM (grand modèle de langage) car il utilise le réseau de neurones du modèle PaLM 2 et l'architecture « Google Transformer », qui sont des caractéristiques typiques des grands modèles de langage. Ces modèles possèdent un grand nombre de paramètres et sont capables de traiter et de générer du texte, ainsi que d'autres types de données, ce qui est le cas de Gemini. De plus, Gemini est capable de générer et de combiner des objets sonores, visuels et textuels, ce qui le rapproche d'une intelligence artificielle générale. 

    Sources : 
    - https://fr.wikipedia.org/wiki/Google_Gemini
    \"\"\"
    
    Question: {query}
    """

    def retrieve_documents(query):

        retrieval_chain_rag_fusion = (
            get_variant_queries |
            retriever.map() |
            reciprocal_rank_fusion
        )

        docs = retrieval_chain_rag_fusion.invoke(query)
        return docs[:5]

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-4o", temperature=0)
    rag_chain = (
        {
            "context": RunnablePassthrough() | retrieve_documents | format_documents,
            "query": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    for chunk in rag_chain.stream(query):
        yield chunk
