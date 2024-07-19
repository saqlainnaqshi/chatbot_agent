import os
from dotenv import load_dotenv
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Optional
import json
from bs4 import BeautifulSoup

def set_vector_store():

    load_dotenv()

    google_api_key = os.getenv('GOOGLE_API_KEY')

    if not google_api_key:
        raise ValueError("Google API Key not found. Please set it in the .env file.")

    os.environ['GOOGLE_API_KEY'] = google_api_key

    with open("./sample.txt") as f:
        dataa = f.read()


    # with open('./data.json', 'r') as f:
    #     data = json.load(f)

    # all_tour_descriptions = ""

    # for item in data['data']:
    #     if 'content' in item and item['content']:
    #         soup = BeautifulSoup(item['content'], 'html.parser')
    #         text = soup.get_text(separator='\n')

    #         all_tour_descriptions += f"\n\nTour Description:\n{text}" 
    #     else:
    #         print("No content found for this tour entry.")

    # print(all_tour_descriptions)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.create_documents([dataa])

    class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
        def embed_documents(self, texts: List[str], task_type: Optional[str] = None, titles: Optional[List[str]] = None, output_dimensionality: Optional[int] = None) -> List[List[float]]:
            embeddings_repeated = super().embed_documents(texts, task_type, titles, output_dimensionality)
            embeddings = [list(emb) for emb in embeddings_repeated]
            return embeddings
        
    gemini_embeddings_wrapper = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=gemini_embeddings_wrapper,    
        persist_directory="./chroma_db",
        collection_name="butterySmooth",
    )

    return 'done'



# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain import hub
# from langchain.docstore.document import Document
# from langchain.document_loaders import WebBaseLoader
# from langchain.schema import StrOutputParser
# from langchain.schema.prompt_template import format_document
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.vectorstores import Chroma
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from typing import List, Optional
# from langchain_google_genai import ChatGoogleGenerativeAI
# import os
# import getpass

# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.memory import ConversationBufferMemory
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import (
#     ChatGoogleGenerativeAI,
#     HarmBlockThreshold,
#     HarmCategory,
# )
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.agents import AgentExecutor
# from langchain import hub, LLMMathChain
# from langchain.agents.format_scratchpad import format_log_to_str
# from langchain.agents.output_parsers import ReActSingleInputOutputParser
# from langchain.tools.render import render_text_description
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.agents import load_tools, initialize_agent, AgentType, Agent
# from dotenv import dotenv_values
# from langchain import PromptTemplate
# from langchain.tools.retriever import create_retriever_tool