import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from typing import List, Optional
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


async def get_response(user_message):

    load_dotenv()

    google_api_key = os.getenv('GOOGLE_API_KEY')

    if not google_api_key:
        raise ValueError("Google API Key not found. Please set it in the .env file.")

    os.environ['GOOGLE_API_KEY'] = google_api_key

    class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
        def embed_documents(self, texts: List[str], task_type: Optional[str] = None, titles: Optional[List[str]] = None, output_dimensionality: Optional[int] = None) -> List[List[float]]:
            embeddings_repeated = super().embed_documents(texts, task_type, titles, output_dimensionality)
            embeddings = [list(emb) for emb in embeddings_repeated]
            return embeddings
        
    gemini_embeddings_wrapper = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore_disk = Chroma(
        persist_directory="./chroma_db",
        embedding_function=gemini_embeddings_wrapper,
        collection_name="butterySmooth",
    )

    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        handle_parsing_errors=True,
        temperature=0.2,
        top_p=0.85,
        max_tokens= 200,
    )

    response = llm.invoke('generate a 3 day itinerary to kashmir')
    output_parser = StrOutputParser()
    res= output_parser.invoke(response)
    return res


    ddg_search = DuckDuckGoSearchAPIWrapper()

    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=ddg_search.run,
            description="Useful to browse information from the internet to know recent results and information you don't know. Then, tell user the result.",
        ),
        create_retriever_tool(
            retriever,
            name="Knowledge Base",
            description="Useful for itinerary for itinerary generation for Kashmir. Has all the knowledge regarding to kashmir."
        )
    ]

    # memory = ConversationBufferMemory(memory_key="chat_history")

    conversation_history = []

    FORMAT_INSTRUCTIONS="""
    
    Generate a response to user query:
    
    {query}

    and use this for the chat history of user with you so that you can reply according to the context from chat history

    {chat_history}
    
    and if you don't find relevant information then use the tools to search the information 
    
    To use a tools, please use the following format:
    '''
    Thought: Do I need to use a tool? Yes
    Action: the action to take.
    Action Input: the input to the action
    Observation: the result of the action
    Final Answer: the final answer
    '''
    
    If the context does not contain relevant information, try using the tools to search for answers to write the response.
    
    Don't try searching anything in the context. Only use it as source of information to write the response.
    
    If you don't find anything relevent in your search or in the context, just write the response with your best guess.

    if generated response is an itinerary then add this to it, itinerary: true/false

    """
    
    PREFIX = '''You are an intelligent assistant to help the users and guide them. Your name is kashmiri guide and you have been designed and created by Saqlain Naqshi who is a web developer and an AI engineer'''
    
    SUFFIX='''
    Begin!
    
    Instructions: {input}
    {agent_scratchpad}
    '''

    prompt = PromptTemplate(
        template=FORMAT_INSTRUCTIONS,
        input_variables=["query", "chat_history"],
    )

    agent = initialize_agent(
        tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "suffix": SUFFIX,
            "prefix": PREFIX
        }
    )

    formatted_prompt = prompt.format(query=user_message, chat_history=conversation_history)
    result = agent.invoke({"input": formatted_prompt})

    # result = agent(prompt.format(query=user_message, chat_history=conversation_history))
    response = result["output"]

    # conversation_history.append(HumanMessage(user_message))
    # conversation_history.append(SystemMessage(content=response))

    return response