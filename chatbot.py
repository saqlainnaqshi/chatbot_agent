# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain_community.vectorstores import Chroma
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# # from langchain.memory import ConversationBufferMemory
# # from langchain_core.messages import HumanMessage, SystemMessage
# from langchain.tools.retriever import create_retriever_tool
# from typing import List, Optional
# # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser


# async def get_response(user_message):

#     load_dotenv()

#     google_api_key = os.getenv('GOOGLE_API_KEY')

#     if not google_api_key:
#         raise ValueError("Google API Key not found. Please set it in the .env file.")

#     os.environ['GOOGLE_API_KEY'] = google_api_key

#     class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
#         def embed_documents(self, texts: List[str], task_type: Optional[str] = None, titles: Optional[List[str]] = None, output_dimensionality: Optional[int] = None) -> List[List[float]]:
#             embeddings_repeated = super().embed_documents(texts, task_type, titles, output_dimensionality)
#             embeddings = [list(emb) for emb in embeddings_repeated]
#             return embeddings
        
#     gemini_embeddings_wrapper = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     vectorstore_disk = Chroma(
#         persist_directory="./chroma_db",
#         embedding_function=gemini_embeddings_wrapper,
#         collection_name="butterySmooth",
#     )

#     retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-pro",
#         handle_parsing_errors=True,
#         temperature=0.2,
#         top_p=0.85,
#         max_tokens= 200,
#     )

#     # response = llm.invoke('if asked about an itinerary generate a 3 day itinerary to kashmir and send with it a key and value as itinerary:true :' )
#     # output_parser = StrOutputParser()
#     # res= output_parser.invoke(response)
#     # return res


#     ddg_search = DuckDuckGoSearchAPIWrapper()

#     tools = [
#         Tool(
#             name="DuckDuckGo Search",
#             func=ddg_search.run,
#             description="Useful to browse information from the internet to know recent results and information you don't know. Then, tell user the result.",
#         ),
#         create_retriever_tool(
#             retriever,
#             name="Knowledge Base",
#             description="Useful for itinerary for itinerary generation for Kashmir. Has all the knowledge regarding to kashmir."
#         )
#     ]

#     # memory = ConversationBufferMemory(memory_key="chat_history")

#     conversation_history = []

#     FORMAT_INSTRUCTIONS="""
    
#     Generate a response to user query:
    
#     {query}

#     and use this for the chat history of user with you so that you can reply according to the context from chat history

#     {chat_history}
    
#     and if you don't find relevant information then use the tools to search the information 
    
#     To use a tools, please use the following format:
#     '''
#     Thought: Do I need to use a tool? Yes
#     Action: the action to take.
#     Action Input: the input to the action
#     Observation: the result of the action
#     Final Answer: the final answer
#     '''
    
#     If the context does not contain relevant information, try using the tools to search for answers to write the response.
    
#     Don't try searching anything in the context. Only use it as source of information to write the response.
    
#     If you don't find anything relevent in your search or in the context, just write the response with your best guess.

#     if generated response is an itinerary then add this to it, itinerary: true/false

#     """
    
#     PREFIX = '''You are an intelligent assistant to help the users and guide them. Your name is kashmiri guide and you have been designed and created by Saqlain Naqshi who is a web developer and an AI engineer'''
    
#     SUFFIX='''
#     Begin!
    
#     Instructions: {input}
#     {agent_scratchpad}
#     '''

#     prompt = PromptTemplate(
#         template=FORMAT_INSTRUCTIONS,
#         input_variables=["query", "chat_history"],
#     )

#     agent = initialize_agent(
#         tools,
#         llm=llm,
#         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#         verbose=True,
#         handle_parsing_errors=True,
#         agent_kwargs={
#             "suffix": SUFFIX,
#             "prefix": PREFIX
#         }
#     )

#     formatted_prompt = prompt.format(query=user_message, chat_history=conversation_history)
#     result = agent.invoke({"input": formatted_prompt})

#     # result = agent(prompt.format(query=user_message, chat_history=conversation_history))
#     response = result["output"]

#     # conversation_history.append(HumanMessage(user_message))
#     # conversation_history.append(SystemMessage(content=response))

#     return response







# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools.retriever import create_retriever_tool

# # Load environment variables
# load_dotenv()
# google_api_key = os.getenv('GOOGLE_API_KEY')
# if not google_api_key:
#     raise ValueError("Google API Key not found. Ensure it's set in the .env file.")

# # Set the API key explicitly
# os.environ['GOOGLE_API_KEY'] = google_api_key

# # Define embeddings
# class CustomEmbeddings(GoogleGenerativeAIEmbeddings):
#     def embed_documents(self, texts, **kwargs):
#         return [list(emb) for emb in super().embed_documents(texts, **kwargs)]

# # Initialize Chroma vector store
# vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=CustomEmbeddings(model="models/embedding-001"),
#     collection_name="butterySmooth",
# )
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # Initialize LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-pro",
#     temperature=0.2,
#     top_p=0.85,
#     max_tokens=200,
#     handle_parsing_errors=True,
# )

# # Initialize tools
# tools = [
#     Tool(
#         name="DuckDuckGo Search",
#         func=DuckDuckGoSearchAPIWrapper().run,
#         description="Search for recent or unknown information."
#     ),
#     create_retriever_tool(
#         retriever, 
#         name="Knowledge Base",
#         description="Retrieve itinerary information for Kashmir."
#     )
# ]

# # Prompt setup
# FORMAT_INSTRUCTIONS = """Your task is to generate a response to user queries{query}.
# Use tools only when needed. Format actions as:
# '''
# Thought: Do I need to use a tool? Yes
# Action: [tool name]
# Action Input: [input]
# Observation: [result]
# Final Answer: [answer]
# '''
# If the response is an itinerary, include 'itinerary: true/false' with your response."""
# prompt = PromptTemplate(template=FORMAT_INSTRUCTIONS, input_variables=["query"])

# # Initialize agent
# agent = initialize_agent(
#     tools=tools, 
#     llm=llm, 
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
#     verbose=True,
# )

# async def get_response(user_message):
#     try:
#         formatted_prompt = prompt.format(query=user_message)
#         result = agent.invoke({"input": formatted_prompt})
#         return result["output"]
#     except Exception as e:
#         return f"Error: {str(e)}"






# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import (
#     GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# )
# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools.retriever import create_retriever_tool
# from typing import List, Optional
# import logging

# # Initialize logging to debug on Heroku
# logging.basicConfig(level=logging.INFO)

# async def get_response(user_message: str) -> str:
#     try:
#         load_dotenv()  # Load environment variables

#         # Get Google API key from the environment
#         google_api_key = os.getenv('GOOGLE_API_KEY')
#         if not google_api_key:
#             raise ValueError("Google API Key not found. Set it in the .env file.")

#         # Ensure the environment variable is set in the runtime
#         os.environ['GOOGLE_API_KEY'] = google_api_key

#         # Define custom embedding wrapper
#         # class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
#         #     def embed_documents(
#         #         self, texts: List[str], task_type: Optional[str] = None,
#         #         titles: Optional[List[str]] = None, output_dimensionality: Optional[int] = None
#         #     ) -> List[List[float]]:
#         #         embeddings_repeated = super().embed_documents(
#         #             texts, task_type, titles, output_dimensionality
#         #         )
#         #         return [list(emb) for emb in embeddings_repeated]


#         class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
#             def embed_documents(self, texts: List[str]) -> List[List[float]]:
#                 # Call the parent class's embed_documents method
#                 embeddings = super().embed_documents(texts)
#                 # Process the embeddings as needed
#                 return [list(emb) for emb in embeddings]



#         # Set up the vector store using Chroma
#         gemini_embeddings_wrapper = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vectorstore_disk = Chroma(
#             persist_directory="./chroma_db",
#             embedding_function=gemini_embeddings_wrapper,
#             collection_name="butterySmooth"
#         )

#         # Initialize retriever with a fallback value for 'k'
#         retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

#         # Set up the Language Model (LLM) using Google Gemini
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-pro",
#             handle_parsing_errors=True,
#             temperature=0.2,
#             top_p=0.85,
#             max_tokens=200,
#         )

#         # Set up DuckDuckGo Search Tool
#         ddg_search = DuckDuckGoSearchAPIWrapper()

#         # Define tools including the retriever
#         tools = [
#             Tool(
#                 name="DuckDuckGo Search",
#                 func=ddg_search.run,
#                 description="Browse the internet for information."
#             ),
#             create_retriever_tool(
#                 retriever,
#                 name="Knowledge Base",
#                 description="Use for itinerary generation for Kashmir."
#             )
#         ]

#         # Conversation history as a list of previous interactions
#         conversation_history = []

#         # Define Prompt Template
#         FORMAT_INSTRUCTIONS = """
#         Generate a response to user query:
#         {query}

#         Use this chat history:
#         {chat_history}

#         Use tools if needed:
#         '''
#         Thought: Do I need to use a tool? Yes
#         Action: <tool_name>
#         Action Input: <input>
#         Observation: <result>
#         Final Answer: <answer>
#         '''
#         """
#         PREFIX = "You are an intelligent assistant named Kashmiri Guide, created by Saqlain Naqshi."
#         SUFFIX = "Begin!\n\nInstructions: {input}\n{agent_scratchpad}"

#         prompt = PromptTemplate(
#             template=FORMAT_INSTRUCTIONS,
#             input_variables=["query", "chat_history"],
#         )

#         # Initialize the agent with tools and LLM
#         agent = initialize_agent(
#             tools,
#             llm=llm,
#             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#             verbose=True,
#             handle_parsing_errors=True,
#             agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX}
#         )

#         # Format the input to the prompt
#         formatted_prompt = prompt.format(
#             query=user_message, 
#             chat_history=conversation_history or "No previous history."
#         )

#         # Invoke the agent and get the response
#         result = agent.invoke({"input": formatted_prompt})
#         response = result.get("output", "No output generated.")

#     except Exception as e:
#         # Log any errors to assist debugging
#         logging.error(f"Error generating response: {str(e)}")
#         response = f"Error: {str(e)}"

#     return response





# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools.retriever import create_retriever_tool
# import logging

# # Initialize logging for debugging
# logging.basicConfig(level=logging.INFO)

# async def get_response(user_message: str) -> str:
#     try:
#         load_dotenv()  # Load environment variables

#         # Ensure API key is loaded
#         google_api_key = os.getenv('GOOGLE_API_KEY')
#         if not google_api_key:
#             raise ValueError("Google API Key not found. Set it in the .env file.")

#         os.environ['GOOGLE_API_KEY'] = google_api_key

#         # Use the standard GoogleGenerativeAIEmbeddings
#         gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#         # Set up Chroma for vector storage and retrieval
#         vectorstore_disk = Chroma(
#             persist_directory="./chroma_db",
#             embedding_function=gemini_embeddings,
#             collection_name="butterySmooth"
#         )

#         # Use the retriever from the vector store
#         retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

#         # Set up the Google Gemini Pro LLM
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-pro",
#             temperature=0.2,
#             top_p=0.85,
#             max_tokens=200
#         )

#         # DuckDuckGo Search Tool
#         ddg_search = DuckDuckGoSearchAPIWrapper()

#         # Define tools including retriever
#         tools = [
#             Tool(
#                 name="DuckDuckGo Search",
#                 func=ddg_search.run,
#                 description="Browse the internet for information."
#             ),
#             create_retriever_tool(
#                 retriever,
#                 name="Knowledge Base",
#                 description="Use for information retrieval"
#             )
#         ]

#         # Conversation history as a list of previous interactions
#         conversation_history = []

#         # Define the prompt template
#         FORMAT_INSTRUCTIONS = """
#         Generate a response to the user query:
#         {query}

#         Use this chat history if relevant:
#         {chat_history}

#         Use tools if needed:
#         '''
#         Thought: Do I need to use a tool? Yes
#         Action: <tool_name>
#         Action Input: <input>
#         Observation: <result>
#         Final Answer: <answer>
#         '''
#         """
#         PREFIX = "You are an intelligent assistant named Kashmiri Guide."
#         SUFFIX = "Begin!\n\nInstructions: {input}\n{agent_scratchpad}"

#         prompt = PromptTemplate(
#             template=FORMAT_INSTRUCTIONS,
#             input_variables=["query", "chat_history"]
#         )

#         # Initialize the conversational agent
#         agent = initialize_agent(
#             tools=tools,
#             llm=llm,
#             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#             verbose=True,
#             handle_parsing_errors=True,
#             agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX}
#         )

#         # Format the input to the prompt
#         formatted_prompt = prompt.format(
#             query=user_message, 
#             chat_history=conversation_history or "No previous history."
#         )

#         # Get the response from the agent
#         result = agent.invoke({"input": formatted_prompt})
#         response = result.get("output", "No output generated.")

#     except Exception as e:
#         logging.error(f"Error generating response: {str(e)}")
#         response = f"Error: {str(e)}"

#     return response




# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools.retriever import create_retriever_tool
# from typing import List, Optional
# import logging

# # Initialize logging for debugging
# logging.basicConfig(level=logging.INFO)

# class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         # Use the parent class's embed_documents method to get embeddings
#         embeddings = super().embed_documents(texts)
        
#         # Flatten the embeddings if they are nested lists
#         flattened_embeddings = [embedding[0] if isinstance(embedding, list) and len(embedding) == 1 else embedding for embedding in embeddings]
        
#         return flattened_embeddings

# async def get_response(user_message: str) -> str:
#     try:
#         load_dotenv()  # Load environment variables

#         # Ensure API key is loaded
#         google_api_key = os.getenv('GOOGLE_API_KEY')
#         if not google_api_key:
#             raise ValueError("Google API Key not found. Set it in the .env file.")

#         os.environ['GOOGLE_API_KEY'] = google_api_key

#         # Use the custom GoogleGenerativeAIEmbeddings
#         gemini_embeddings = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")

#         # Set up Chroma for vector storage and retrieval
#         vectorstore_disk = Chroma(
#             persist_directory="./chroma_db",
#             embedding_function=gemini_embeddings,
#             collection_name="butterySmooth"
#         )

#         # Use the retriever from the vector store
#         retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

#         # Set up the Google Gemini Pro LLM
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-pro",
#             temperature=0.2,
#             top_p=0.85,
#             max_tokens=200
#         )

#         # DuckDuckGo Search Tool
#         ddg_search = DuckDuckGoSearchAPIWrapper()

#         # Define tools including retriever
#         tools = [
#             Tool(
#                 name="DuckDuckGo Search",
#                 func=ddg_search.run,
#                 description="Browse the internet for information."
#             ),
#             create_retriever_tool(
#                 retriever,
#                 name="Knowledge Base",
#                 description="Use for information retrieval"
#             )
#         ]

#         # Conversation history as a list of previous interactions
#         conversation_history = []

#         # Define the prompt template
#         FORMAT_INSTRUCTIONS = """
#         Generate a response to the user query:
#         {query}

#         Use this chat history if relevant:
#         {chat_history}

#         Use tools if needed:
#         '''
#         Thought: Do I need to use a tool? Yes
#         Action: <tool_name>
#         Action Input: <input>
#         Observation: <result>
#         Final Answer: <answer>
#         '''
#         """
#         PREFIX = "You are an intelligent assistant named Kashmiri Guide."
#         SUFFIX = "Begin!\n\nInstructions: {input}\n{agent_scratchpad}"

#         prompt = PromptTemplate(
#             template=FORMAT_INSTRUCTIONS,
#             input_variables=["query", "chat_history"]
#         )

#         # Initialize the conversational agent
#         agent = initialize_agent(
#             tools=tools,
#             llm=llm,
#             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#             verbose=True,
#             handle_parsing_errors=True,
#             agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX}
#         )

#         # Format the input to the prompt
#         formatted_prompt = prompt.format(
#             query=user_message, 
#             chat_history=conversation_history or "No previous history."
#         )

#         # Get the response from the agent
#         result = agent.invoke({"input": formatted_prompt})
#         response = result.get("output", "No output generated.")

#     except Exception as e:
#         logging.error(f"Error generating response: {str(e)}")
#         response = f"Error: {str(e)}"

#     return response




# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools.retriever import create_retriever_tool
# from typing import List
# import logging

# # Initialize logging for debugging
# logging.basicConfig(level=logging.INFO)

# class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         # Call the parent class's embed_documents method with only the required parameters
#         embeddings_repeated = super().embed_documents(texts)
#         # Ensure the embeddings are in the expected format
#         return [list(emb) for emb in embeddings_repeated]

# async def get_response(user_message: str) -> str:
#     try:
#         load_dotenv()  # Load environment variables

#         # Ensure API key is loaded
#         google_api_key = os.getenv('GOOGLE_API_KEY')
#         if not google_api_key:
#             raise ValueError("Google API Key not found. Set it in the .env file.")

#         os.environ['GOOGLE_API_KEY'] = google_api_key

#         # Use the custom embedding wrapper
#         gemini_embeddings = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")

#         # Set up Chroma for vector storage and retrieval
#         vectorstore_disk = Chroma(
#             persist_directory="./chroma_db",
#             embedding_function=gemini_embeddings,
#             collection_name="butterySmooth"
#         )

#         # Use the retriever from the vector store
#         retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

#         # Set up the Google Gemini Pro LLM
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-pro",
#             temperature=0.2,
#             top_p=0.85,
#             max_tokens=200
#         )

#         # DuckDuckGo Search Tool
#         ddg_search = DuckDuckGoSearchAPIWrapper()

#         # Define tools including retriever
#         tools = [
#             Tool(
#                 name="DuckDuckGo Search",
#                 func=ddg_search.run,
#                 description="Browse the internet for information."
#             ),
#             create_retriever_tool(
#                 retriever,
#                 name="Knowledge Base",
#                 description="Use for information retrieval"
#             )
#         ]

#         # Conversation history as a list of previous interactions
#         conversation_history = []

#         # Define the prompt template
#         FORMAT_INSTRUCTIONS = """
#         Generate a response to the user query:
#         {query}

#         Use this chat history if relevant:
#         {chat_history}

#         Use tools if needed:
#         '''
#         Thought: Do I need to use a tool? Yes
#         Action: <tool_name>
#         Action Input: <input>
#         Observation: <result>
#         Final Answer: <answer>
#         '''
#         """
#         PREFIX = "You are an intelligent assistant named Kashmiri Guide."
#         SUFFIX = "Begin!\n\nInstructions: {input}\n{agent_scratchpad}"

#         prompt = PromptTemplate(
#             template=FORMAT_INSTRUCTIONS,
#             input_variables=["query", "chat_history"]
#         )

#         # Initialize the conversational agent
#         agent = initialize_agent(
#             tools=tools,
#             llm=llm,
#             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#             verbose=True,
#             handle_parsing_errors=True,
#             agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX}
#         )

#         # Format the input to the prompt
#         formatted_prompt = prompt.format(
#             query=user_message, 
#             chat_history=conversation_history or "No previous history."
#         )

#         # Get the response from the agent
#         result = agent.invoke({"input": formatted_prompt})
#         response = result.get("output", "No output generated.")

#     except Exception as e:
#         logging.error(f"Error generating response: {str(e)}")
#         response = f"Error: {str(e)}"

#     return response





import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools.retriever import create_retriever_tool
import logging
from typing import List, Any

# Initialize logging for debugging
logging.basicConfig(level=logging.INFO)

class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        # Call the parent class's embed_documents method with only the required parameters
        embeddings_repeated = super().embed_documents(texts)
        # Ensure the embeddings are in the expected format
        return [list(emb) for emb in embeddings_repeated]

async def get_response(user_message: str) -> str:
    try:
        load_dotenv()  # Load environment variables

        # Ensure API key is loaded
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise ValueError("Google API Key not found. Set it in the .env file.")

        os.environ['GOOGLE_API_KEY'] = google_api_key

        # Use the custom embedding wrapper
        gemini_embeddings = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Set up Chroma for vector storage and retrieval
        vectorstore_disk = Chroma(
            persist_directory="./chroma_db",
            embedding_function=gemini_embeddings,
            collection_name="butterySmooth"
        )

        # Use the retriever from the vector store
        retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

        # Set up the Google Gemini Pro LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.2,
            top_p=0.85,
            max_tokens=200
        )

        # DuckDuckGo Search Tool
        ddg_search = DuckDuckGoSearchAPIWrapper()

        # Define tools including retriever
        tools = [
            Tool(
                name="DuckDuckGo Search",
                func=ddg_search.run,
                description="Browse the internet for information."
            ),
            create_retriever_tool(
                retriever,
                name="Knowledge Base",
                description="Use for information retrieval"
            )
        ]

        # Conversation history as a list of previous interactions
        conversation_history = []

        # Define the prompt template
        FORMAT_INSTRUCTIONS = """
        Generate a response to the user query:
        {query}

        Use this chat history if relevant:
        {chat_history}

        Use tools if needed:
        '''
        Thought: Do I need to use a tool? Yes
        Action: <tool_name>
        Action Input: <input>
        Observation: <result>
        Final Answer: <answer>
        '''
        """
        PREFIX = "You are an intelligent assistant named Kashmiri Guide."
        SUFFIX = "Begin!\n\nInstructions: {input}\n{agent_scratchpad}"

        prompt = PromptTemplate(
            template=FORMAT_INSTRUCTIONS,
            input_variables=["query", "chat_history"]
        )

        # Initialize the conversational agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX}
        )

        # Format the input to the prompt
        formatted_prompt = prompt.format(
            query=user_message, 
            chat_history=conversation_history or "No previous history."
        )

        # Get the response from the agent
        result = agent.invoke({"input": formatted_prompt})
        response = result.get("output", "No output generated.")

    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        response = f"Error: {str(e)}"

    return response




# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import (
#     GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# )
# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools.retriever import create_retriever_tool
# from typing import List, Optional
# import logging

# # Initialize logging for debugging on Heroku
# logging.basicConfig(level=logging.INFO)

# async def get_response(user_message: str) -> str:
#     try:
#         # Load environment variables
#         load_dotenv()

#         # Retrieve the Google API key
#         google_api_key = os.getenv('GOOGLE_API_KEY')
#         if not google_api_key:
#             raise ValueError("Google API Key not found. Set it in the .env file.")

#         # Ensure API key is available at runtime
#         os.environ['GOOGLE_API_KEY'] = google_api_key

#         # Define a custom class for embedding documents using Gemini
#         class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
#             def embed_documents(self, texts: List[str]) -> List[List[float]]:
#                 # Use the correct function call as per Google Gemini's documentation
#                 embeddings = self.embed_content(
#                     model="models/embedding-001",  # Specify model correctly
#                     content=texts,  # Pass content to be embedded
#                     task_type="retrieval_document"  # Optional: Specify task type for better embedding
#                 )
#                 return [list(emb) for emb in embeddings]

#         # Instantiate the embeddings wrapper
#         gemini_embeddings = CustomGoogleGenerativeAIEmbeddings()

#         # Configure Chroma vector store with embeddings
#         vectorstore_disk = Chroma(
#             persist_directory="./chroma_db",
#             embedding_function=gemini_embeddings,
#             collection_name="butterySmooth"
#         )

#         # Initialize retriever from Chroma with a 'k' fallback
#         retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

#         # Set up the LLM using Google Gemini Pro model
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-pro",
#             handle_parsing_errors=True,
#             temperature=0.2,
#             top_p=0.85,
#             max_tokens=200
#         )

#         # Configure the DuckDuckGo Search tool
#         ddg_search = DuckDuckGoSearchAPIWrapper()

#         # Define tools to use within the agent
#         tools = [
#             Tool(
#                 name="DuckDuckGo Search",
#                 func=ddg_search.run,
#                 description="Search the web for current information."
#             ),
#             create_retriever_tool(
#                 retriever,
#                 name="Knowledge Base",
#                 description="Retrieve information for itinerary generation for Kashmir."
#             )
#         ]

#         # Maintain conversation history for contextual responses
#         conversation_history = []

#         # Define the prompt template
#         FORMAT_INSTRUCTIONS = """
#         Generate a response to the user query:
#         {query}

#         Use the chat history if relevant:
#         {chat_history}

#         If required, use tools to gather information:
#         '''
#         Thought: Do I need to use a tool? Yes
#         Action: <tool_name>
#         Action Input: <input>
#         Observation: <result>
#         Final Answer: <answer>
#         '''
#         """

#         PREFIX = "You are an intelligent assistant named Kashmiri Guide, created by Saqlain Naqshi."
#         SUFFIX = "Begin!\n\nInstructions: {input}\n{agent_scratchpad}"

#         # Create the prompt template object
#         prompt = PromptTemplate(
#             template=FORMAT_INSTRUCTIONS,
#             input_variables=["query", "chat_history"]
#         )

#         # Initialize the conversational agent
#         agent = initialize_agent(
#             tools=tools,
#             llm=llm,
#             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#             verbose=True,
#             handle_parsing_errors=True,
#             agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX}
#         )

#         # Format the prompt with the user's query and history
#         formatted_prompt = prompt.format(
#             query=user_message, 
#             chat_history=conversation_history or "No previous history."
#         )

#         # Invoke the agent to generate a response
#         result = agent.invoke({"input": formatted_prompt})
#         response = result.get("output", "No output generated.")

#     except Exception as e:
#         # Log any exceptions for debugging purposes
#         logging.error(f"Error generating response: {str(e)}")
#         response = f"Error: {str(e)}"

#     return response





# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import (
#     GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# )
# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools.retriever import create_retriever_tool
# from typing import List, Optional
# import logging

# # Initialize logging for debugging on Heroku
# logging.basicConfig(level=logging.INFO)

# async def get_response(user_message: str) -> str:
#     try:
#         # Load environment variables
#         load_dotenv()

#         # Retrieve the Google API key
#         google_api_key = os.getenv('GOOGLE_API_KEY')
#         if not google_api_key:
#             raise ValueError("Google API Key not found. Set it in the .env file.")

#         # Ensure API key is available at runtime
#         os.environ['GOOGLE_API_KEY'] = google_api_key

#         # Define a custom class for embedding documents using Gemini
#         class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
#             def __init__(self, model: str):
#                 super().__init__(model=model)

#             def embed_documents(self, texts: List[str]) -> List[List[float]]:
#                 # Call the parent class's method without the task_type parameter
#                 embeddings = self.embed_content(
#                     model=self.model,  # Use the model initialized in the constructor
#                     content=texts  # Pass the content to be embedded
#                 )
#                 return [list(emb) for emb in embeddings]

#         # Instantiate the embeddings wrapper with the model specified
#         gemini_embeddings = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")

#         # Configure Chroma vector store with embeddings
#         vectorstore_disk = Chroma(
#             persist_directory="./chroma_db",
#             embedding_function=gemini_embeddings,
#             collection_name="butterySmooth"
#         )

#         # Initialize retriever from Chroma with a 'k' fallback
#         retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

#         # Set up the LLM using Google Gemini Pro model
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-pro",
#             handle_parsing_errors=True,
#             temperature=0.2,
#             top_p=0.85,
#             max_tokens=200
#         )

#         # Configure the DuckDuckGo Search tool
#         ddg_search = DuckDuckGoSearchAPIWrapper()

#         # Define tools to use within the agent
#         tools = [
#             Tool(
#                 name="DuckDuckGo Search",
#                 func=ddg_search.run,
#                 description="Search the web for current information."
#             ),
#             create_retriever_tool(
#                 retriever,
#                 name="Knowledge Base",
#                 description="Retrieve information for itinerary generation for Kashmir."
#             )
#         ]

#         # Maintain conversation history for contextual responses
#         conversation_history = []

#         # Define the prompt template
#         FORMAT_INSTRUCTIONS = """
#         Generate a response to the user query:
#         {query}

#         Use the chat history if relevant:
#         {chat_history}

#         If required, use tools to gather information:
#         '''
#         Thought: Do I need to use a tool? Yes
#         Action: <tool_name>
#         Action Input: <input>
#         Observation: <result>
#         Final Answer: <answer>
#         '''
#         """

#         PREFIX = "You are an intelligent assistant named Kashmiri Guide, created by Saqlain Naqshi."
#         SUFFIX = "Begin!\n\nInstructions: {input}\n{agent_scratchpad}"

#         # Create the prompt template object
#         prompt = PromptTemplate(
#             template=FORMAT_INSTRUCTIONS,
#             input_variables=["query", "chat_history"]
#         )

#         # Initialize the conversational agent
#         agent = initialize_agent(
#             tools=tools,
#             llm=llm,
#             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#             verbose=True,
#             handle_parsing_errors=True,
#             agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX}
#         )

#         # Format the prompt with the user's query and history
#         formatted_prompt = prompt.format(
#             query=user_message, 
#             chat_history=conversation_history or "No previous history."
#         )

#         # Invoke the agent to generate a response
#         result = agent.invoke({"input": formatted_prompt})
#         response = result.get("output", "No output generated.")

#     except Exception as e:
#         # Log any exceptions for debugging purposes
#         logging.error(f"Error generating response: {str(e)}")
#         response = f"Error: {str(e)}"

#     return response





# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import (
#     GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# )
# from langchain.agents import Tool, AgentType, initialize_agent
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools.retriever import create_retriever_tool
# from typing import List
# import logging

# # Initialize logging for debugging on Heroku
# logging.basicConfig(level=logging.INFO)

# async def get_response(user_message: str) -> str:
#     try:
#         # Load environment variables
#         load_dotenv()

#         # Retrieve the Google API key
#         google_api_key = os.getenv('GOOGLE_API_KEY')
#         if not google_api_key:
#             raise ValueError("Google API Key not found. Set it in the .env file.")

#         # Ensure API key is available at runtime
#         os.environ['GOOGLE_API_KEY'] = google_api_key

#         class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
#             def __init__(self, model: str):
#                 super().__init__(model=model)

#             def embed_documents(self, texts: List[str], task_type: str = 'retrieval_document') -> List[List[float]]:
#                 # Call the parent class's method with the specified task_type
#                 embeddings = super().embed_documents(texts, task_type=task_type)
#                 return [list(emb) for emb in embeddings]

#         # Instantiate the embeddings wrapper with the model specified
#         gemini_embeddings = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")

#         # Configure Chroma vector store with embeddings
#         vectorstore_disk = Chroma(
#             persist_directory="./chroma_db",
#             embedding_function=gemini_embeddings,
#             collection_name="butterySmooth"
#         )

#         # Initialize retriever from Chroma with a 'k' fallback
#         retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

#         # Set up the LLM using Google Gemini Pro model
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-pro",
#             handle_parsing_errors=True,
#             temperature=0.2,
#             top_p=0.85,
#             max_tokens=200
#         )

#         # Configure the DuckDuckGo Search tool
#         ddg_search = DuckDuckGoSearchAPIWrapper()

#         # Define tools to use within the agent
#         tools = [
#             Tool(
#                 name="DuckDuckGo Search",
#                 func=ddg_search.run,
#                 description="Search the web for current information."
#             ),
#             create_retriever_tool(
#                 retriever,
#                 name="Knowledge Base",
#                 description="Retrieve information for itinerary generation for Kashmir."
#             )
#         ]

#         # Maintain conversation history for contextual responses
#         conversation_history = []

#         # Define the prompt template
#         FORMAT_INSTRUCTIONS = """
#         Generate a response to the user query:
#         {query}

#         Use the chat history if relevant:
#         {chat_history}

#         If required, use tools to gather information:
#         '''
#         Thought: Do I need to use a tool? Yes
#         Action: <tool_name>
#         Action Input: <input>
#         Observation: <result>
#         Final Answer: <answer>
#         '''
#         """

#         PREFIX = "You are an intelligent assistant named Kashmiri Guide, created by Saqlain Naqshi."
#         SUFFIX = "Begin!\n\nInstructions: {input}\n{agent_scratchpad}"

#         # Create the prompt template object
#         prompt = PromptTemplate(
#             template=FORMAT_INSTRUCTIONS,
#             input_variables=["query", "chat_history"]
#         )

#         # Initialize the conversational agent
#         agent = initialize_agent(
#             tools=tools,
#             llm=llm,
#             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#             verbose=True,
#             handle_parsing_errors=True,
#             agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX}
#         )

#         # Format the prompt with the user's query and history
#         formatted_prompt = prompt.format(
#             query=user_message, 
#             chat_history=conversation_history or "No previous history."
#         )

#         # Invoke the agent to generate a response
#         result = agent.invoke({"input": formatted_prompt})
#         response = result.get("output", "No output generated.")

#     except Exception as e:
#         # Log any exceptions for debugging purposes
#         logging.error(f"Error generating response: {str(e)}")
#         response = f"Error: {str(e)}"

#     return response
