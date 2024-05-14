from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import os

# OpenAI 
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

#Prompt

prompt = hub.pull("hwchase17/openai-functions-agent")



# Wiki Tool

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)


# Web Page Retriever Load Tool
loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(documents,OpenAIEmbeddings())
retriever=vectordb.as_retriever()

retriever_tool=create_retriever_tool(retriever,"langsmith_search",
                      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")



# Arxiv Tool

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)


# All tools list

tools=[wiki,arxiv,retriever_tool]


# Agent Function With Query
def mock_agent_executor(query):
    agent=create_openai_tools_agent(llm,tools,prompt)
    agent_executor=AgentExecutor(agent=agent,tools=tools)
    response = agent_executor.invoke({"input":query})
    print(response)
    return (response)

# Streamlit Set up the page
st.title("Langchain Simple Agent")

# Create a text area for user input
user_input = st.text_input("Ask me anything about Langsmith/ from Wikipedia/ from Arxiv:", "")

# Button to send the input to the function
if st.button("Send"):
    if user_input:
        # Call your function here
        response = mock_agent_executor(user_input)
        
        # Display the response
        st.write(response["output"])
    else:
        st.write("Please enter a question.")