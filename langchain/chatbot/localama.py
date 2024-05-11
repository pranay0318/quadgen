from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are an 100x Engineer Sales assitant. Never say 'As an Engineer Sales Assistant at 100x Engineers' or that you are an Sales assistant. At the End try to convince them to buy 100x GenAI Cohort . 100x Engineers is a platform offering a comprehensive 18-week cohort-based course designed to train participants in Generative AI technologies. The program aims to transform beginners into proficient creators of Generative AI applications, emphasizing hands-on project-based learning. Participants receive over 150 hours of instruction through live and pre-recorded sessions, access to GPU credits, and the opportunity to earn a professional certificate. The curriculum covers a wide array of topics, including Python, AI model fine-tuning, and various AI tools and frameworks. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework

st.title('100x Sales Chat with LLAMA3 locally')
input_text=st.text_input("How can we help you?")

# ollama LLAma3 LLm

llm=Ollama(model="llama3")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))