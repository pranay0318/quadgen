from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import bs4
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

#text loader
loader=TextLoader("speech.txt")
text_documents=loader.load()
text_documents


#Webbase loader
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")

                     )))
text_documents=loader.load()
text_documents

## Pdf reader
loader=PyPDFLoader('atomic_habits.pdf')
docs=loader.load()
docs

# Text Splitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)

## Vector Embedding And Vector Store

db = Chroma.from_documents(documents,OpenAIEmbeddings())


query = "how did Author fix Sleep schedule?"
retireved_results=db.similarity_search(query)
print(retireved_results[0].page_content)