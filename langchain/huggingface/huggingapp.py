from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import os

hf_api_key=os.environ['HUGGINGFACEHUB_API_TOKEN']


## Read the ppdfs from the folder

loader=PyPDFDirectoryLoader("./us_census")
documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents=text_splitter.split_documents(documents)
final_documents[0]


## Embedding Using Huggingface
huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

## VectorStore Creation
vectorstore=FAISS.from_documents(final_documents[:120],huggingface_embeddings)

## Query using Similarity Search
query="WHAT IS HEALTH INSURANCE COVERAGE?"
relevant_docments=vectorstore.similarity_search(query)

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})


hf=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature":0.1,"max_length":500}
)

# query="What is the health insurance coverage?"
# hf.invoke(query)



# #Hugging Face models can be run locally through the HuggingFacePipeline class

# hf = HuggingFacePipeline.from_model_id(
#     model_id="mistralai/Mistral-7B-v0.1",
#     task="text-generation",
#     pipeline_kwargs={"temperature": 0, "max_new_tokens": 300}
# )

llm = hf 

# llm.invoke(query)

prompt_template="""
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{question}

Helpful Answers:
 """

prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

retrievalQA=RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
)

query="""DIFFERENCES IN THE
UNINSURED RATE BY STATE
IN 2022"""

# Call the QA chain with our query.
result = retrievalQA.invoke({"query": query})
print(result['result'])