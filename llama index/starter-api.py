from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

app = FastAPI()
class Query(BaseModel):
    question: str

@app.post("/query_program")
def query_program(query: Query):
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents,show_progress=True)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do fix sleep schedule?")
    return{"response": response}