from pathlib import Path
from src.ingestion.loader import loader
from src.ingestion.chunker import chunking
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File

import os
import shutil


app = FastAPI()
KNOWLEDGE_PATH = Path(__file__).parent / "knowledge" / "knowledge.txt"

knowledge = KNOWLEDGE_PATH.read_text(encoding="utf-8")

class loader:
     def __init__(self, filepath, filename):

        self.filepath = filepath
        self.filename = filename

        self.name = self.filename.rsplit(".", 1)[0]

        self.index_pt = f"vector_store/{self.name}_index.faiss"
        self.chunks_pt = f"vector_store/{self.name}_chunks.json"
        self.index=None
        self.chunk=None

        self.index = None
     def index_and_chunk(self):
         load = loader(self.filepath)
         chunkes = chunking(load["document"]) 
         self.chunk = chunkes["chunk"]
         embedding = embedding(self.chunk)
         self.index = indexing(embedding("embeddings"))
         vector_storage = save_index(self.index,self.chunk,self.index_pt,self.chunks_pt)
         print(vector_storage["message"])
         return vector_storage





@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)):
    os.makedirs("data/raw",exist_ok=True )

    filepath = f"data/raw/{file.filename}"

    with open(filepath,"wb") as buffer:
        shutil.copyfileobj( file.file,buffer)

    load = loader(
        filepath,
        file.filename
    )

    result = load.index_and_chunk()

    return result


class querry(BaseModel):

    querry: str
