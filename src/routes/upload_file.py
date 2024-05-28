from src.services.vectorstore_interface import *
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from nltk.corpus import stopwords
import nltk
import pandas as pd

router = APIRouter()

@router.post("/upload_doc/")
async def upload_doc(file_name, file: UploadFile = File(...)):
    
    contents = await file.read()
    text = contents.decode("utf-8")

    file_path = f"""data_store/context_store/{file_name}.txt"""
    with open(file_path, 'w') as doc:
        doc.write(text)
    
    embed_insert_doc(
        collection_name = "COLLECTION_NAME",
        text_content = text,
        file_name = file_name
    )

    return JSONResponse(content={"message": "File uploaded, embedded and inserted successfully"}, status_code=200)

def embed_insert_doc(collection_name, text_content,file_name):

    insert_vdb = InsertVectorData(
        collection_name = collection_name,
        text_context = text_content,
        db_path = r"data_store/vector_store"
    )
    
    insert_vdb.make_pull_collection()
    chunked_data = insert_vdb.context_chunking()
    insert_vdb.insert_data(chunked_context = chunked_data, file_name=file_name)