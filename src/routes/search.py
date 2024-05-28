from src.services.vectorstore_interface import *
from src.services.llm_interface import *
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/vector_search/")
async def get_message(text_context: str, collection_name: str):

    grounding, supporting_docs_list = embed_prompt_search(
        text_prompt = text_context,
        collection_name = collection_name
    )

    generated_response = openai_call(
        grounding = grounding, 
        text_prompt = text_context
    )

    return JSONResponse(content={
        "Text prompt": text_context,
        "Grounding": grounding,
        "Generated response": generated_response,
        "Supporting documents": supporting_docs_list
    },
    status_code = 200)

def embed_prompt_search(text_prompt, collection_name):

    VSS = VectorSimilaritySearch(
        db_path = r"data_store/vector_store",
        text_prompt = text_prompt,
        collection_name = collection_name
    )

    embedded_text = VSS.embedding_prompt()
    grounding, supporting_docs_list = VSS.query_vectorstore(text_embedding = embedded_text)

    #print("text prompt: ",text_prompt)
    #print("grounding: ",grounding)

    return grounding, supporting_docs_list

def openai_call(grounding, text_prompt):

    llm = LLMInterface(
        grounding = grounding,
        text_prompt = text_prompt
    )

    generated_response = llm.api_call(type = 'letter_generation')

    return generated_response