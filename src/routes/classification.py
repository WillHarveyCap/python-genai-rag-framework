from src.services.vectorstore_interface import *
from src.services.llm_interface import *
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/classification/")
async def get_message(text_context: str):

    generated_classification = classify_prompt(text_prompt = text_context)

    return JSONResponse(content={
        "Classification": generated_classification
    },
    status_code = 200)

def classify_prompt(text_prompt):

    grounding = 'N/a'

    llm = LLMInterface(
        grounding = grounding,
        text_prompt = text_prompt
    )

    generated_classification = llm.api_call(type = 'classification')

    return generated_classification