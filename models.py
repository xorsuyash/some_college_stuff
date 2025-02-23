from pydantic import BaseModel

class EmailResponse(BaseModel):
    email_response:str

class EvaluateResponse(BaseModel):
    Relevance:int
    Conciseness:int
    Politeness:int
    Adaptability:int 
    response_feedback:int 

class EvolvedResponse(BaseModel):
    evolved_response:str