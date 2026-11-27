# backend.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
app = FastAPI()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class AnalysisRequest(BaseModel):
    prediction: str
    confidence: float
    symptoms: str

def get_llm_report(prediction, confidence, symptoms):
    example_input = "CNN Prediction: Non-Tumor (90.0%). Symptoms: mild headache."
    example_output = """
1. Finding: The MRI analysis shows no signs of a tumor with 90.0% confidence.
2. Symptom Context: The reported mild headache does not strongly correlate with severe neurological issues.
3. Risk Level: Low.
4. Recommendation: Routine checkup if symptoms persist.
5. Disclaimer: This is an educational demo, not medical advice.
"""
    prompt = f"""
    You are a strict medical AI assistant. You must follow the exact format of the example below.
    
    --- EXAMPLE START ---
    Input: {example_input}
    Output: {example_output}
    --- EXAMPLE END ---
    
    Now, generate the report for this real case using the EXACT same 5-point format:
    Input: CNN Prediction: {prediction} ({confidence*100:.1f}%). Symptoms: {symptoms}
    Output:
    """
    
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b", 
        temperature=0.1 
    )
    return response.choices[0].message.content

@app.post("/analyze")
async def analyze(req: AnalysisRequest):
    try:
        report = get_llm_report(req.prediction, req.confidence, req.symptoms)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))