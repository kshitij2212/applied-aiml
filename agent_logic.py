import os
import json
from typing import TypedDict, List, Annotated, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    patient_data: Dict
    prediction_results: Dict
    identified_factors: List[str]
    retrieved_guidelines: List[Dict]
    final_report: Dict
    messages: List[Annotated[str, "The conversation history"]]

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY not found in .env file")
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.1
    )

def analyze_risk_factors(state: AgentState):
    """Analyze patient data and ML prediction to identify key risk factors."""
    llm = get_llm()
    patient = state['patient_data']
    pred = state['prediction_results']
    
    prompt = f"""
    Analyze the following patient's no-show risk.
    ML Prediction: {pred['probability']:.2%} probability of no-show.
    Patient Profile:
    - Age: {patient['Age']}
    - Gender: {patient['Gender']}
    - Lead Time: {patient['LeadTime']} days
    - SMS Received: {'Yes' if patient['SMS_received'] else 'No'}
    - Chronic Conditions: Diabetes ({patient['Diabetes']}), Hypertension ({patient['Hipertension']})
    - Social Support: Scholarship ({patient['Scholarship']})
    
    Identify the top 3 specific factors contributing to this risk.
    Return ONLY a JSON list of strings.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.replace("```json", "").replace("```", "").strip()
    try:
        factors = json.loads(content)
    except:
        factors = ["High Lead Time", "Logistic Barriers", "Chronic Condition Management"]
        
    return {**state, "identified_factors": factors}

def retrieve_guidelines(state: AgentState):
    """Retrieve relevant care coordination guidelines (Basic RAG)."""
    with open("knowledge_base.json", "r") as f:
        kb = json.load(f)
    
    factors = " ".join(state['identified_factors']).lower()
    retrieved = []
    
    for entry in kb:
        if any(word in entry['strategy'].lower() or word in entry['intervention'].lower() for word in factors.split()):
            retrieved.append(entry)
        
    if not retrieved:
        retrieved = kb[:2]
        
    return {**state, "retrieved_guidelines": retrieved}

def generate_report(state: AgentState):
    """Generate the final structured care coordination report."""
    llm = get_llm()
    patient = state['patient_data']
    pred = state['prediction_results']
    factors = state['identified_factors']
    guidelines = state['retrieved_guidelines']
    
    prompt = f"""
    Generate a structured Care Coordination Report.
    
    Patient Risk: {pred['probability']:.2%}
    Top Contributors: {', '.join(factors)}
    Available Guidelines: {json.dumps(guidelines)}
    
    The report MUST include:
    1. Risk Summary: Brief overview of the likelihood.
    2. Key Contributing Factors: Detailed explanation of why.
    3. Recommended Intervention Strategies: Actionable steps for the clinic.
    4. Supporting Sources: Reference the guidelines provided.
    5. Operational/Ethical Disclaimers: Standard medical AI warnings.
    
    Return the response as a valid JSON object with these keys: 
    'summary', 'factors', 'strategies', 'sources', 'disclaimers'.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.replace("```json", "").replace("```", "").strip()
    try:
        report = json.loads(content)
    except:
        report = {
            "summary": "High risk of no-show detected.",
            "factors": factors,
            "strategies": ["Immediate SMS reminder", "Consider telehealth"],
            "sources": ["General Healthcare Guidelines"],
            "disclaimers": "This is an AI-generated suggestion. Clinical judgment should prevail."
        }
        
    return {**state, "final_report": report}

def create_care_agent():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyzer", analyze_risk_factors)
    workflow.add_node("retriever", retrieve_guidelines)
    workflow.add_node("generator", generate_report)

    workflow.set_entry_point("analyzer")

    workflow.add_edge("analyzer", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)
    
    return workflow.compile()

def run_agent(patient_data, prediction_results):
    app = create_care_agent()
    initial_state = {
        "patient_data": patient_data,
        "prediction_results": prediction_results,
        "identified_factors": [],
        "retrieved_guidelines": [],
        "final_report": {},
        "messages": []
    }
    return app.invoke(initial_state)

if __name__ == "__main__":
    test_patient = {
        'Age': 45, 'Gender': 'F', 'Scholarship': 1, 'Hipertension': 1, 
        'Diabetes': 0, 'Alcoholism': 0, 'Handcap': 0, 'SMS_received': 0, 
        'LeadTime': 20, 'DayOfWeek': 2
    }
    test_pred = {'probability': 0.72}
