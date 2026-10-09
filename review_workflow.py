from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel,Field


load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=512)

class SentimentSchema(BaseModel):
    sentiment: Literal["positive","negative"] = Field(description="Sentiment of the review")

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX","Performance","Bug","Support","Other"] = Field(description="The category of issue mentioned in the review")
    tone: Literal["angry","frustrated","disappointed","calm"] = Field(description="The emotional tone of the user")
    urgency: Literal["low","high","medium"] = Field(description="How urgent or critical appears to be")

class ReviewState(TypedDict):
    review: str
    sentiment : Literal["positive","negative"]
    diagnosis :  dict
    response: str

structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)

def find_sentiment(state: ReviewState):
    prompt = f"For the following review find the sentiment \n {state['review']}"
    sentiment = structured_model.invoke(prompt)
    return {"sentiment":sentiment}

def positive_response(state:ReviewState):
    prompt = f"""Write a warm thank you message in response to this review:
    \n\n"{state['review']}\"\n
    Also, kindly ask the the user to leave the feedback on our website."""
    response = model.invoke(prompt).content
    return {"response":response}

def run_diagnosis(state:ReviewState):
    prompt = f"""Diagnose this negative review: \n\n{state['review']}\n"
    "Return issue_type, tone and urgency.
"""
    response = structured_model2.invoke(prompt)
    return {"diagnosis":response.model_dump()}

def negative_response(state:ReviewState):
    diagnosis = state['diagnosis']
    prompt = f"""You are a support assistant.
This user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}' and marked urgency as '{diagnosis['urgency']}'
Write an empathetic, helpful resolution message
"""
    response = model.invoke(prompt).content
    return {"response": response}


def check_sentiment(state:ReviewState) -> Literal["positive_response","run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"

# prompt = "What is the sentiment of the following review - This software is too bad"
# result= structured_model.invoke(prompt)
# result2 = structured_model2.invoke(prompt)
# print(result)

graph = StateGraph(ReviewState)
graph.add_node("find_sentiment",find_sentiment)
graph.add_node("positive_response",positive_response)
graph.add_node("run_diagnosis",run_diagnosis)
graph.add_node("negative_response",negative_response)


graph.add_edge(START,"find_sentiment")
graph.add_conditional_edges("find_sentiment",check_sentiment)
graph.add_edge("positive_response",END)
graph.add_edge("run_diagnosis","negative_response")
graph.add_edge("negative_response",END)

workflow = graph.compile()
initial_state = {
    "review":"Constant crashes, a clunky interface, and slow customer support make this software a waste of time and money. Look elsewhere."
}

result = workflow.invoke(initial_state)

print(result)