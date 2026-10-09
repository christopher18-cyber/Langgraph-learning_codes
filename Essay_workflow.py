from langgraph.graph import START,END,StateGraph
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator


load_dotenv()
model = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=512)

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)
#
#
# essay = """The Core Role of Reducers in LangGraph
# Modern AI agents require robust state management to handle complex, multi-step tasks. In LangGraph, a framework built for stateful LLM workflows, this is controlled by reducers. A reducer is a function assigned to a specific state key that dictates how new data integrates with existing data, preventing vital information from being lost.
# Overcoming the Default "Overwrite" Behavior
# By default, LangGraph uses a "last-write-wins" rule. When a graph node finishes executing, its output completely overwrites the previous value of that state key. While this works for simple variables like counters, it fails for tasks that require memory, such as tracking a conversation history or accumulating search results. Reducers solve this by intercepting the update and executing a merge strategy: reducer(old_value, new_value).
# Common Integration Strategies
# • List Accumulation (operator.add): Instead of replacing a list, this appends new elements to the old ones. This is essential for building audit logs, step counters, or simple history tracking.
# • Intelligent Chat Memory (add_messages): Managing AI conversations requires more than simple appending. LangGraph’s built-in add_messages reducer reads unique message IDs. If a message ID already exists, it updates that message in place (useful for streaming or tool outputs); if it is new, it appends it to the chat history.
# """
#
# prompt = f"Evaluate the language quality of the following essay and provide and provide a feedback and assign a score out of 10 \n {essay}"
# result = structured_model.invoke(prompt)
# print(result.feedback)
# print(result.score)


class EssayState(TypedDict):
    essay:str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

def evaluate_language(state:EssayState):
    prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}"
    output = structured_model.invoke(prompt)
    return {"language_feedback":output.feedback,"individual_scores":[output.score]}

def evaluate_analysis(state:EssayState):
    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}"
    output = structured_model.invoke(prompt)
    return {"analysis_feedback":output.feedback,"individual_scores":[output.score]}

def evaluate_thought(state:EssayState):
    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}"
    output = structured_model.invoke(prompt)
    return {"clarity_feedback":output.feedback,"individual_scores":[output.scores]}

def final_evaluation (state:EssayState):
    prompt = f"Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}"
    overall_feedback = model.invoke(prompt).content

    avg_score = sum(state["individual_scores"])/len(state["individual_scores"])
    return {"overall_feedback":overall_feedback, "avg_score":avg_score}

graph = StateGraph(EssayState)
graph.add_node("evaluate_language",evaluate_language)
graph.add_node("evaluate_analysis",evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation",final_evaluation)


graph.add_edge(START,"evaluate_language")
graph.add_edge(START,"evaluate_analysis")
graph.add_edge(START,"evaluate_thought")
graph.add_edge("evaluate_language","final_evaluation")
graph.add_edge("evaluate_analysis","final_evaluation")
graph.add_edge("evaluate_thought","final_evaluation")
graph.add_edge("final_evaluation",END)

workflow = graph.compile()

essay = """The Core Role of Reducers in LangGraph
Modern AI agents require robust state management to handle complex, multi-step tasks. In LangGraph, a framework built for stateful LLM workflows, this is controlled by reducers. A reducer is a function assigned to a specific state key that dictates how new data integrates with existing data, preventing vital information from being lost.
Overcoming the Default "Overwrite" Behavior
By default, LangGraph uses a "last-write-wins" rule. When a graph node finishes executing, its output completely overwrites the previous value of that state key. While this works for simple variables like counters, it fails for tasks that require memory, such as tracking a conversation history or accumulating search results. Reducers solve this by intercepting the update and executing a merge strategy: reducer(old_value, new_value).
Common Integration Strategies
• List Accumulation (operator.add): Instead of replacing a list, this appends new elements to the old ones. This is essential for building audit logs, step counters, or simple history tracking.
• Intelligent Chat Memory (add_messages): Managing AI conversations requires more than simple appending. LangGraph’s built-in add_messages reducer reads unique message IDs. If a message ID already exists, it updates that message in place (useful for streaming or tool outputs); if it is new, it appends it to the chat history.
"""

initial_state = {"essay":essay}

result = workflow.invoke(initial_state)
print(result)