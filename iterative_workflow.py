from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, Literal, Annotated
from langchain_core.messages import SystemMessage,HumanMessage
import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

generator_llm = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=512)
evaluator_llm = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=512)
optimizer_llm = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=512)


class PostEvaluation(BaseModel):
    evaluation: Literal["approved","needs_improvement"] = Field(...,description="Final evaluation result")
    feedback: str = Field(...,description="Feedback for the facebook post")

structured_evaluator_llm = evaluator_llm.with_structured_output(PostEvaluation)

post = """
The Shift from Prompting to Orchestrating: Why Agentic AI is the Next Big Leap 🚀For the past few years, our relationship with AI has largely been transactional: Prompt $\rightarrow$ Wait $\rightarrow$ Copy/Paste.
We treat models as advanced search engines or static text generators, doing the heavy lifting of breaking down tasks ourselves.Enter Agentic AI—and the paradigm is shifting entirely.
Instead of just answering questions, AI agents can now reason, plan, use tools, and execute multi-step workflows autonomously.
We are moving from chatting with models to orchestrating agent teams.   What makes Agentic AI different?Autonomous Reasoning & Planning: Give an agent a high-level goal, and it breaks the objective down into a sequence of actionable steps.Tool Integration: Agents aren't trapped in a text box. 
They can query databases, execute code, call APIs, scrape documentation, and validate their own outputs.   Self-Correction Loops: Unlike traditional scripts that fail outright when things go sideways, agentic loops allow models to evaluate results, catch errors, retry, and adapt on the fly. 
Building the FutureWhether it's orchestrating stateful graphs with frameworks like LangGraph, managing microservice-backed workflows, or building multi-agent systems for security and automation, engineering has evolved. We aren't just writing application logic anymore; we are designing collaborative cognitive architectures.
How are you incorporating AI agents into your development stack or daily workflows? Let’s discuss in the comments! 👇
"""
#
# result = structured_evaluator_llm.invoke(post)
# print(result)
class PostState(TypedDict):
    topic: str
    post: str
    evaluation: Literal["approved","needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

    past_history: Annotated[list[str],operator.add]
    feedback_history: Annotated[list[str], operator.add]

def generate_post(state:PostState):
    messages = [
        SystemMessage(content="you are a funny and clever facebook influencer"),
        HumanMessage(content=f"""
        Write a short, original and hilarious facebook post on the topic: "{state['topic']}"
        
        Rules:
        - DO NOT use question-answer format.
        - Max 500 characters.
        - Use observational humor, irony, sarcasm or cultural reference.
        - Think in meme logic, punchlines or relatable takes
        - Use simple, day to day English
        """)
    ]
    response = generator_llm.invoke(messages).content

    return {'post':response,'post_history':[response]}


def evaluate_post(state: PostState):
    # prompt
    messages = [
        SystemMessage(
            content="You are a ruthless, no-laugh-given Facebook critic. You evaluate posts based on humor, originality, virality, and post format."),
        HumanMessage(content=f"""
Evaluate the following Facebook post:

Post: "{state['post']}"

Use the criteria below to evaluate the post:

1. Originality $\square$ Is this fresh, or have you seen it a hundred times before?
2. Humor $\square$ Did it genuinely make you smile, laugh, or chuckle?
3. Punchiness $\square$ Is it short, sharp, and scroll-stopping?
4. Virality Potential $\\square$ Would people share, react, or comment on it?
5. Format $\square$ Is it a well-formed Facebook post (not a setup-punchline joke, not a Q&A joke, and under 500 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 500 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"
- feedback: One paragraph explaining the strengths and weaknesses
""")
    ]

    response = structured_evaluator_llm.invoke(messages)

    return {'evaluation': response.evaluation, 'feedback': response.feedback, 'feedback_history': [response.feedback]}


def optimize_post(state: PostState):
    messages = [
        SystemMessage(content="You punch up Facebook posts for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the Facebook post based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Post:
{state['post']}

Re-write it as a short, viral-worthy Facebook post. Avoid Q&A style and stay under 500 characters.
""")
    ]

    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'post': response, 'iteration': iteration, 'post_history': [response]}

def route_evaluation(state:PostState):
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return "approved"
    else:
        return "needs_improvement"


graph = StateGraph(PostState)
graph.add_node("generate",generate_post)
graph.add_node("evaluate",evaluate_post)
graph.add_node("optimize",optimize_post)


graph.add_edge(START,"generate")
graph.add_edge("generate","evaluate")
graph.add_conditional_edges("evaluate",route_evaluation,{'approved':END,'needs_improvement':'optimize'})
graph.add_edge("optimize","evaluate")

workflow = graph.compile()

initial_state = {
    "topic":"agentic AI",
    "iteration": 1,
    "max_iteration":3
}


result = workflow.invoke(initial_state)
print(result)