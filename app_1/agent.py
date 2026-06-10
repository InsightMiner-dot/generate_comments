import os
import io
import pandas as pd
import matplotlib.pyplot as plt
from typing import TypedDict, Annotated, Dict, List, Any
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pptx import Presentation
from pptx.util import Inches

# --- 1. LLM Setup ---
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_KEY")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model_name = os.getenv("MODEL_NAME")

llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version=azure_api_version,
    deployment_name=azure_deployment,
    model=model_name,
    temperature=0.2
)

# --- 2. Data Models & State ---
class SuggestionList(BaseModel):
    suggestions: List[str] = Field(description="Exactly 3 short, distinct suggestions.")

class SlideContent(BaseModel):
    title: str = Field(description="Slide title")
    bullet_points: List[str] = Field(description="3-5 bullet points")

class ChartSpecification(BaseModel):
    chart_title: str = Field(description="Title of the chart")
    x_column: str = Field(description="Exact column for X-axis from schema")
    y_column: str = Field(description="Exact column for Y-axis from schema")
    chart_type: str = Field(description="'bar', 'line', or 'scatter'")

class PresentationDeck(BaseModel):
    slides: List[SlideContent]

class PresentationState(TypedDict):
    messages: Annotated[list, add_messages]
    schema_info: str
    data_summary: str
    dataframe_json: str  
    # Wizard State
    current_step: str 
    suggestions: List[str]
    # Gathered Parameters
    persona: str
    topics: str
    pages: str
    title: str
    graph_request: str
    # Outputs
    output_ready: bool
    final_pptx_bytes: bytes

# --- 3. Graph Nodes ---

def ingest_and_summarize(state: PresentationState):
    """Step 0: On Upload. Summarizes data and asks for Persona."""
    schema = state["schema_info"]
    prompt = f"Analyze this dataset schema and provide a brief summary:\n{schema}"
    response = llm.invoke([SystemMessage(content=prompt)])
    
    msg = (f"### Data Ingestion Summary\n{response.content}\n\n"
           "Let's build your presentation. **First, who is the target persona or audience?**")
    
    return {
        "data_summary": response.content,
        "current_step": "persona",
        "suggestions": ["Financial Analysts", "Executive Board", "Marketing Team"],
        "messages": [AIMessage(content=msg)]
    }

def process_wizard_step(state: PresentationState):
    """Steps 1-6: Processes user input and transitions to the next step dynamically."""
    step = state.get("current_step")
    user_input = state["messages"][-1].content.strip()
    schema = state["schema_info"]
    
    sugg_llm = llm.with_structured_output(SuggestionList)

    if step == "persona":
        prompt = f"Given data schema:\n{schema}\nAnd persona: {user_input}\nSuggest 3 key analytical topics or scenarios."
        topics = sugg_llm.invoke([SystemMessage(content=prompt)]).suggestions
        return {
            "persona": user_input,
            "current_step": "topics",
            "suggestions": topics,
            "messages": [AIMessage(content=f"Got it. Target: **{user_input}**.\n\nWhat specific **topics or scenarios** should we focus on?")]
        }

    elif step == "topics":
        return {
            "topics": user_input,
            "current_step": "pages",
            "suggestions": ["5 Slides", "10 Slides", "15 Slides"],
            "messages": [AIMessage(content=f"Excellent. How many **pages/slides** do you want?")]
        }

    elif step == "pages":
        prompt = f"Data schema: {schema}\nPersona: {state['persona']}\nTopics: {state['topics']}\nSuggest 3 catchy Presentation Titles."
        titles = sugg_llm.invoke([SystemMessage(content=prompt)]).suggestions
        return {
            "pages": user_input,
            "current_step": "title",
            "suggestions": titles,
            "messages": [AIMessage(content=f"Noted. What should be the **Main Title** of this deck?")]
        }

    elif step == "title":
        prompt = f"Data schema: {schema}\nSuggest 3 relevant graphs (e.g., 'Bar chart of Revenue vs Date')."
        graphs = sugg_llm.invoke([SystemMessage(content=prompt)]).suggestions
        return {
            "title": user_input,
            "current_step": "graph",
            "suggestions": graphs + ["No Graph Needed"],
            "messages": [AIMessage(content=f"Great title! What kind of **graph or visualization** would you like to add?")]
        }

    elif step == "graph":
        return {
            "graph_request": user_input,
            "current_step": "approve",
            "suggestions": ["Approve & Generate", "Restart"],
            "messages": [AIMessage(content=f"Graph noted: {user_input}.\n\n✅ Everything is ready! Click **Approve & Generate** to build the PowerPoint.")]
        }

    elif step == "approve":
        if "approve" in user_input.lower() or "generate" in user_input.lower():
            return {"current_step": "generating", "suggestions": [], "messages": [AIMessage(content="⚙️ Generating your presentation...")]}
        else:
            return {"messages": [AIMessage(content="Waiting for approval. Reply 'Approve' to proceed.")]}

def generate_presentation(state: PresentationState):
    """Final Step: Generates the PPTX and Graph."""
    prompt = f"""
    Create a structured slide deck based on:
    Schema: {state['schema_info']}
    Summary: {state['data_summary']}
    Title: {state['title']}
    Audience: {state['persona']}
    Key Topics: {state['topics']}
    Target Slides: {state['pages']}
    """
    deck_data = llm.with_structured_output(PresentationDeck).invoke([SystemMessage(content=prompt)])
    
    # Generate Graph Specs based on user request
    graph_prompt = f"Schema: {state['schema_info']}\nUser requested graph: {state['graph_request']}\nDetermine chart config."
    chart_spec = llm.with_structured_output(ChartSpecification).invoke([SystemMessage(content=graph_prompt)])
    
    df = pd.read_json(io.StringIO(state["dataframe_json"]))
    chart_bytes = io.BytesIO()
    has_plot = False
    
    try:
        if "no graph" not in state['graph_request'].lower():
            plt.figure(figsize=(8, 4.5))
            if chart_spec.chart_type == 'bar':
                df.head(10).plot(kind='bar', x=chart_spec.x_column, y=chart_spec.y_column, ax=plt.gca(), color="#2563eb")
            elif chart_spec.chart_type == 'scatter':
                df.plot(kind='scatter', x=chart_spec.x_column, y=chart_spec.y_column, ax=plt.gca(), color="#ef4444")
            else:
                df.head(10).plot(kind='line', x=chart_spec.x_column, y=chart_spec.y_column, ax=plt.gca(), linewidth=2.5)
                
            plt.title(chart_spec.chart_title, fontweight='bold')
            plt.tight_layout()
            plt.savefig(chart_bytes, format='png', dpi=300)
            chart_bytes.seek(0)
            plt.close()
            has_plot = True
    except Exception as e:
        print(f"Graph Error: {e}")

    # Build PPTX
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = state.get('title', 'Generated Report')
    slide.placeholders[1].text = f"Prepared for: {state.get('persona', 'Team')}"
    
    for slide_data in deck_data.slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = slide_data.title
        tf = slide.placeholders[1].text_frame
        for i, bullet in enumerate(slide_data.bullet_points):
            if i == 0: tf.text = bullet
            else: tf.add_paragraph().text = bullet
                
    if has_plot:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = chart_spec.chart_title
        slide.shapes.add_picture(chart_bytes, Inches(1.0), Inches(2.0), width=Inches(8.0))
        
    output_stream = io.BytesIO()
    prs.save(output_stream)
    
    return {
        "output_ready": True,
        "current_step": "done",
        "final_pptx_bytes": output_stream.getvalue(),
        "messages": [AIMessage(content="🎉 **Success!** Your presentation is fully compiled. You can download it using the button in the sidebar.")]
    }

# --- 4. State Machine Routing ---
def router(state: PresentationState):
    if state["current_step"] == "generating": return "generate_presentation"
    if state["current_step"] == "done": return END
    return "process_wizard_step"

workflow = StateGraph(PresentationState)
workflow.add_node("ingest_and_summarize", ingest_and_summarize)
workflow.add_node("process_wizard_step", process_wizard_step)
workflow.add_node("generate_presentation", generate_presentation)

workflow.add_edge(START, "ingest_and_summarize")
workflow.add_edge("ingest_and_summarize", END) # Pause after init

# When user inputs, we process step, then route either to generate or pause again
workflow.add_conditional_edges("process_wizard_step", router, {"process_wizard_step": END, "generate_presentation": "generate_presentation", END: END})
workflow.add_edge("generate_presentation", END)

memory = MemorySaver()
app_engine = workflow.compile(checkpointer=memory)
