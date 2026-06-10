import os
import io
import pandas as pd
import matplotlib.pyplot as plt
from typing import TypedDict, Annotated, Dict, List, Any
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, SystemMessage
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
    temperature=0.0
)

# --- 2. Data Models & State ---
class SlideContent(BaseModel):
    title: str = Field(description="The title of the slide")
    bullet_points: List[str] = Field(description="3-5 bullet points summarizing key insights")

class ChartSpecification(BaseModel):
    chart_title: str = Field(description="Title of the chart")
    x_column: str = Field(description="Column for X-axis")
    y_column: str = Field(description="Column for Y-axis")
    chart_type: str = Field(description="'bar', 'line', or 'scatter'")

class PresentationDeck(BaseModel):
    slides: List[SlideContent]
    chart_suggestion: ChartSpecification

class PresentationState(TypedDict):
    messages: Annotated[list, add_messages]
    schema_info: str
    data_summary: str
    missing_params: List[str]
    presentation_params: Dict[str, str]
    dataframe_json: str  
    output_ready: bool
    final_pptx_bytes: bytes

# --- 3. Graph Nodes ---
def ingest_and_summarize(state: PresentationState):
    schema = state["schema_info"]
    prompt = f"Analyze this dataset schema and provide a brief summary:\n{schema}"
    response = llm.invoke([SystemMessage(content=prompt)])
    
    msg = (f"**Data Summary:**\n{response.content}\n\n"
           "What type of **commentary** style are you looking for (e.g., Executive Summary, Deep Dive)?")
    return {"data_summary": response.content, "messages": [AIMessage(content=msg)]}

def gather_parameters(state: PresentationState):
    missing = list(state.get("missing_params", []))
    params = dict(state.get("presentation_params", {}))
    last_message = state["messages"][-1].content
    
    if "commentary_type" in missing:
        params["commentary_type"] = last_message
        missing.remove("commentary_type")
        next_question = "Understood. Who is the target **persona or audience**?"
    elif "persona" in missing:
        params["persona"] = last_message
        missing.remove("persona")
        next_question = "Got it. What **topics or metrics** should we emphasize?"
    elif "topics" in missing:
        params["topics"] = last_message
        missing.remove("topics")
        next_question = "Perfect. How many total **slides** would you like?"
    elif "pages" in missing:
        params["pages"] = last_message
        missing.remove("pages")
        next_question = "Finally, what should be the **Main Title** of the presentation?"
    elif "title" in missing:
        params["title"] = last_message
        missing.remove("title")
        next_question = "Everything is gathered! Reply with **'Approve'** to generate the deck."
    else:
        next_question = "Reply with **'Approve'** to generate the deck."

    return {"presentation_params": params, "missing_params": missing, "messages": [AIMessage(content=next_question)]}

def generate_presentation_content(state: PresentationState):
    params = state["presentation_params"]
    
    prompt = f"""
    Create a structured slide deck and visualization based on:
    Schema: {state['schema_info']}
    Summary: {state['data_summary']}
    Title: {params.get('title')}
    Commentary Type: {params.get('commentary_type')}
    Audience: {params.get('persona')}
    Key Topics: {params.get('topics')}
    Target Slides: {params.get('pages')}
    """
    
    structured_llm = llm.with_structured_output(PresentationDeck)
    deck_data = structured_llm.invoke([SystemMessage(content=prompt)])
    
    df = pd.read_json(io.StringIO(state["dataframe_json"]))
    chart_spec = deck_data.chart_suggestion
    chart_bytes = io.BytesIO()
    has_plot = False
    
    try:
        plt.figure(figsize=(8, 4.5))
        if chart_spec.chart_type == 'bar':
            df.head(10).plot(kind='bar', x=chart_spec.x_column, y=chart_spec.y_column, ax=plt.gca(), color="#2c3e50")
        elif chart_spec.chart_type == 'scatter':
            df.plot(kind='scatter', x=chart_spec.x_column, y=chart_spec.y_column, ax=plt.gca(), color="#e74c3c")
        else:
            df.head(10).plot(kind='line', x=chart_spec.x_column, y=chart_spec.y_column, ax=plt.gca(), linewidth=2.5)
            
        plt.title(chart_spec.chart_title, fontweight='bold')
        plt.tight_layout()
        plt.savefig(chart_bytes, format='png', dpi=300)
        chart_bytes.seek(0)
        plt.close()
        has_plot = True
    except Exception as e:
        print(f"Chart error: {e}")

    # Build PPTX
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = params.get('title', 'Generated Report')
    slide.placeholders[1].text = f"Prepared for: {params.get('persona', 'Team')}"
    
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
        "final_pptx_bytes": output_stream.getvalue(),
        "messages": [AIMessage(content="✅ **Success!** Your presentation is compiled and ready to download.")]
    }

def route_step(state: PresentationState):
    if len(state.get("missing_params", [])) > 0: return "gather_parameters"
    if state["messages"][-1].content.strip().lower() == "approve": return "generate_presentation_content"
    return END

# --- 4. Compilation ---
workflow = StateGraph(PresentationState)
workflow.add_node("ingest_and_summarize", ingest_and_summarize)
workflow.add_node("gather_parameters", gather_parameters)
workflow.add_node("generate_presentation_content", generate_presentation_content)

workflow.add_edge(START, "ingest_and_summarize")
workflow.add_edge("ingest_and_summarize", "gather_parameters")
workflow.add_conditional_edges("gather_parameters", route_step, ["gather_parameters", "generate_presentation_content", END])
workflow.add_edge("generate_presentation_content", END)

memory = MemorySaver()
app_engine = workflow.compile(checkpointer=memory)
