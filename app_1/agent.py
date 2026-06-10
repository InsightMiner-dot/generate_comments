import os
import io
import base64
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
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

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
    bullet_points: List[str] = Field(description="3-5 impactful bullet points")

class ChartSpecification(BaseModel):
    chart_title: str = Field(description="Title of the chart")
    x_column: str = Field(description="Exact column for X-axis from schema")
    y_column: str = Field(description="Exact column for Y-axis from schema")
    chart_type: str = Field(description="'bar', 'line', or 'scatter'")
    key_takeaways: List[str] = Field(description="3 context-aware bullets explaining what this chart demonstrates from data summary.")

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
    chart_img_base64: str

# --- Color Palette Constants ---
NAVY_PRIMARY = (15, 23, 42)     # #0F172A
BLUE_ACCENT = (37, 99, 235)     # #2563EB
SLATE_TEXT = (71, 85, 105)     # #475569

def add_styled_text(tf, text, size_pt, bold=False, color_rgb=NAVY_PRIMARY, is_first=False, space_after=12):
    """Helper utility to ensure uniform formatting and block text overflow."""
    p = tf.paragraphs[0] if is_first else tf.add_paragraph()
    p.text = text
    p.font.name = 'Segoe UI'
    p.font.size = Pt(size_pt)
    p.font.bold = bold
    p.font.color.rgb = RGBColor(*color_rgb)
    p.space_after = Pt(space_after)
    return p

# --- 3. Graph Nodes ---

def ingest_and_summarize(state: PresentationState):
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
    
    # Generate Graph Specs alongside analytical explanation bullets
    graph_prompt = f"Schema: {state['schema_info']}\nSummary Context: {state['data_summary']}\nUser requested graph: {state['graph_request']}\nDetermine chart config and provide key analysis insights."
    chart_spec = llm.with_structured_output(ChartSpecification).invoke([SystemMessage(content=graph_prompt)])
    
    df = pd.read_json(io.StringIO(state["dataframe_json"]))
    chart_bytes = io.BytesIO()
    has_plot = False
    chart_base64 = ""
    
    try:
        if "no graph" not in state['graph_request'].lower():
            plt.figure(figsize=(7, 4.5))
            # Clean minimalistic plotting design
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if chart_spec.chart_type == 'bar':
                df.head(10).plot(kind='bar', x=chart_spec.x_column, y=chart_spec.y_column, ax=ax, color="#2563eb", width=0.6)
            elif chart_spec.chart_type == 'scatter':
                df.plot(kind='scatter', x=chart_spec.x_column, y=chart_spec.y_column, ax=ax, color="#ef4444", s=50)
            else:
                df.head(10).plot(kind='line', x=chart_spec.x_column, y=chart_spec.y_column, ax=ax, color="#2563eb", linewidth=2.5, marker='o')
                
            plt.title(chart_spec.chart_title, fontweight='bold', fontsize=12, pad=15, color="#0F172A")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(chart_bytes, format='png', dpi=300, transparent=True)
            chart_bytes.seek(0)
            chart_base64 = base64.b64encode(chart_bytes.getvalue()).decode('utf-8')
            plt.close()
            has_plot = True
    except Exception as e:
        print(f"Graph Error: {e}")

    # Build PPTX with modern Widescreen (16:9) Configuration
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # completely blank layout layout to enforce rule bounds

    # --- Slide 1: Title Slide ---
    slide = prs.slides.add_slide(blank_layout)
    tx_box = slide.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.333), Inches(3.5))
    tf = tx_box.text_frame
    tf.word_wrap = True
    add_styled_text(tf, state.get('title', 'Generated Report'), 44, bold=True, color_rgb=NAVY_PRIMARY, is_first=True, space_after=18)
    add_styled_text(tf, f"Target Target Audience: {state.get('persona', 'Enterprise Stakeholders')}", 20, bold=False, color_rgb=BLUE_ACCENT)

    # --- Core Slides Processing ---
    for slide_data in deck_data.slides:
        slide = prs.slides.add_slide(blank_layout)
        # Main Header Box
        title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.6), Inches(11.333), Inches(1.0))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        add_styled_text(tf_title, slide_data.title, 32, bold=True, color_rgb=NAVY_PRIMARY, is_first=True)
        
        # Content Body Box
        body_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(5.0))
        tf_body = body_box.text_frame
        tf_body.word_wrap = True
        for idx, bullet in enumerate(slide_data.bullet_points):
            add_styled_text(tf_body, f"•  {bullet}", 16, bold=False, color_rgb=SLATE_TEXT, is_first=(idx == 0), space_after=14)
                
    # --- Side-By-Side Visual Insights Slide ---
    if has_plot:
        slide = prs.slides.add_slide(blank_layout)
        
        # Title
        title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.6), Inches(11.333), Inches(1.0))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        add_styled_text(tf_title, chart_spec.chart_title, 32, bold=True, color_rgb=NAVY_PRIMARY, is_first=True)
        
        # Left Side Placement: Chart Graphics Frame
        slide.shapes.add_picture(chart_bytes, Inches(0.8), Inches(1.8), width=Inches(6.5))
        
        # Right Side Placement: Narrative and Analysis Framework
        explanation_box = slide.shapes.add_textbox(Inches(7.6), Inches(1.8), Inches(4.8), Inches(4.8))
        tf_explain = explanation_box.text_frame
        tf_explain.word_wrap = True
        
        add_styled_text(tf_explain, "Analytical Takeaways", 20, bold=True, color_rgb=BLUE_ACCENT, is_first=True, space_after=14)
        for idx, insight in enumerate(chart_spec.key_takeaways):
            add_styled_text(tf_explain, f"• {insight}", 15, bold=False, color_rgb=SLATE_TEXT, is_first=False, space_after=12)
        
    output_stream = io.BytesIO()
    prs.save(output_stream)
    
    return {
        "output_ready": True,
        "current_step": "done",
        "final_pptx_bytes": output_stream.getvalue(),
        "chart_img_base64": chart_base64,
        "messages": [AIMessage(content="🎉 **Success!** Your modern 16:9 widescreen presentation is compiled with side-by-side graphical explanation views.")]
    }

# --- 4. State Machine Routing ---

def start_router(state: PresentationState):
    if state.get("current_step") == "init":
        return "ingest_and_summarize"
    return "process_wizard_step"

def router(state: PresentationState):
    if state["current_step"] == "generating": return "generate_presentation"
    if state["current_step"] == "done": return END
    return "process_wizard_step"

workflow = StateGraph(PresentationState)
workflow.add_node("ingest_and_summarize", ingest_and_summarize)
workflow.add_node("process_wizard_step", process_wizard_step)
workflow.add_node("generate_presentation", generate_presentation)

workflow.add_conditional_edges(START, start_router, {"ingest_and_summarize": "ingest_and_summarize", "process_wizard_step": "process_wizard_step"})
workflow.add_edge("ingest_and_summarize", END)
workflow.add_conditional_edges("process_wizard_step", router, {"process_wizard_step": END, "generate_presentation": "generate_presentation", END: END})
workflow.add_edge("generate_presentation", END)

memory = MemorySaver()
app_engine = workflow.compile(checkpointer=memory)
