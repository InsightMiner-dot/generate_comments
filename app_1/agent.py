import os
import io
import json
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
    slide_index: int = Field(description="Sequential index starting at 1")
    title: str = Field(description="Slide title capturing a concrete insight or conclusion.")
    bullet_points: List[str] = Field(description="3-5 impactful bullets containing real data metrics/figures.")

class PresentationDeck(BaseModel):
    slides: List[SlideContent]

class ChartSpecification(BaseModel):
    chart_title: str = Field(description="Title of the chart")
    x_column: str = Field(description="Exact column for X-axis from schema")
    y_column: str = Field(description="Exact column for Y-axis from schema")
    chart_type: str = Field(description="'bar', 'line', or 'scatter'")
    key_takeaways: List[str] = Field(description="3 concrete bullet points explaining what this specific data trend means.")

class PresentationState(TypedDict):
    messages: Annotated[list, add_messages]
    schema_info: str
    data_summary: str
    dataframe_json: str  
    current_step: str 
    suggestions: List[str]
    persona: str
    topics: str
    pages: str
    title: str
    graph_request: str
    draft_slides: List[Dict[str, Any]]
    output_ready: bool
    final_pptx_bytes: bytes
    chart_img_base64: str

# --- Color Palette Constants ---
NAVY_PRIMARY = (15, 23, 42)     # #0F172A
BLUE_ACCENT = (37, 99, 235)     # #2563EB
SLATE_TEXT = (71, 85, 105)     # #475569

def add_styled_text(tf, text, size_pt, bold=False, color_rgb=NAVY_PRIMARY, is_first=False, space_after=12):
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
    df = pd.read_json(io.StringIO(state["dataframe_json"]))
    
    # Cast identity columns to qualitative string features to guarantee data cleaning safety
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['id', 'code', 'invoice', 'zip', 'phone', 'account', 'serial', 'sl']):
            df[col] = df[col].astype(str)
            
    profile_summary = []
    profile_summary.append(f"Total Rows Processed: {len(df)}, Total Schema Fields: {len(df.columns)}")
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if num_cols:
        profile_summary.append("\n--- Statistical Distributions (True Math Metrics) ---")
        profile_summary.append(df[num_cols].describe().to_string())
        
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        profile_summary.append("\n--- Nominal Variables & Identifier Attributes Distributions ---")
        for col in cat_cols[:5]:
            profile_summary.append(f"Field '{col}' Frequency Matrix:\n{df[col].value_counts().head(3).to_string()}")
            
    compiled_profile = "\n".join(profile_summary)
    
    prompt = f"""
    You are an elite Business Intelligence Analyst. Review this statistical summary frame.
    Note that identity variables like Invoice IDs have been filtered out of numerical averages to ensure mathematical accuracy.
    
    Data Profile Matrix:
    {compiled_profile}
    
    Summarize major high-level findings, concentrations, totals, and outliers. Keep conclusions data-rich.
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    
    msg = (f"### 📊 Cleaned Data Profile Analytics\n{response.content}\n\n"
           "Presentation schema calibrated. **Who is the target audience for this report?**")
    
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
        prompt = f"Given data summary profile:\n{state['data_summary']}\nAnd persona: {user_input}\nSuggest 3 key analytical topics or scenarios."
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
            "suggestions": ["3 Slides", "5 Slides", "7 Slides"],
            "messages": [AIMessage(content=f"Excellent. How many **pages/slides** do you want?")]
        }

    elif step == "pages":
        prompt = f"Data Summary Context: {state['data_summary']}\nPersona: {state['persona']}\nTopics: {state['topics']}\nSuggest 3 catching, metric-focused Presentation Titles."
        titles = sugg_llm.invoke([SystemMessage(content=prompt)]).suggestions
        return {
            "pages": user_input,
            "current_step": "title",
            "suggestions": titles,
            "messages": [AIMessage(content=f"Noted. What should be the **Main Title** of this deck?")]
        }

    elif step == "title":
        prompt = f"Data Summary Context: {state['data_summary']}\nSuggest 3 relevant analytical graphs matching these statistics."
        graphs = sugg_llm.invoke([SystemMessage(content=prompt)]).suggestions
        return {
            "title": user_input,
            "current_step": "graph",
            "suggestions": graphs + ["No Graph Needed"],
            "messages": [AIMessage(content=f"Great title! What kind of **graph or visualization** would you like to add?")]
        }

    elif step == "graph":
        # AUDIENCE ADAPTATION ENGINE LAYER
        persona_lower = str(state.get("persona", "")).lower()
        if any(w in persona_lower for w in ["analyst", "analytics", "scientist", "engineer", "data"]):
            audience_guardrail = """
            CRITICAL DIRECTIVE: TARGET AUDIENCE IS A DATA ANALYST.
            - Eliminate high-level observations, qualitative fluff, or basic introductory definitions.
            - Slide titles must capture an absolute metric state or finding (e.g., 'Total Cost Peaked at $4.2M with 14% Variance' instead of 'Cost Optimization Overview').
            - Every bullet point must include explicit figures, counts, ratios, math sample boundaries, or statistical limits extracted directly from the profile summary.
            """
        else:
            audience_guardrail = """
            CRITICAL DIRECTIVE: TARGET AUDIENCE IS GENERAL MANAGEMENT.
            - Blend clear, factual quantitative data metrics with operational takeaways, business strategic insights, and summaries.
            """

        generation_prompt = f"""
        You are an expert slide deck content architect. Create a comprehensive presentation blueprint framework outline.
        
        {audience_guardrail}
        
        Data Summary Profile Context:
        {state['data_summary']}
        
        Presentation Blueprint Directives:
        - Title: {state['title']}
        - Target Audience: {state['persona']}
        - Core Objective: {state['topics']}
        - Total Slide Count: {user_input} Slides
        """
        deck_draft = llm.with_structured_output(PresentationDeck).invoke([SystemMessage(content=generation_prompt)])
        slides_dict = [slide.model_dump() for slide in deck_draft.slides]
        
        msg = ("### 🛠️ Review Slide Deck Plan Blueprint\n"
               "The proposed layout configuration framework has been compiled. "
               "You can **manually edit any slide title or bullet point directly in the outline panel on the right**, "
               "or use the chat below to make bulk changes. Click **Approve Plan & Compile** when you are ready to finalize.")
        
        return {
            "graph_request": user_input,
            "current_step": "review_slides",
            "draft_slides": slides_dict,
            "suggestions": ["Approve Plan & Compile", "Make headings punchier", "Add more data insights"],
            "messages": [AIMessage(content=msg)]
        }

    elif step == "review_slides":
        if "approve" in user_input.lower() or "compile" in user_input.lower():
            return {
                "current_step": "generating", 
                "suggestions": [], 
                "messages": [AIMessage(content="⚙️ Compilation approved. Building widescreen layouts and generating charts...")]
            }
        else:
            edit_prompt = f"""
            You are an interactive slide blueprint modifier.
            Foundation Data Metrics Context: {state['data_summary']}
            Current Slide Structure Array Layout: {json.dumps(state.get('draft_slides', []))}
            
            User Mutation Command: "{user_input}"
            
            Reconstruct the PresentationDeck slides data model matching the request. Maintain metrics and unchanged content.
            """
            updated_deck = llm.with_structured_output(PresentationDeck).invoke([SystemMessage(content=edit_prompt)])
            slides_dict = [slide.model_dump() for slide in updated_deck.slides]
            
            return {
                "draft_slides": slides_dict,
                "suggestions": ["Approve Plan & Compile"],
                "messages": [AIMessage(content=f"🔄 Blueprint updated with request: *\"{user_input}\"*. Please review the changes in the preview panel.")]
            }

def generate_presentation(state: PresentationState):
    graph_prompt = f"Schema: {state['schema_info']}\nUser requested graph: {state['graph_request']}\nDetermine chart config."
    chart_spec = llm.with_structured_output(ChartSpecification).invoke([SystemMessage(content=graph_prompt)])
    
    df = pd.read_json(io.StringIO(state["dataframe_json"]))
    chart_bytes = io.BytesIO()
    has_plot = False
    chart_base64 = ""
    
    try:
        if "no graph" not in state['graph_request'].lower():
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
            
            fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color='#f1f5f9', linestyle='-', linewidth=1)
            
            sample_df = df.head(10).copy()
            sample_df[chart_spec.x_column] = sample_df[chart_spec.x_column].astype(str)
            
            if chart_spec.chart_type == 'bar':
                bars = ax.bar(sample_df[chart_spec.x_column], sample_df[chart_spec.y_column], color="#2563eb", width=0.55, edgecolor='none')
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.01), f"{yval:,.0f}", ha='center', va='bottom', fontsize=8, color='#475569', weight='bold')
            elif chart_spec.chart_type == 'scatter':
                ax.scatter(sample_df[chart_spec.x_column], sample_df[chart_spec.y_column], color="#ef4444", s=70, alpha=0.8, edgecolors='white', linewidths=1)
            else:
                ax.plot(sample_df[chart_spec.x_column], sample_df[chart_spec.y_column], color="#2563eb", linewidth=3, marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2)
                
            ax.set_title(chart_spec.chart_title, fontweight='bold', fontsize=13, pad=18, color="#0F172A", loc='left')
            plt.xticks(rotation=35, ha='right', fontsize=9, color='#475569')
            plt.yticks(fontsize=9, color='#475569')
            plt.tight_layout()
            
            plt.savefig(chart_bytes, format='png', dpi=300, transparent=True)
            chart_bytes.seek(0)
            chart_base64 = base64.b64encode(chart_bytes.getvalue()).decode('utf-8')
            plt.close()
            has_plot = True
    except Exception as e:
        print(f"Enhanced Plotting Engine Exception: {e}")

    # Build PPTX 16:9 file structures
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] 

    # --- Slide 1: Title Slide ---
    slide = prs.slides.add_slide(blank_layout)
    tx_box = slide.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.333), Inches(3.5))
    tf = tx_box.text_frame
    tf.word_wrap = True
    add_styled_text(tf, state.get('title', 'Generated Report'), 44, bold=True, color_rgb=NAVY_PRIMARY, is_first=True, space_after=18)
    add_styled_text(tf, f"Prepared for Audience: {state.get('persona', 'Enterprise Stakeholders')}", 20, bold=False, color_rgb=BLUE_ACCENT)

    # --- Process Approved Slide Blueprint Content ---
    for raw_slide in state.get("draft_slides", []):
        slide = prs.slides.add_slide(blank_layout)
        title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.6), Inches(11.333), Inches(1.0))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        add_styled_text(tf_title, raw_slide.get("title", "Slide"), 32, bold=True, color_rgb=NAVY_PRIMARY, is_first=True)
        
        body_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(5.0))
        tf_body = body_box.text_frame
        tf_body.word_wrap = True
        for idx, bullet in enumerate(raw_slide.get("bullet_points", [])):
            add_styled_text(tf_body, f"•  {bullet}", 16, bold=False, color_rgb=SLATE_TEXT, is_first=(idx == 0), space_after=14)
                
    # --- Visual Insight Slide ---
    if has_plot:
        slide = prs.slides.add_slide(blank_layout)
        title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.6), Inches(11.333), Inches(1.0))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        add_styled_text(tf_title, chart_spec.chart_title, 32, bold=True, color_rgb=NAVY_PRIMARY, is_first=True)
        
        slide.shapes.add_picture(chart_bytes, Inches(0.6), Inches(1.8), width=Inches(6.8))
        
        explanation_box = slide.shapes.add_textbox(Inches(7.8), Inches(1.8), Inches(4.8), Inches(4.8))
        tf_explain = explanation_box.text_frame
        tf_explain.word_wrap = True
        
        add_styled_text(tf_explain, "Analytical Takeaways", 20, bold=True, color_rgb=BLUE_ACCENT, is_first=True, space_after=14)
        for insight in chart_spec.key_takeaways:
            add_styled_text(tf_explain, f"• {insight}", 15, bold=False, color_rgb=SLATE_TEXT, is_first=False, space_after=12)
        
    output_stream = io.BytesIO()
    prs.save(output_stream)
    
    return {
        "output_ready": True,
        "current_step": "done",
        "final_pptx_bytes": output_stream.getvalue(),
        "chart_img_base64": chart_base64,
        "messages": [AIMessage(content="🎉 **Success!** Your data-validated widescreen report is compiled and ready for download.")]
    }

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
