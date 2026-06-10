import os
import uuid
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
from agent import app_engine, llm, SlideContent, generate_chart_img_base64

app = FastAPI(title="Enterprise Presentation Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Stateful Session Persistence Cache ---
SESSIONS_DB = {}

def compute_dataset_statistics(df: pd.DataFrame) -> list:
    """Computes mathematical profiles filtering qualitative indicators from numerical frames."""
    stats_list = []
    for col in df.columns:
        col_type = "Numeric" if pd.api.types.is_numeric_dtype(df[col]) else "Categorical/Text"
        missing_pct = f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%"
        unique_cnt = df[col].nunique()
        
        if col_type == "Numeric":
            mean_val = f"{df[col].mean():,.2f}"
            median_val = f"{df[col].median():,.2f}"
            mode_series = df[col].mode()
            mode_val = f"{mode_series.iloc[0]:,.2f}" if not mode_series.empty else "N/A"
        else:
            mean_val = "N/A"
            median_val = "N/A"
            mode_series = df[col].mode()
            mode_val = str(mode_series.iloc[0]) if not mode_series.empty else "N/A"
            
        stats_list.append({
            "column": str(col), "type": col_type, "missing": missing_pct,
            "unique": unique_cnt, "mean": mean_val, "median": median_val, "mode": mode_val
        })
    return stats_list

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f: return f.read()

@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...), sheet_name: str = Form(None)):
    contents = await file.read()
    file_bytes = io.BytesIO(contents)
    
    if file.filename.endswith(('.xlsx', '.xls')):
        xl = pd.ExcelFile(file_bytes)
        if len(xl.sheet_names) > 1 and not sheet_name:
            return {"requires_sheet_selection": True, "sheets": xl.sheet_names}
        df = pd.read_excel(file_bytes, sheet_name=sheet_name if sheet_name else xl.sheet_names[0])
    else:
        df = pd.read_csv(file_bytes)
        
    session_id = str(uuid.uuid4())
    
    # Cast structural code fields to string attributes to protect averages logic
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['id', 'code', 'invoice', 'zip', 'phone', 'serial', 'sl']):
            df[col] = df[col].astype(str)

    # Persist dataframe object into local session store cache
    SESSIONS_DB[session_id] = {"df": df, "schema_info": f"Columns: {df.columns.tolist()}\nTypes: {df.dtypes.to_dict()}"}
    
    metrics = {
        "rows": f"{len(df):,}", "columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
        "numeric_features": len(df.select_dtypes(include=['number']).columns),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    }
    
    stats_matrix = compute_dataset_statistics(df)
    preview_records = df.head(10).fillna("").to_dict(orient="records")
    
    # Kickoff baseline LangGraph state engine cycle
    initial_state = {
        "schema_info": SESSIONS_DB[session_id]["schema_info"], "dataframe_json": df.to_json(),
        "current_step": "init", "output_ready": False, "suggestions": [], "messages": []
    }
    config = {"configurable": {"thread_id": session_id}}
    
    initial_msg = ""
    suggestions = []
    for event in app_engine.stream(initial_state, config):
        for value in event.values():
            if "messages" in value: initial_msg = value["messages"][-1].content
            if "suggestions" in value: suggestions = value["suggestions"]
            
    # Capture profile foundation context within memory
    app_engine.update_state(config, {"data_summary": f"Dataset row count: {len(df)}. Numerical descriptions:\n{df.describe().to_string()}"})

    return {
        "session_id": session_id, "metrics": metrics, "columns": df.columns.tolist(),
        "preview": preview_records, "stats": stats_matrix,
        "initial_message": initial_msg, "suggestions": suggestions
    }

@app.post("/api/clean")
async def clean_dataset_pipeline(session_id: str = Form(...), action: str = Form(...)):
    """Pipeline Optimization Engine Node treating missing thresholds and variance outliers."""
    if session_id not in SESSIONS_DB:
        return {"error": "Active workspace session expired."}
        
    df = SESSIONS_DB[session_id]["df"].copy()
    
    if action == "drop_na":
        df = df.dropna()
    elif action == "fill_mean":
        num_cols = df.select_dtypes(include=['number']).columns
        for c in num_cols: df[c] = df[c].fillna(df[c].mean())
    elif action == "clip_outliers":
        num_cols = df.select_dtypes(include=['number']).columns
        for c in num_cols:
            q_low = df[c].quantile(0.01)
            q_hi = df[c].quantile(0.99)
            df[c] = df[c].clip(lower=q_low, upper=q_hi)
            
    # Update active workspace references
    SESSIONS_DB[session_id]["df"] = df
    config = {"configurable": {"thread_id": session_id}}
    app_engine.update_state(config, {
        "dataframe_json": df.to_json(),
        "data_summary": f"Cleaned Framework Matrix. Row count remaining: {len(df)}. Describe summary:\n{df.describe().to_string()}"
    })
    
    return {"success": True, "stats": compute_dataset_statistics(df), "rows": f"{len(df):,}"}

@app.post("/api/plot_lab")
async def plot_insights_laboratory(session_id: str = Form(...), column: str = Form(...), plot_type: str = Form(...)):
    """Generates immediate visualizations for Page 1 Insights Lab container."""
    if session_id not in SESSIONS_DB: return {"error": "Invalid workspace session context."}
    df = SESSIONS_DB[session_id]["df"]
    
    try:
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=200)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if plot_type == "box" and pd.api.types.is_numeric_dtype(df[column]):
            ax.boxplot(df[column].dropna(), patch_artist=True, boxprops=dict(facecolor="#eff6ff", color="#2563eb"))
            ax.set_title(f"Outlier Spread Boxplot: {column}", fontweight='bold', fontsize=10)
        else: # Default Distribution Histogram
            df[column].dropna().head(20).value_counts().plot(kind='bar', ax=ax, color="#38bdf8")
            ax.set_title(f"Distribution Matrix: {column}", fontweight='bold', fontsize=10)
            
        plt.xticks(rotation=25, ha='right', fontsize=8)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return {"success": True, "chart_img": b64_str}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    config = {"configurable": {"thread_id": session_id}}
    try:
        while True:
            payload = await websocket.receive_json()
            current_state = app_engine.get_state(config).values
            slides = list(current_state.get("draft_slides", []))
            
            if payload.get("type") == "direct_edit":
                target_idx = payload.get("slide_index")
                for s in slides:
                    if s.get("slide_index") == target_idx:
                        s["title"] = payload.get("title")
                        s["bullet_points"] = payload.get("bullet_points")
                        break
                app_engine.update_state(config, {"draft_slides": slides})
                await websocket.send_json({"type": "message", "content": f"📝 **Slide {target_idx} updates written into core checkpointer memory.**", "suggestions": ["Approve Plan & Compile Presentation"], "draft_slides": slides})
                continue

            elif payload.get("type") == "regenerate_slide":
                target_idx = payload.get("slide_index")
                regen_prompt = f"""
                You are a granular analytical template builder re-drafting Slide Index {target_idx}.
                Dataset Summary Context: {current_state.get('data_summary')}
                Target Persona Requirements Focus Guidelines: {current_state.get('persona')}
                Core Operational Focus Scenario: {current_state.get('topics')}
                """
                new_slide_content = llm.with_structured_output(SlideContent).invoke([SystemMessage(content=regen_prompt)])
                for s in slides:
                    if s.get("slide_index") == target_idx:
                        s.update(new_slide_content.model_dump())
                        break
                app_engine.update_state(config, {"draft_slides": slides})
                await websocket.send_json({"type": "message", "content": f"🔄 **Slide {target_idx} successfully re-drafted.**", "suggestions": ["Approve Plan & Compile Presentation"], "draft_slides": slides})
                continue
            
            elif payload.get("type") == "chat":
                user_input = payload.get("content")
                if current_state.get("current_step") == "review_slides" and any(w in user_input.lower() for w in ["graph", "plot", "chart"]):
                    new_b64, _ = generate_chart_img_base64(current_state, user_input)
                    if new_b64:
                        app_engine.update_state(config, {"chart_img_base64": new_b64, "graph_request": user_input})
                
                for event in app_engine.stream({"messages": [HumanMessage(content=user_input)]}, config):
                    for value in event.values():
                        if "messages" in value:
                            bot_response = value["messages"][-1].content
                            suggs = value.get("suggestions", [])
                            updated_state = app_engine.get_state(config).values
                            draft_slides = updated_state.get("draft_slides", [])
                            chart_b64 = updated_state.get("chart_img_base64", "")
                            
                            await websocket.send_json({
                                "type": "message", "content": bot_response, 
                                "suggestions": suggs, "draft_slides": draft_slides,
                                "chart_img": chart_b64
                            })
                            
                state = app_engine.get_state(config).values
                if state.get("output_ready"):
                    chart_base64 = state.get("chart_img_base64", "")
                    await websocket.send_json({"type": "ready", "chart_img": chart_base64, "suggestions": ["Yes, Start New Session"]})
    except WebSocketDisconnect: pass

@app.get("/api/download/{session_id}")
async def download_presentation(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    state = app_engine.get_state(config).values
    pptx_bytes = state.get("final_pptx_bytes")
    if not pptx_bytes: return {"error": "Presentation file structure stream not ready yet."}
    title = state.get("title", "Insight_Report").replace(" ", "_")
    return StreamingResponse(
        io.BytesIO(pptx_bytes), media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f"attachment; filename={title}.pptx"}
    )
