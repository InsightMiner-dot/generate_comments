import os
import uuid
import io
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
from agent import app_engine, llm, SlideContent, generate_chart_img_base64

app = FastAPI(title="Enterprise Presentation Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['id', 'code', 'invoice', 'zip', 'phone', 'serial', 'sl']):
            df[col] = df[col].astype(str)

    metrics = {
        "rows": f"{len(df):,}", "columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
        "numeric_features": len(df.select_dtypes(include=['number']).columns),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    }
    
    preview_df = df.head(15).fillna("") 
    columns = preview_df.columns.tolist()
    
    initial_state = {
        "schema_info": f"Columns: {columns}\nTypes: {df.dtypes.to_dict()}",
        "dataframe_json": df.to_json(),
        "current_step": "init", "output_ready": False, "suggestions": [], "messages": []
    }
    
    config = {"configurable": {"thread_id": session_id}}
    initial_msg = ""
    suggestions = []
    for event in app_engine.stream(initial_state, config):
        for value in event.values():
            if "messages" in value: initial_msg = value["messages"][-1].content
            if "suggestions" in value: suggestions = value["suggestions"]
                
    return {
        "session_id": session_id, "metrics": metrics, "columns": columns,
        "preview": preview_df.to_dict(orient="records"),
        "initial_message": initial_msg, "suggestions": suggestions
    }

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
                await websocket.send_json({"type": "message", "content": f"📝 **Slide {target_idx} manual changes saved to state.**", "suggestions": ["Approve Plan & Compile"], "draft_slides": slides})
                continue

            elif payload.get("type") == "regenerate_slide":
                target_idx = payload.get("slide_index")
                regen_prompt = f"""
                You are a granular text re-drafting assistant. Regenerate a high-density slide layout index {target_idx}.
                Dataset Foundations: {current_state.get('data_summary')}
                Target Audience Context: {current_state.get('persona')}
                Core Topic Frame: {current_state.get('topics')}
                """
                new_slide_content = llm.with_structured_output(SlideContent).invoke([SystemMessage(content=regen_prompt)])
                for s in slides:
                    if s.get("slide_index") == target_idx:
                        s["title"] = new_slide_content.title
                        s["bullet_points"] = new_slide_content.bullet_points
                        s["layout_type"] = new_slide_content.layout_type
                        s["table_headers"] = new_slide_content.table_headers
                        s["table_rows"] = new_slide_content.table_rows
                        break
                app_engine.update_state(config, {"draft_slides": slides})
                await websocket.send_json({"type": "message", "content": f"🔄 **Slide {target_idx} successfully contextually re-drafted.**", "suggestions": ["Approve Plan & Compile"], "draft_slides": slides})
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
    if not pptx_bytes: return {"error": "Presentation not ready."}
    title = state.get("title", "Insight_Report").replace(" ", "_")
    return StreamingResponse(
        io.BytesIO(pptx_bytes), media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f"attachment; filename={title}.pptx"}
    )
