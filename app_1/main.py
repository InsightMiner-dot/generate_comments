import os
import uuid
import io
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage

load_dotenv()
from agent import app_engine

app = FastAPI(title="Enterprise Presentation Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f: return f.read()

@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith(".csv") else pd.read_excel(io.BytesIO(contents))
    session_id = str(uuid.uuid4())
    
    metrics = {
        "rows": f"{len(df):,}", "columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_features": len(df.select_dtypes(include=['number']).columns),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    }
    
    preview_df = df.head(15).fillna("") 
    columns = preview_df.columns.tolist()
    
    initial_state = {
        "schema_info": f"Columns: {columns}\nTypes: {df.dtypes.to_dict()}",
        "dataframe_json": df.to_json(),
        "current_step": "init",
        "output_ready": False,
        "suggestions": [],
        "messages": []
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
            user_input = await websocket.receive_text()
            for event in app_engine.stream({"messages": [HumanMessage(content=user_input)]}, config):
                for value in event.values():
                    if "messages" in value:
                        bot_response = value["messages"][-1].content
                        suggs = value.get("suggestions", [])
                        await websocket.send_json({"type": "message", "content": bot_response, "suggestions": suggs})
                        
            state = app_engine.get_state(config).values
            if state.get("output_ready"):
                chart_base64 = state.get("chart_img_base64", "")
                await websocket.send_json({"type": "ready", "chart_img": chart_base64})
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
