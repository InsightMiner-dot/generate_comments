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

# Mount static files to serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """Profiles the dataset and initializes the LangGraph state."""
    contents = await file.read()
    
    # Read into Pandas safely
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Failed to parse file: {str(e)}"}
        
    session_id = str(uuid.uuid4())
    
    # Calculate Enterprise Data Quality Metrics
    metrics = {
        "rows": f"{len(df):,}",
        "columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_features": len(df.select_dtypes(include=['number']).columns),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    }
    
    # Extract Dense Data Preview (15 rows, clean NaNs for JSON)
    preview_df = df.head(15).fillna("") 
    preview_data = preview_df.to_dict(orient="records")
    columns = preview_df.columns.tolist()
    
    # Initialize LangGraph State
    schema_str = f"Columns: {columns}\nTypes: {df.dtypes.to_dict()}"
    initial_state = {
        "schema_info": schema_str,
        "dataframe_json": df.to_json(),
        "missing_params": ["commentary_type", "persona", "topics", "pages", "title"],
        "presentation_params": {},
        "output_ready": False,
        "messages": []
    }
    
    config = {"configurable": {"thread_id": session_id}}
    
    # Run first node to generate the initial greeting
    initial_msg = ""
    for event in app_engine.stream(initial_state, config):
        for value in event.values():
            if "messages" in value:
                initial_msg = value["messages"][-1].content
                
    return {
        "session_id": session_id,
        "filename": file.filename,
        "metrics": metrics,
        "columns": columns,
        "preview": preview_data,
        "initial_message": initial_msg
    }

@app.websocket("/ws/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """Manages the interactive presentation builder loop."""
    await websocket.accept()
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        while True:
            user_input = await websocket.receive_text()
            
            # Stream the LangGraph execution
            for event in app_engine.stream({"messages": [HumanMessage(content=user_input)]}, config):
                for value in event.values():
                    if "messages" in value:
                        bot_response = value["messages"][-1].content
                        await websocket.send_json({"type": "message", "content": bot_response})
                        
            # Check if workflow is complete
            state = app_engine.get_state(config).values
            if state.get("output_ready"):
                await websocket.send_json({"type": "ready"})
                
    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected.")

@app.get("/api/download/{session_id}")
async def download_presentation(session_id: str):
    """Serves the generated PowerPoint file."""
    config = {"configurable": {"thread_id": session_id}}
    state = app_engine.get_state(config).values
    
    pptx_bytes = state.get("final_pptx_bytes")
    if not pptx_bytes:
        return {"error": "Presentation not ready."}
        
    title = state.get("presentation_params", {}).get("title", "Insight_Report").replace(" ", "_")
    
    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f"attachment; filename={title}.pptx"}
    )
