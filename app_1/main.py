import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
import io

# Load environment variables
load_dotenv()

# Import our LangGraph engine
from agent import app_engine

app = FastAPI(title="Presentation API")

# Serve the frontend HTML
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """Handles file upload and initiates the LangGraph session."""
    
    # Read file directly into pandas
    contents = await file.read()
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
        
    # Generate unique session ID for this user
    session_id = str(uuid.uuid4())
    
    schema_str = f"Columns: {df.columns.tolist()}\nTypes: {df.dtypes.to_dict()}\nShape: {df.shape}"
    
    initial_state = {
        "schema_info": schema_str,
        "dataframe_json": df.to_json(),
        "missing_params": ["commentary_type", "persona", "topics", "pages", "title"],
        "presentation_params": {},
        "output_ready": False,
        "messages": []
    }
    
    config = {"configurable": {"thread_id": session_id}}
    
    # Trigger initial node
    response_msg = ""
    for event in app_engine.stream(initial_state, config):
        for value in event.values():
            if "messages" in value:
                response_msg = value["messages"][-1].content
                
    return {"session_id": session_id, "message": response_msg}

@app.websocket("/ws/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """Handles real-time LangGraph conversation."""
    await websocket.accept()
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        while True:
            user_input = await websocket.receive_text()
            
            # Stream response back
            for event in app_engine.stream({"messages": [HumanMessage(content=user_input)]}, config):
                for value in event.values():
                    if "messages" in value:
                        bot_response = value["messages"][-1].content
                        await websocket.send_json({"type": "message", "content": bot_response})
                        
            # Check if PPTX is ready
            state = app_engine.get_state(config).values
            if state.get("output_ready"):
                await websocket.send_json({"type": "ready"})
                
    except WebSocketDisconnect:
        pass

@app.get("/api/download/{session_id}")
async def download_presentation(session_id: str):
    """Retrieves the generated PPTX bytes from LangGraph memory."""
    config = {"configurable": {"thread_id": session_id}}
    state = app_engine.get_state(config).values
    
    pptx_bytes = state.get("final_pptx_bytes")
    if not pptx_bytes:
        return {"error": "Presentation not generated yet."}
        
    title = state.get("presentation_params", {}).get("title", "Presentation").replace(" ", "_")
    
    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f"attachment; filename={title}.pptx"}
    )
