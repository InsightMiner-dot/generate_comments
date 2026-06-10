import uuid
import io
import pandas as pd
from fastapi import UploadFile, File

@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """Handles file upload, analyzes data quality, and initiates the session."""
    contents = await file.read()
    
    # Read into Pandas
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
        
    session_id = str(uuid.uuid4())
    
    # --- Step 1: Calculate Data Quality Metrics ---
    metrics = {
        "rows": f"{len(df):,}",
        "columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_features": len(df.select_dtypes(include=['number']).columns),
        "categorical_features": len(df.select_dtypes(exclude=['number']).columns),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    }
    
    # --- Step 2: Extract Data Preview (First 10 rows) ---
    preview_df = df.head(10).fillna("NaN") # Clean NaNs for JSON serialization
    preview_data = preview_df.to_dict(orient="records")
    columns = preview_df.columns.tolist()
    
    # --- Step 3: Initialize LangGraph State ---
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
    
    # Trigger initial LangGraph node in the background
    for event in app_engine.stream(initial_state, config):
        pass 
                
    return {
        "session_id": session_id,
        "filename": file.filename,
        "metrics": metrics,
        "columns": columns,
        "preview": preview_data
    }
