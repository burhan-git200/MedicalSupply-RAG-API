import tempfile
import os
import logging
import time
from contextlib import asynccontextmanager
import uvicorn

import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# from fastapi.concurrency import run_in_Athreadpool
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional, Any

from api_processor import PrescriptionAnalyzer
from policy_manager import PolicyManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management for startup and shutdown."""
    logger.info("🚀 Starting Prescription ICD Analysis API...")
    try:
        app.state.analyzer = PrescriptionAnalyzer()
        app.state.policy_manager = PolicyManager()
        logger.info("✅ Prescription analyzer and policy manager initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}", exc_info=True)
        # Ensure state is clean on failure
        app.state.analyzer = None
        app.state.policy_manager = None
        raise RuntimeError(f"Service initialization failed: {e}")
    
    yield
    
    logger.info("🛑 Shutting down Prescription ICD Analysis API...")

app = FastAPI(
    title="Prescription ICD Analysis API",
    description="Analyze prescription PDFs for ICD code and medical equipment policy verification.",
    version="1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",

    swagger_ui_parameters={"requestTimeout": 30000000}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_analyzer(request: Request) -> PrescriptionAnalyzer:
    if not hasattr(request.app.state, 'analyzer') or request.app.state.analyzer is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Analyzer not initialized")
    return request.app.state.analyzer

def get_policy_manager(request: Request) -> PolicyManager:
    if not hasattr(request.app.state, 'policy_manager') or request.app.state.policy_manager is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Policy Manager not initialized")
    return request.app.state.policy_manager

class AnalysisResponse(BaseModel):
    success: bool
    analysis_results: Optional[List[dict]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

@app.get("/", tags=["Health Check"])
async def root():
    """Root endpoint for basic health check."""
    return {"service": "Prescription ICD Analysis API", "status": "running"}

@app.post("/update-policies", tags=["Policies"])
async def update_policies(
    file: UploadFile = File(..., description="The new policy PDF document to load."),
    policy_manager: PolicyManager = Depends(get_policy_manager),
    analyzer: PrescriptionAnalyzer = Depends(get_analyzer)
):
    """
    Update the policy document and rebuild the vector store knowledge base.
    This overwrites the existing policy.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    try:
        result = await run_in_threadpool(policy_manager.update_vector_store, file)
        
        if result["success"]:
            await run_in_threadpool(analyzer.reload_components)
            logger.info(f"Policy updated successfully from file: {file.filename}")
            return JSONResponse(status_code=200, content={"message": "Policy updated successfully"})
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        logger.error(f"Error during policy update: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_prescription(
    file: UploadFile = File(..., description="The prescription PDF to be analyzed."),
    analyzer_instance: PrescriptionAnalyzer = Depends(get_analyzer)
):
    """
    Analyze a prescription PDF for ICD codes and equipment requests against the loaded policy.
    """
    start_time = time.time()
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_file_path = tmp_file.name
        
    try:
        logger.info(f"📄 Processing prescription: {file.filename}")
        
        result = await run_in_threadpool(analyzer_instance.analyze_prescription, temp_file_path)
        
        processing_time = time.time() - start_time
        result["processing_time"] = round(processing_time, 2)
        
        analysis_count = len(result.get('analysis_results', []))
        logger.info(f"✅ Analysis completed for {file.filename}: {analysis_count} items analyzed in {processing_time:.2f}s")
        
        return AnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Error processing {file.filename}: {e}", exc_info=True)
        return AnalysisResponse(success=False, error=str(e), processing_time=round(time.time() - start_time, 2))
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")