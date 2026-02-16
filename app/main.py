from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import uuid
from datetime import datetime
import logging

from app.ocr_processor import ocr_processor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Receipt OCR API",
    description="API for extracting and validating information from receipts",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ValidationRequest(BaseModel):
    expected_amount: float
    expected_date: str  # YYYY-MM-DD format

class ReceiptResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    upload_time: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

class BatchValidationRequest(BaseModel):
    expected_amount: float
    expected_date: str
    files: List[str]  # List of file IDs

# Temporary storage for uploaded files (use database in production)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Receipt OCR API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "validate": "/validate",
            "validate-receipts": "/validate-receipts",
            "batch_validate": "/batch-validate"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "receipt-ocr-api"
    }

@app.post("/upload")
async def upload_receipt(
    file: UploadFile = File(...),
    expected_amount: Optional[float] = None,
    expected_date: Optional[str] = None
):
    """
    Upload and process a single receipt file.
    
    Supported file types: JPEG, PNG, PDF
    """
    try:
        # Validate file type
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{file_id}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save uploaded file
        file_size = 0
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            file_size = len(content)
        
        logger.info(f"File uploaded: {filename} ({file_size} bytes)")
        
        # Process the file
        result = ocr_processor.process_file(file_path, expected_amount, expected_date)
        
        # Prepare response
        response = {
            "id": file_id,
            "filename": file.filename,
            "file_type": file.content_type,
            "file_size": file_size,
            "upload_time": datetime.now().isoformat(),
            "status": "processed",
            "result": result
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
async def validate_receipt(
    validation_request: ValidationRequest,
    file: UploadFile = File(...)
):
    """
    Upload and validate a receipt against expected values.
    """
    try:
        # Validate file type
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Process and validate
            result = ocr_processor.process_file(
                tmp_path,
                validation_request.expected_amount,
                validation_request.expected_date
            )
            
            response = {
                "success": True,
                "validation_result": result,
                "filename": file.filename,
                "file_type": file.content_type,
                "expected_amount": validation_request.expected_amount,
                "expected_date": validation_request.expected_date
            }
            
            return JSONResponse(content=response)
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-receipts")
async def validate_receipts_v2(
    files: List[UploadFile] = File(...),
    amount: float = Form(...),
    date: str = Form(...)
):
    """
    Upload and validate up to 2 receipts.
    Optimized for the frontend requirement: sender, recipient, amount, date, match_percentage, isMatch and file metadata.
    """
    try:
        if len(files) > 2:
            raise HTTPException(
                status_code=400,
                detail="Maximum 2 files allowed"
            )
        
        results = []
        temp_files = []
        
        try:
            for file in files:
                # Validate file type
                allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']
                if file.content_type not in allowed_types:
                    continue
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                    content = await file.read()
                    file_size = len(content)
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                    temp_files.append(tmp_path)
                
                # Process and validate
                result = ocr_processor.process_file(
                    tmp_path,
                    amount,
                    date
                )
                # Attach file metadata for the response
                result["_file_info"] = {
                    "filename": file.filename,
                    "file_type": file.content_type,
                    "file_size": file_size
                }
                results.append(result)
            
            if not results:
                raise HTTPException(status_code=400, detail="No valid receipts uploaded")
            
            # Picking the best result based on validation score
            best_result = max(results, key=lambda x: x.get("validation_result", {}).get("score", 0))
            
            extracted_data = best_result.get("extracted_data", {})
            validation_result = best_result.get("validation_result") or {}
            file_info = best_result.get("_file_info", {})
            
            # Structuring the response as requested
            response = {
                "id": str(uuid.uuid4()),
                "filename": file_info.get("filename"),
                "file_type": file_info.get("file_type"),
                "file_size": file_info.get("file_size"),
                "upload_time": datetime.now().isoformat(),
                "status": "processed",
                "sender": extracted_data.get("sender") or "Not Found",
                "recipient": extracted_data.get("recipient") or "Not Found",
                "amount": amount,  # User's expected amount
                "date": date,      # User's expected date
                "match_amount": validation_result.get("matched_amount"),
                "match_date": validation_result.get("matched_date"),
                "isMatch": validation_result.get("is_valid", False),
                "match_percentage": round(validation_result.get("score", 0) * 100, 2),
                "performance": best_result.get("performance"),
                "ocr_confidence": best_result.get("ocr_confidence")
            }
            
            return JSONResponse(content=response)
            
        finally:
            # Clean up temp files
            for tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in validate-receipts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-validate")
async def batch_validate_receipts(
    files: List[UploadFile] = File(...),
    expected_amount: float = None,
    expected_date: str = None
):
    """
    Upload and validate multiple receipts at once.
    Max 10 files per request.
    """
    try:
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 files allowed per batch"
            )
        
        if expected_amount is None or expected_date is None:
            raise HTTPException(
                status_code=400,
                detail="expected_amount and expected_date are required"
            )
        
        results = []
        temp_files = []
        
        try:
            for file in files:
                # Validate file type
                allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']
                if file.content_type not in allowed_types:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": f"Unsupported file type: {file.content_type}"
                    })
                    continue
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                    temp_files.append(tmp_path)
                
                # Process and validate
                result = ocr_processor.process_file(
                    tmp_path,
                    expected_amount,
                    expected_date
                )
                
                results.append({
                    "filename": file.filename,
                    "file_type": file.content_type,
                    "status": "processed",
                    "result": result
                })
            
            # Calculate overall validation status
            all_valid = all(r.get("result", {}).get("is_valid", False) for r in results if r.get("status") == "processed")
            valid_count = sum(1 for r in results if r.get("result", {}).get("is_valid", False))
            
            response = {
                "success": True,
                "total_files": len(files),
                "processed_files": len([r for r in results if r.get("status") == "processed"]),
                "valid_files": valid_count,
                "all_valid": all_valid,
                "expected_amount": expected_amount,
                "expected_date": expected_date,
                "results": results
            }
            
            return JSONResponse(content=response)
            
        finally:
            # Clean up temp files
            for tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/extract/{file_id}")
async def get_extraction_result(file_id: str):
    """
    Get extraction result for a previously uploaded file.
    """
    # This is a placeholder - in production, you'd store results in a database
    return {"message": "This endpoint would retrieve stored results", "file_id": file_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)