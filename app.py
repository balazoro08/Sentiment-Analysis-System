import os
import io
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentiment_engine import engine
from dataset_generator import generate_sample_csv, get_combined_dataset

app = FastAPI(
    title="Sentiment Analysis System API",
    description="NLP & Machine Learning API for Positive, Negative, Neutral Sentiment Analysis",
    version="1.0.0"
)

# Enable CORS for local development & frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schemas
class TextAnalysisRequest(BaseModel):
    text: str
    domain: Optional[str] = "general"

class BatchTextsRequest(BaseModel):
    texts: List[str]


# Mount static directory for frontend assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Sentiment Analysis API Server Running</h1><p>Frontend loading...</p>")


@app.post("/api/analyze")
async def analyze_single_text(req: TextAnalysisRequest):
    """Analyze sentiment for a single text input."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text string cannot be empty")
    
    res = engine.analyze_text(req.text)
    res["domain"] = req.domain
    return res


@app.post("/api/batch-analyze")
async def batch_analyze_texts(req: BatchTextsRequest):
    """Analyze a list of text strings in batch."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    results = []
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    total_conf = 0.0

    for idx, text in enumerate(req.texts):
        res = engine.analyze_text(text)
        res["id"] = idx + 1
        results.append(res)
        
        s_type = res["sentiment"]
        if s_type in sentiment_counts:
            sentiment_counts[s_type] += 1
        total_conf += res["confidence"]

    total = len(results)
    avg_conf = round(total_conf / total, 1) if total > 0 else 0.0

    return {
        "total_items": total,
        "summary": {
            "counts": sentiment_counts,
            "percentages": {
                k: round((v / total) * 100, 1) if total > 0 else 0.0 for k, v in sentiment_counts.items()
            },
            "average_confidence": avg_conf
        },
        "items": results
    }


@app.post("/api/batch-upload")
async def batch_upload_csv(file: UploadFile = File(...)):
    """Upload CSV or JSON file for batch sentiment processing."""
    try:
        content = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or JSON.")

        # Find text column automatically
        possible_text_cols = ["text", "review", "comment", "feedback", "content", "message", "body"]
        text_col = None
        for col in df.columns:
            if str(col).lower() in possible_text_cols:
                text_col = col
                break

        if not text_col:
            # Fallback to first string column
            for col in df.columns:
                if df[col].dtype == object or df[col].dtype == str:
                    text_col = col
                    break
        
        if not text_col:
            raise HTTPException(status_code=400, detail="Could not find a valid text column in uploaded file.")

        results = []
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        total_conf = 0.0

        for idx, row in df.iterrows():
            raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            res = engine.analyze_text(raw_text)
            
            # Preserve extra metadata if available
            row_dict = row.to_dict()
            res["id"] = row_dict.get("id", idx + 1)
            res["original_row"] = {str(k): (str(v) if pd.notna(v) else "") for k, v in row_dict.items()}
            
            results.append(res)
            
            s_type = res["sentiment"]
            if s_type in sentiment_counts:
                sentiment_counts[s_type] += 1
            total_conf += res["confidence"]

        total = len(results)
        avg_conf = round(total_conf / total, 1) if total > 0 else 0.0

        return {
            "filename": file.filename,
            "text_column_used": text_col,
            "total_items": total,
            "summary": {
                "counts": sentiment_counts,
                "percentages": {
                    k: round((v / total) * 100, 1) if total > 0 else 0.0 for k, v in sentiment_counts.items()
                },
                "average_confidence": avg_conf
            },
            "items": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/api/model/stats")
async def get_model_stats():
    """Retrieve model performance metrics, confusion matrix, and feature weights."""
    return engine.model_metrics


@app.post("/api/model/train")
async def train_model():
    """Trigger model retraining."""
    metrics = engine.train_model()
    return {"status": "success", "message": "Model successfully retrained.", "metrics": metrics}


@app.get("/api/presets")
async def get_presets():
    """Return pre-configured sample texts across domains."""
    return [
        {
            "category": "E-Commerce / Product Reviews",
            "samples": [
                {"title": "Five-Star Gadget", "text": "This wireless headset is incredible! The sound clarity and active noise cancellation are world-class.", "expected": "positive"},
                {"title": "Average Commuter Purchase", "text": "Battery life is mediocre, lasting around 4 hours. It is acceptable for commuting.", "expected": "neutral"},
                {"title": "Damaged Delivery", "text": "Arrived severely scratched and right earbud doesn't charge. Extremely unsatisfied.", "expected": "negative"}
            ]
        },
        {
            "category": "Social Media & Community",
            "samples": [
                {"title": "Sleek UI Update", "text": "Just tried the new platform update and the UI design is sleek and responsive! 👏", "expected": "positive"},
                {"title": "Maintenance Announcement", "text": "The scheduled maintenance announcement was posted earlier today.", "expected": "neutral"},
                {"title": "Broken Patch", "text": "Worst update ever! The app crashes repeatedly and logs me out continuously 😡", "expected": "negative"}
            ]
        },
        {
            "category": "Customer Support Tickets",
            "samples": [
                {"title": "Quick Ticket Resolution", "text": "Agent Sarah was super helpful, patient, and solved my billing issue in 5 minutes. Excellent service!", "expected": "positive"},
                {"title": "Routine Verification", "text": "Support requested my account details for further verification.", "expected": "neutral"},
                {"title": "Unresolved Delay", "text": "My ticket #8921 is still open after 4 business days. No status updates received.", "expected": "negative"}
            ]
        }
    ]


@app.get("/api/sample-csv")
async def get_sample_csv():
    """Generate and return sample CSV file for testing."""
    sample_file = generate_sample_csv()
    return FileResponse(sample_file, filename="sample_reviews.csv", media_type="text/csv")


if __name__ == "__main__":
    import uvicorn
    print("Starting Sentiment Analysis System Server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
