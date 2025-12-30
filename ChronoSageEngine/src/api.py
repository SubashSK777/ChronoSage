import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.utils import load_config, setup_logging
from src.ingestion import DataIngestion
from src.causal_model import CausalGraphBuilder
from src.simulation import ScenarioSimulator
from src.automation import Pipeline, MonitoringService

logger = logging.getLogger("ChronoSage.API")

app = FastAPI(
    title="ChronoSage API",
    description="Predictive Intelligence Engine with Causal Modeling",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = load_config()
pipeline = Pipeline()
monitoring = MonitoringService(config)

class InterventionRequest(BaseModel):
    intervention: Dict[str, float]
    use_baseline: bool = True

class CounterfactualRequest(BaseModel):
    outcome_variable: str
    desired_value: float
    candidate_interventions: List[str]

class MultipleScenarioRequest(BaseModel):
    scenarios: List[Dict[str, float]]

class SensitivityRequest(BaseModel):
    variable: str
    min_value: float
    max_value: float
    num_points: int = 10

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "ChronoSage API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/simulate",
            "/counterfactual",
            "/scenarios",
            "/sensitivity",
            "/graph/structure",
            "/pipeline/status"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        health = pipeline.health_check()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
async def simulate_intervention(request: InterventionRequest):
    """Simulate the effect of an intervention."""
    try:
        logger.info(f"Received simulation request: {request.intervention}")
        
        if pipeline.simulator.baseline_data is None and request.use_baseline:
            raise HTTPException(
                status_code=400, 
                detail="No baseline data available. Please upload data first or set use_baseline=False"
            )
        
        result = pipeline.simulator.simulate_intervention(request.intervention)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/counterfactual")
async def run_counterfactual(request: CounterfactualRequest):
    """Run counterfactual analysis to find interventions for desired outcome."""
    try:
        logger.info(f"Counterfactual request: {request.outcome_variable} = {request.desired_value}")
        
        result = pipeline.simulator.run_counterfactual(
            request.outcome_variable,
            request.desired_value,
            request.candidate_interventions
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Counterfactual analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scenarios")
async def simulate_multiple_scenarios(request: MultipleScenarioRequest):
    """Simulate multiple scenarios and compare results."""
    try:
        logger.info(f"Simulating {len(request.scenarios)} scenarios")
        
        results = pipeline.simulator.simulate_multiple_scenarios(request.scenarios)
        
        comparison = pipeline.simulator.compare_scenarios(results)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "num_scenarios": len(results),
            "results": results,
            "comparison": comparison
        }
        
    except Exception as e:
        logger.error(f"Multiple scenario simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sensitivity")
async def sensitivity_analysis(request: SensitivityRequest):
    """Perform sensitivity analysis for a variable."""
    try:
        import numpy as np
        
        logger.info(f"Sensitivity analysis for {request.variable}")
        
        value_range = np.linspace(
            request.min_value, 
            request.max_value, 
            request.num_points
        )
        
        result = pipeline.simulator.sensitivity_analysis(request.variable, value_range)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/structure")
async def get_graph_structure():
    """Get causal graph structure."""
    try:
        structure = pipeline.causal_builder.get_graph_structure()
        is_valid, issues = pipeline.causal_builder.validate_graph()
        
        return {
            "status": "success",
            "structure": structure,
            "validation": {
                "is_valid": is_valid,
                "issues": issues
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/node/{node_name}")
async def get_node_info(node_name: str):
    """Get information about a specific node in the causal graph."""
    try:
        if node_name not in pipeline.causal_builder.graph:
            raise HTTPException(status_code=404, detail=f"Node '{node_name}' not found")
        
        parents = pipeline.causal_builder.get_parent_nodes(node_name)
        children = pipeline.causal_builder.get_child_nodes(node_name)
        markov_blanket = pipeline.causal_builder.get_markov_blanket(node_name)
        
        return {
            "node": node_name,
            "parents": parents,
            "children": children,
            "markov_blanket": markov_blanket,
            "in_degree": len(parents),
            "out_degree": len(children)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get node info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get pipeline execution status and metrics."""
    try:
        metrics = pipeline.get_pipeline_metrics()
        
        return {
            "status": "success",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/drift")
async def check_drift():
    """Check for data drift."""
    try:
        if monitoring.baseline_stats is None:
            return {
                "status": "no_baseline",
                "message": "No baseline statistics available for drift detection"
            }
        
        alerts = monitoring.get_alerts()
        
        return {
            "status": "success",
            "alerts": alerts,
            "num_alerts": len(alerts)
        }
        
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload data file for processing."""
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        import shutil
        raw_path = config['data']['raw_path']
        os.makedirs(raw_path, exist_ok=True)
        
        file_path = os.path.join(raw_path, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        df = pipeline.ingestion.ingest_from_file(file_path)
        
        if df is not None:
            summary = pipeline.ingestion.get_data_summary(df)
            
            pipeline.simulator.set_baseline(df)
            monitoring.set_baseline(df)
            
            return {
                "status": "success",
                "message": f"File uploaded and processed successfully",
                "file": file.filename,
                "summary": summary
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to process uploaded file")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/history")
async def get_simulation_history():
    """Get history of all simulations."""
    try:
        summary = pipeline.simulator.get_simulation_summary()
        
        return {
            "status": "success",
            "summary": summary,
            "history": pipeline.simulator.simulation_history[-10:]
        }
        
    except Exception as e:
        logger.error(f"Failed to get simulation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    
    logger.info(f"Starting ChronoSage API server on {host}:{port}")
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=api_config.get('reload', False)
    )
