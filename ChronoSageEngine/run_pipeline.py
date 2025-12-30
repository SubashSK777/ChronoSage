#!/usr/bin/env python3
"""
ChronoSage Pipeline Execution Script
Run the complete predictive intelligence pipeline
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import setup_logging, load_config
from src.automation import Pipeline
from src.ingestion import DataIngestion
from src.causal_model import CausalGraphBuilder
from src.simulation import ScenarioSimulator
from src.visualization import create_quick_plot

logger = logging.getLogger("ChronoSage")

def run_example_pipeline():
    """Run example pipeline with sample data."""
    logger.info("Running example pipeline with sample data")
    
    config = load_config()
    setup_logging(
        config['automation']['log_path'],
        config['automation']['log_level']
    )
    
    pipeline = Pipeline()
    
    sample_data_path = "data/sample/customer_behavior.csv"
    
    if not os.path.exists(sample_data_path):
        logger.error(f"Sample data not found at {sample_data_path}")
        return False
    
    example_scenarios = [
        {"marketing_spend": 3000, "product_quality": 9.0},
        {"marketing_spend": 1000, "product_quality": 8.5},
        {"marketing_spend": 2500, "product_quality": 7.5}
    ]
    
    try:
        results = pipeline.run_full_pipeline(
            data_path=sample_data_path,
            build_graph=True,
            run_simulations=True,
            scenarios=example_scenarios
        )
        
        logger.info("=" * 80)
        logger.info("PIPELINE RESULTS SUMMARY")
        logger.info("=" * 80)
        
        for step_name, step_result in results.get('steps', {}).items():
            logger.info(f"{step_name}: {step_result}")
        
        if 'simulation_results' in results:
            logger.info(f"Simulations completed: {len(results['simulation_results'])}")
        
        logger.info("=" * 80)
        logger.info("Pipeline execution completed successfully!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return False

def run_custom_pipeline(data_path: str, scenarios_file: str = None):
    """Run pipeline with custom data and scenarios."""
    logger.info(f"Running custom pipeline with data from {data_path}")
    
    pipeline = Pipeline()
    
    scenarios = None
    if scenarios_file and os.path.exists(scenarios_file):
        import json
        with open(scenarios_file, 'r') as f:
            scenarios = json.load(f)
    
    try:
        results = pipeline.run_full_pipeline(
            data_path=data_path,
            build_graph=True,
            run_simulations=bool(scenarios),
            scenarios=scenarios
        )
        
        logger.info("Custom pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Custom pipeline failed: {e}", exc_info=True)
        return False

def run_api_server():
    """Start the FastAPI server."""
    logger.info("Starting ChronoSage API server")
    
    try:
        from src.api import app
        import uvicorn
        
        config = load_config()
        api_config = config.get('api', {})
        
        host = api_config.get('host', '0.0.0.0')
        port = api_config.get('port', 5000)
        
        logger.info(f"API server will start on {host}:{port}")
        
        uvicorn.run(
            "src.api:app",
            host=host,
            port=port,
            reload=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}", exc_info=True)
        return False

def run_dashboard():
    """Start the interactive dashboard."""
    logger.info("Starting ChronoSage Dashboard")
    
    try:
        from src.visualization import DashboardBuilder
        
        dashboard = DashboardBuilder()
        dashboard.run(host="0.0.0.0", port=8050, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}", exc_info=True)
        return False

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="ChronoSage - Predictive Intelligence Engine"
    )
    
    parser.add_argument(
        '--mode',
        choices=['example', 'custom', 'api', 'dashboard'],
        default='api',
        help='Execution mode: example pipeline, custom pipeline, API server, or dashboard'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to custom data file (for custom mode)'
    )
    
    parser.add_argument(
        '--scenarios',
        type=str,
        help='Path to scenarios JSON file (for custom mode)'
    )
    
    args = parser.parse_args()
    
    config = load_config()
    setup_logging(
        config['automation']['log_path'],
        config['automation']['log_level']
    )
    
    logger.info("=" * 80)
    logger.info("ChronoSage - Predictive Intelligence Engine")
    logger.info("=" * 80)
    
    if args.mode == 'example':
        success = run_example_pipeline()
    elif args.mode == 'custom':
        if not args.data:
            logger.error("--data argument required for custom mode")
            sys.exit(1)
        success = run_custom_pipeline(args.data, args.scenarios)
    elif args.mode == 'api':
        success = run_api_server()
    elif args.mode == 'dashboard':
        success = run_dashboard()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    if success is False:
        sys.exit(1)

if __name__ == "__main__":
    main()
