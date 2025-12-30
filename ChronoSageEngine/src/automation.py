import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from src.utils import load_config, setup_logging, ensure_directory, save_results, retry_on_failure, PerformanceTimer
from src.ingestion import DataIngestion
from src.causal_model import CausalGraphBuilder
from src.simulation import ScenarioSimulator

logger = logging.getLogger("ChronoSage.Automation")

class Pipeline:
    """Automated pipeline for data processing, modeling, and simulation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline."""
        self.config = load_config(config_path)
        self.automation_config = self.config.get('automation', {})
        
        log_path = self.automation_config.get('log_path', 'logs')
        log_level = self.automation_config.get('log_level', 'INFO')
        self.logger = setup_logging(log_path, log_level)
        
        self.ingestion = DataIngestion(self.config)
        self.causal_builder = CausalGraphBuilder(self.config)
        self.simulator = ScenarioSimulator(self.causal_builder, self.config)
        
        self.retry_attempts = self.automation_config.get('retry_attempts', 3)
        self.retry_delay = self.automation_config.get('retry_delay', 5)
        
        self.pipeline_state = {
            'status': 'initialized',
            'last_run': None,
            'errors': [],
            'metrics': {}
        }
        
        logger.info("Pipeline initialized successfully")
    
    @retry_on_failure(max_attempts=3, delay=5)
    def run_full_pipeline(self, data_path: str, 
                         build_graph: bool = True,
                         run_simulations: bool = False,
                         scenarios: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
        """Run complete pipeline from data ingestion to simulation."""
        try:
            with PerformanceTimer("Full pipeline execution"):
                logger.info("=" * 80)
                logger.info("Starting ChronoSage Pipeline Execution")
                logger.info("=" * 80)
                
                results = {
                    'timestamp': datetime.now().isoformat(),
                    'steps': {}
                }
                
                df = self._step_ingest_data(data_path)
                if df is None:
                    raise Exception("Data ingestion failed")
                results['steps']['ingestion'] = {'status': 'success', 'rows': len(df)}
                
                df_processed = self._step_preprocess_data(df)
                results['steps']['preprocessing'] = {'status': 'success', 'rows': len(df_processed)}
                
                if build_graph:
                    graph = self._step_build_causal_graph(df_processed)
                    results['steps']['causal_modeling'] = {
                        'status': 'success',
                        'nodes': graph.number_of_nodes(),
                        'edges': graph.number_of_edges()
                    }
                
                if run_simulations and scenarios:
                    sim_results = self._step_run_simulations(df_processed, scenarios)
                    results['steps']['simulations'] = {
                        'status': 'success',
                        'num_scenarios': len(sim_results)
                    }
                    results['simulation_results'] = sim_results
                
                self.pipeline_state['status'] = 'completed'
                self.pipeline_state['last_run'] = datetime.now()
                self.pipeline_state['metrics'] = results
                
                logger.info("=" * 80)
                logger.info("Pipeline completed successfully")
                logger.info("=" * 80)
                
                return results
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.pipeline_state['status'] = 'failed'
            self.pipeline_state['errors'].append(str(e))
            raise
    
    def _step_ingest_data(self, data_path: str) -> Optional[pd.DataFrame]:
        """Step 1: Ingest data from source."""
        logger.info("Step 1: Data Ingestion")
        logger.info("-" * 40)
        
        df = self.ingestion.ingest_from_file(data_path)
        
        if df is not None:
            summary = self.ingestion.get_data_summary(df)
            logger.info(f"Data summary: {summary['shape'][0]} rows, {summary['shape'][1]} columns")
        
        return df
    
    def _step_preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Preprocess and clean data."""
        logger.info("Step 2: Data Preprocessing")
        logger.info("-" * 40)
        
        df_processed = self.ingestion.preprocess(df, save=True)
        
        logger.info("Data preprocessing completed")
        return df_processed
    
    def _step_build_causal_graph(self, df: pd.DataFrame) -> Any:
        """Step 3: Build causal graph."""
        logger.info("Step 3: Causal Graph Construction")
        logger.info("-" * 40)
        
        graph = self.causal_builder.build_graph_from_data(df, method='correlation')
        
        is_valid, issues = self.causal_builder.validate_graph()
        if not is_valid:
            logger.warning(f"Graph validation issues: {issues}")
        
        self.causal_builder.save_graph()
        
        structure = self.causal_builder.get_graph_structure()
        logger.info(f"Graph structure: {structure['num_nodes']} nodes, {structure['num_edges']} edges")
        
        return graph
    
    def _step_run_simulations(self, df: pd.DataFrame, 
                             scenarios: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Step 4: Run scenario simulations."""
        logger.info("Step 4: Scenario Simulation")
        logger.info("-" * 40)
        
        self.simulator.set_baseline(df)
        
        results = self.simulator.simulate_multiple_scenarios(scenarios)
        
        logger.info(f"Completed {len(results)} simulations")
        
        return results
    
    def run_data_ingestion_only(self, data_path: str) -> Optional[pd.DataFrame]:
        """Run only data ingestion step."""
        logger.info("Running data ingestion only")
        
        df = self._step_ingest_data(data_path)
        
        return df
    
    def run_modeling_only(self, df: pd.DataFrame) -> Any:
        """Run only causal modeling step."""
        logger.info("Running causal modeling only")
        
        graph = self._step_build_causal_graph(df)
        
        return graph
    
    def health_check(self) -> Dict[str, Any]:
        """Perform pipeline health check."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'checks': {}
        }
        
        try:
            config_exists = os.path.exists('config.yaml')
            health['checks']['config_file'] = 'pass' if config_exists else 'fail'
            
            data_dir_exists = os.path.exists(self.config['data']['raw_path'])
            health['checks']['data_directory'] = 'pass' if data_dir_exists else 'fail'
            
            health['checks']['ingestion_module'] = 'pass' if self.ingestion else 'fail'
            health['checks']['causal_model_module'] = 'pass' if self.causal_builder else 'fail'
            health['checks']['simulator_module'] = 'pass' if self.simulator else 'fail'
            
            health['pipeline_state'] = self.pipeline_state['status']
            
            if all(v == 'pass' for v in health['checks'].values()):
                health['status'] = 'healthy'
            else:
                health['status'] = 'unhealthy'
            
            logger.info(f"Health check: {health['status']}")
            
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline execution metrics."""
        return {
            'state': self.pipeline_state,
            'config': {
                'retry_attempts': self.retry_attempts,
                'retry_delay': self.retry_delay,
                'log_level': self.automation_config.get('log_level')
            }
        }
    
    def reset_pipeline(self) -> None:
        """Reset pipeline state."""
        self.pipeline_state = {
            'status': 'initialized',
            'last_run': None,
            'errors': [],
            'metrics': {}
        }
        logger.info("Pipeline reset completed")

class MonitoringService:
    """Service for monitoring pipeline health and model performance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize monitoring service."""
        self.config = config or load_config()
        self.monitoring_config = self.config.get('monitoring', {})
        
        self.drift_detection_enabled = self.monitoring_config.get('enable_drift_detection', True)
        self.drift_threshold = self.monitoring_config.get('drift_threshold', 0.1)
        
        self.baseline_stats = None
        self.alerts = []
        
        logger.info("MonitoringService initialized")
    
    def set_baseline(self, df: pd.DataFrame) -> None:
        """Set baseline statistics for drift detection."""
        self.baseline_stats = {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
            'shape': df.shape
        }
        logger.info("Baseline statistics set for monitoring")
    
    def detect_drift(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift from baseline."""
        if not self.drift_detection_enabled or self.baseline_stats is None:
            return {'drift_detected': False, 'message': 'Drift detection not enabled or no baseline'}
        
        drift_report = {
            'drift_detected': False,
            'variables_with_drift': [],
            'drift_scores': {}
        }
        
        current_stats = {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict()
        }
        
        for col in self.baseline_stats['mean'].keys():
            if col in current_stats['mean']:
                baseline_mean = self.baseline_stats['mean'][col]
                current_mean = current_stats['mean'][col]
                baseline_std = self.baseline_stats['std'][col]
                
                if baseline_std > 0:
                    drift_score = abs(current_mean - baseline_mean) / baseline_std
                    drift_report['drift_scores'][col] = drift_score
                    
                    if drift_score > self.drift_threshold:
                        drift_report['drift_detected'] = True
                        drift_report['variables_with_drift'].append(col)
        
        if drift_report['drift_detected']:
            alert = f"Data drift detected in {len(drift_report['variables_with_drift'])} variables"
            self.alerts.append({'timestamp': datetime.now(), 'message': alert, 'type': 'drift'})
            logger.warning(alert)
        
        return drift_report
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get all monitoring alerts."""
        return self.alerts
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []
        logger.info("Monitoring alerts cleared")
