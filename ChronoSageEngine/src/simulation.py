import logging
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Union
from src.utils import load_config, ensure_directory, save_results, PerformanceTimer
from src.causal_model import CausalGraphBuilder

logger = logging.getLogger("ChronoSage.Simulation")

class ScenarioSimulator:
    """What-if scenario simulation engine using causal graphs."""
    
    def __init__(self, causal_graph: Optional[CausalGraphBuilder] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """Initialize scenario simulator."""
        self.config = config or load_config()
        self.sim_config = self.config.get('simulation', {})
        
        self.causal_graph = causal_graph or CausalGraphBuilder(config)
        self.baseline_data = None
        self.simulation_history = []
        
        self.max_iterations = self.sim_config.get('max_iterations', 1000)
        self.convergence_threshold = self.sim_config.get('convergence_threshold', 0.001)
        
        logger.info("ScenarioSimulator initialized")
    
    def set_baseline(self, df: pd.DataFrame) -> None:
        """Set baseline data for simulations."""
        self.baseline_data = df.copy()
        logger.info(f"Baseline data set with {len(df)} rows")
    
    def simulate_intervention(self, intervention: Dict[str, float], 
                             data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Simulate the effect of an intervention on the system."""
        try:
            with PerformanceTimer(f"Simulating intervention: {list(intervention.keys())}"):
                if data is None:
                    if self.baseline_data is None:
                        logger.error("No baseline data available")
                        return {}
                    data = self.baseline_data.copy()
                else:
                    data = data.copy()
                
                affected_variables = self._propagate_intervention(intervention, data)
                
                predictions = self._calculate_predictions(data, affected_variables)
                
                result = {
                    'intervention': intervention,
                    'affected_variables': affected_variables,
                    'predictions': predictions,
                    'baseline_comparison': self._compare_to_baseline(predictions)
                }
                
                self.simulation_history.append(result)
                logger.info(f"Simulation completed, affected {len(affected_variables)} variables")
                
                return result
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            return {'error': str(e)}
    
    def _propagate_intervention(self, intervention: Dict[str, float], 
                                data: pd.DataFrame) -> Dict[str, float]:
        """Propagate intervention effects through causal graph."""
        affected = {}
        
        for var, value in intervention.items():
            if var in data.columns:
                data[var] = value
                affected[var] = value
                logger.info(f"Applied intervention: {var} = {value}")
        
        graph = self.causal_graph.graph
        
        for var in intervention.keys():
            if var in graph:
                descendants = nx.descendants(graph, var)
                
                for desc in descendants:
                    if desc in data.columns:
                        effect = self._calculate_downstream_effect(var, desc, intervention[var], data)
                        affected[desc] = effect
        
        return affected
    
    def _calculate_downstream_effect(self, source: str, target: str, 
                                    intervention_value: float, data: pd.DataFrame) -> float:
        """Calculate effect on downstream variable."""
        try:
            if source not in data.columns or target not in data.columns:
                return data[target].mean() if target in data.columns else 0.0
            
            correlation = data[[source, target]].corr().iloc[0, 1]
            
            baseline_source = self.baseline_data[source].mean() if self.baseline_data is not None else data[source].mean()
            baseline_target = self.baseline_data[target].mean() if self.baseline_data is not None else data[target].mean()
            
            source_change = intervention_value - baseline_source
            
            target_std = data[target].std()
            source_std = data[source].std()
            
            if source_std > 0:
                normalized_change = source_change / source_std
                target_change = normalized_change * target_std * correlation
                new_value = baseline_target + target_change
            else:
                new_value = baseline_target
            
            return new_value
            
        except Exception as e:
            logger.warning(f"Error calculating downstream effect: {e}")
            return data[target].mean() if target in data.columns else 0.0
    
    def _calculate_predictions(self, data: pd.DataFrame, 
                              affected_variables: Dict[str, float]) -> Dict[str, Any]:
        """Calculate predictions based on simulated data."""
        predictions = {}
        
        for var, value in affected_variables.items():
            if var in data.columns:
                baseline_mean = self.baseline_data[var].mean() if self.baseline_data is not None else data[var].mean()
                baseline_std = self.baseline_data[var].std() if self.baseline_data is not None else data[var].std()
                
                predictions[var] = {
                    'predicted_value': value,
                    'baseline_value': baseline_mean,
                    'change': value - baseline_mean,
                    'percent_change': ((value - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0,
                    'z_score': ((value - baseline_mean) / baseline_std) if baseline_std > 0 else 0
                }
        
        return predictions
    
    def _compare_to_baseline(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Compare predictions to baseline statistics."""
        if self.baseline_data is None:
            return {}
        
        comparison = {
            'variables_changed': len(predictions),
            'total_change': sum(pred['change'] for pred in predictions.values()),
            'avg_percent_change': np.mean([pred['percent_change'] for pred in predictions.values()]),
            'significant_changes': []
        }
        
        for var, pred in predictions.items():
            if abs(pred['z_score']) > 2:
                comparison['significant_changes'].append({
                    'variable': var,
                    'z_score': pred['z_score'],
                    'change': pred['change']
                })
        
        return comparison
    
    def run_counterfactual(self, observed_outcome: str, 
                          desired_outcome: float,
                          candidate_interventions: List[str]) -> Dict[str, Any]:
        """Find interventions needed to achieve desired outcome."""
        try:
            logger.info(f"Running counterfactual analysis for {observed_outcome} = {desired_outcome}")
            
            if self.baseline_data is None:
                logger.error("No baseline data available")
                return {}
            
            baseline_value = self.baseline_data[observed_outcome].mean()
            required_change = desired_outcome - baseline_value
            
            results = []
            
            for var in candidate_interventions:
                if var not in self.baseline_data.columns:
                    continue
                
                correlation = self.baseline_data[[var, observed_outcome]].corr().iloc[0, 1]
                
                var_std = self.baseline_data[var].std()
                outcome_std = self.baseline_data[observed_outcome].std()
                
                if var_std > 0 and abs(correlation) > 0.1:
                    required_var_change = (required_change / outcome_std) * (var_std / correlation)
                    
                    baseline_var = self.baseline_data[var].mean()
                    suggested_value = baseline_var + required_var_change
                    
                    results.append({
                        'variable': var,
                        'suggested_value': suggested_value,
                        'current_value': baseline_var,
                        'required_change': required_var_change,
                        'correlation': correlation,
                        'confidence': abs(correlation)
                    })
            
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'target_outcome': observed_outcome,
                'desired_value': desired_outcome,
                'baseline_value': baseline_value,
                'required_change': required_change,
                'recommended_interventions': results[:5]
            }
            
        except Exception as e:
            logger.error(f"Error in counterfactual analysis: {e}")
            return {'error': str(e)}
    
    def simulate_multiple_scenarios(self, scenarios: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Simulate multiple scenarios in batch."""
        logger.info(f"Simulating {len(scenarios)} scenarios")
        
        results = []
        for i, scenario in enumerate(scenarios):
            logger.info(f"Simulating scenario {i+1}/{len(scenarios)}")
            result = self.simulate_intervention(scenario)
            results.append(result)
        
        return results
    
    def compare_scenarios(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple scenario outcomes."""
        if not scenario_results:
            return {}
        
        comparison = {
            'num_scenarios': len(scenario_results),
            'scenario_comparison': []
        }
        
        for i, result in enumerate(scenario_results):
            if 'predictions' in result:
                total_change = sum(
                    pred.get('change', 0) 
                    for pred in result['predictions'].values()
                )
                
                comparison['scenario_comparison'].append({
                    'scenario_id': i,
                    'intervention': result.get('intervention', {}),
                    'total_impact': total_change,
                    'affected_variables': len(result.get('affected_variables', {}))
                })
        
        comparison['scenario_comparison'].sort(
            key=lambda x: abs(x['total_impact']), 
            reverse=True
        )
        
        return comparison
    
    def sensitivity_analysis(self, variable: str, value_range: np.ndarray) -> Dict[str, Any]:
        """Perform sensitivity analysis for a variable."""
        logger.info(f"Running sensitivity analysis for {variable}")
        
        results = []
        
        for value in value_range:
            intervention = {variable: float(value)}
            sim_result = self.simulate_intervention(intervention)
            
            if 'predictions' in sim_result:
                results.append({
                    'input_value': float(value),
                    'predictions': sim_result['predictions']
                })
        
        return {
            'variable': variable,
            'value_range': value_range.tolist(),
            'results': results
        }
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of all simulations run."""
        return {
            'total_simulations': len(self.simulation_history),
            'unique_interventions': len(set(
                tuple(sorted(sim['intervention'].items())) 
                for sim in self.simulation_history 
                if 'intervention' in sim
            )),
            'baseline_set': self.baseline_data is not None,
            'baseline_size': len(self.baseline_data) if self.baseline_data is not None else 0
        }
