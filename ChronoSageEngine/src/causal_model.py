import os
import pickle
import logging
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from src.utils import load_config, ensure_directory, save_results, PerformanceTimer

logger = logging.getLogger("ChronoSage.CausalModel")

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy not available, using simplified causal modeling")

class CausalGraphBuilder:
    """Build and manage causal graphs for predictive modeling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize causal graph builder."""
        self.config = config or load_config()
        self.causal_config = self.config.get('causal_model', {})
        
        self.graph = nx.DiGraph()
        self.causal_model = None
        self.treatment_effects = {}
        
        self.graph_output_path = self.causal_config.get('graph_output', 'models/causal_graph.pkl')
        ensure_directory(os.path.dirname(self.graph_output_path))
        
        logger.info("CausalGraphBuilder initialized")
    
    def build_graph_from_data(self, df: pd.DataFrame, method: str = 'correlation') -> nx.DiGraph:
        """Automatically discover causal relationships from data."""
        try:
            with PerformanceTimer("Building causal graph"):
                if method == 'correlation':
                    self.graph = self._build_from_correlation(df)
                elif method == 'pc' and DOWHY_AVAILABLE:
                    self.graph = self._build_with_pc_algorithm(df)
                else:
                    logger.warning(f"Method {method} not available, using correlation")
                    self.graph = self._build_from_correlation(df)
                
                logger.info(f"Causal graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
                return self.graph
        except Exception as e:
            logger.error(f"Error building causal graph: {e}")
            return self.graph
    
    def _build_from_correlation(self, df: pd.DataFrame, threshold: float = 0.5) -> nx.DiGraph:
        """Build causal graph based on correlation analysis."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            logger.warning("Insufficient numerical columns for correlation analysis")
            return self.graph
        
        corr_matrix = df[numerical_cols].corr().abs()
        
        graph = nx.DiGraph()
        
        for col in numerical_cols:
            graph.add_node(col)
        
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                correlation = corr_matrix.loc[col1, col2]
                
                if correlation > threshold:
                    if df[col1].std() > df[col2].std():
                        graph.add_edge(col1, col2, weight=correlation)
                    else:
                        graph.add_edge(col2, col1, weight=correlation)
        
        return graph
    
    def _build_with_pc_algorithm(self, df: pd.DataFrame) -> nx.DiGraph:
        """Build causal graph using PC algorithm (requires DoWhy)."""
        logger.info("PC algorithm not fully implemented, using correlation-based approach")
        return self._build_from_correlation(df)
    
    def add_manual_edges(self, edges: List[Tuple[str, str, Optional[float]]]) -> None:
        """Manually add causal edges to the graph."""
        for edge in edges:
            if len(edge) == 2:
                source, target = edge
                self.graph.add_edge(source, target, weight=1.0)
            elif len(edge) == 3:
                source, target, weight = edge
                self.graph.add_edge(source, target, weight=weight)
            
            logger.info(f"Added edge: {edge[0]} -> {edge[1]}")
    
    def remove_edge(self, source: str, target: str) -> None:
        """Remove a causal edge from the graph."""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            logger.info(f"Removed edge: {source} -> {target}")
    
    def get_causal_paths(self, source: str, target: str) -> List[List[str]]:
        """Get all causal paths from source to target."""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target))
            logger.info(f"Found {len(paths)} causal paths from {source} to {target}")
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.warning(f"No path found: {e}")
            return []
    
    def estimate_treatment_effect(self, df: pd.DataFrame, treatment: str, outcome: str, 
                                   confounders: Optional[List[str]] = None) -> Dict[str, float]:
        """Estimate causal effect of treatment on outcome."""
        try:
            if not DOWHY_AVAILABLE:
                logger.warning("DoWhy not available, using simple correlation")
                effect = df[[treatment, outcome]].corr().iloc[0, 1]
                return {'ate': effect, 'method': 'correlation'}
            
            if confounders is None:
                confounders = []
            
            gml_graph = self._convert_to_gml()
            
            model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                graph=gml_graph,
                common_causes=confounders
            )
            
            identified_estimand = model.identify_effect()
            
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            result = {
                'ate': estimate.value,
                'method': 'dowhy',
                'estimand': str(identified_estimand)
            }
            
            self.treatment_effects[f"{treatment}_to_{outcome}"] = result
            logger.info(f"Estimated treatment effect: {treatment} -> {outcome} = {estimate.value:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error estimating treatment effect: {e}")
            effect = df[[treatment, outcome]].corr().iloc[0, 1]
            return {'ate': effect, 'method': 'correlation_fallback', 'error': str(e)}
    
    def _convert_to_gml(self) -> str:
        """Convert NetworkX graph to GML string for DoWhy."""
        gml_lines = ["graph [", "  directed 1"]
        
        node_mapping = {node: idx for idx, node in enumerate(self.graph.nodes())}
        
        for node, idx in node_mapping.items():
            gml_lines.append(f'  node [ id {idx} label "{node}" ]')
        
        for source, target in self.graph.edges():
            source_id = node_mapping[source]
            target_id = node_mapping[target]
            gml_lines.append(f'  edge [ source {source_id} target {target_id} ]')
        
        gml_lines.append("]")
        
        return "\n".join(gml_lines)
    
    def get_graph_structure(self) -> Dict[str, Any]:
        """Get graph structure information."""
        return {
            'nodes': list(self.graph.nodes()),
            'edges': list(self.graph.edges()),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_dag': nx.is_directed_acyclic_graph(self.graph)
        }
    
    def validate_graph(self) -> Tuple[bool, List[str]]:
        """Validate causal graph structure."""
        issues = []
        
        if self.graph.number_of_nodes() == 0:
            issues.append("Graph has no nodes")
        
        if self.graph.number_of_edges() == 0:
            issues.append("Graph has no edges")
        
        if not nx.is_directed_acyclic_graph(self.graph):
            issues.append("Graph contains cycles")
            try:
                cycles = list(nx.simple_cycles(self.graph))
                issues.append(f"Found {len(cycles)} cycles")
            except:
                pass
        
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            issues.append(f"Found {len(isolated_nodes)} isolated nodes: {isolated_nodes}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Graph validation passed")
        else:
            logger.warning(f"Graph validation issues: {issues}")
        
        return is_valid, issues
    
    def save_graph(self, output_path: Optional[str] = None) -> bool:
        """Save causal graph to file."""
        try:
            path = output_path or self.graph_output_path
            ensure_directory(os.path.dirname(path))
            
            graph_data = {
                'graph': self.graph,
                'treatment_effects': self.treatment_effects,
                'structure': self.get_graph_structure()
            }
            
            with open(path, 'wb') as f:
                pickle.dump(graph_data, f)
            
            logger.info(f"Graph saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            return False
    
    def load_graph(self, input_path: Optional[str] = None) -> bool:
        """Load causal graph from file."""
        try:
            path = input_path or self.graph_output_path
            
            with open(path, 'rb') as f:
                graph_data = pickle.load(f)
            
            self.graph = graph_data.get('graph', nx.DiGraph())
            self.treatment_effects = graph_data.get('treatment_effects', {})
            
            logger.info(f"Graph loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            return False
    
    def get_parent_nodes(self, node: str) -> List[str]:
        """Get parent nodes (causes) of a given node."""
        if node in self.graph:
            return list(self.graph.predecessors(node))
        return []
    
    def get_child_nodes(self, node: str) -> List[str]:
        """Get child nodes (effects) of a given node."""
        if node in self.graph:
            return list(self.graph.successors(node))
        return []
    
    def get_markov_blanket(self, node: str) -> List[str]:
        """Get Markov blanket of a node (parents, children, and children's parents)."""
        if node not in self.graph:
            return []
        
        blanket = set()
        
        blanket.update(self.get_parent_nodes(node))
        
        children = self.get_child_nodes(node)
        blanket.update(children)
        
        for child in children:
            blanket.update(self.get_parent_nodes(child))
        
        blanket.discard(node)
        
        return list(blanket)
