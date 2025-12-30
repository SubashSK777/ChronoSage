import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, List, Optional, Any
import networkx as nx

from src.utils import load_config
from src.causal_model import CausalGraphBuilder
from src.simulation import ScenarioSimulator
from src.automation import Pipeline

logger = logging.getLogger("ChronoSage.Visualization")

class DashboardBuilder:
    """Build interactive dashboards for ChronoSage."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dashboard builder."""
        self.config = config or load_config()
        self.dashboard_config = self.config.get('dashboard', {})
        
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title=self.dashboard_config.get('title', 'ChronoSage Dashboard')
        )
        
        self.pipeline = Pipeline()
        
        logger.info("DashboardBuilder initialized")
    
    def create_causal_graph_plot(self, graph_builder: CausalGraphBuilder) -> go.Figure:
        """Create interactive visualization of causal graph."""
        graph = graph_builder.graph
        
        if graph.number_of_nodes() == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No causal graph available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        try:
            pos = nx.spring_layout(graph, k=1, iterations=50)
        except:
            pos = nx.random_layout(graph)
        
        edge_trace = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            degree = graph.in_degree(node) + graph.out_degree(node)
            node_size.append(20 + degree * 5)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color='#1f77b4',
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title="Causal Graph Structure",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def create_simulation_comparison_plot(self, simulation_results: List[Dict[str, Any]]) -> go.Figure:
        """Create comparison plot for multiple simulation scenarios."""
        if not simulation_results:
            fig = go.Figure()
            fig.add_annotation(
                text="No simulation results available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        scenarios = []
        variables = set()
        
        for i, result in enumerate(simulation_results):
            if 'predictions' in result:
                scenarios.append(f"Scenario {i+1}")
                variables.update(result['predictions'].keys())
        
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["Scenario Comparison: Predicted Changes"]
        )
        
        for var in list(variables)[:10]:
            changes = []
            for result in simulation_results:
                if 'predictions' in result and var in result['predictions']:
                    changes.append(result['predictions'][var].get('change', 0))
                else:
                    changes.append(0)
            
            fig.add_trace(
                go.Bar(name=var, x=scenarios, y=changes)
            )
        
        fig.update_layout(
            title="Scenario Impact Comparison",
            xaxis_title="Scenario",
            yaxis_title="Change from Baseline",
            barmode='group',
            height=500
        )
        
        return fig
    
    def create_sensitivity_plot(self, sensitivity_data: Dict[str, Any]) -> go.Figure:
        """Create sensitivity analysis plot."""
        if 'results' not in sensitivity_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No sensitivity data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        variable = sensitivity_data.get('variable', 'Variable')
        results = sensitivity_data['results']
        
        fig = go.Figure()
        
        if results:
            input_values = [r['input_value'] for r in results]
            
            all_outcomes = set()
            for r in results:
                if 'predictions' in r:
                    all_outcomes.update(r['predictions'].keys())
            
            for outcome in list(all_outcomes)[:5]:
                output_values = []
                for r in results:
                    if 'predictions' in r and outcome in r['predictions']:
                        output_values.append(r['predictions'][outcome].get('predicted_value', 0))
                    else:
                        output_values.append(None)
                
                fig.add_trace(go.Scatter(
                    x=input_values,
                    y=output_values,
                    mode='lines+markers',
                    name=outcome
                ))
        
        fig.update_layout(
            title=f"Sensitivity Analysis: {variable}",
            xaxis_title=variable,
            yaxis_title="Predicted Outcome",
            height=500
        )
        
        return fig
    
    def create_data_distribution_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create distribution plots for numerical variables."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        
        if not numerical_cols:
            fig = go.Figure()
            fig.add_annotation(
                text="No numerical data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        num_cols = len(numerical_cols)
        rows = (num_cols + 2) // 3
        
        fig = make_subplots(
            rows=rows, cols=min(3, num_cols),
            subplot_titles=numerical_cols
        )
        
        for i, col in enumerate(numerical_cols):
            row = i // 3 + 1
            col_pos = i % 3 + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title="Data Distributions",
            height=300 * rows,
            showlegend=False
        )
        
        return fig
    
    def build_layout(self) -> html.Div:
        """Build dashboard layout."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("ChronoSage - Predictive Intelligence Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Pipeline Status"),
                        dbc.CardBody([
                            html.Div(id="pipeline-status", children="Not started")
                        ])
                    ])
                ], width=12, className="mb-4")
            ]),
            
            dbc.Tabs([
                dbc.Tab(label="Causal Graph", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="causal-graph-plot")
                        ], width=12)
                    ], className="mt-4")
                ]),
                
                dbc.Tab(label="Simulations", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H4("Run Simulation", className="mt-4"),
                            dbc.Input(
                                id="intervention-variable",
                                placeholder="Variable name",
                                className="mb-2"
                            ),
                            dbc.Input(
                                id="intervention-value",
                                placeholder="Intervention value",
                                type="number",
                                className="mb-2"
                            ),
                            dbc.Button(
                                "Run Simulation",
                                id="run-simulation-btn",
                                color="primary",
                                className="mb-4"
                            ),
                            html.Div(id="simulation-results")
                        ], width=12)
                    ])
                ]),
                
                dbc.Tab(label="Data Overview", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="data-distribution-plot")
                        ], width=12)
                    ], className="mt-4")
                ])
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=self.dashboard_config.get('update_interval', 5000),
                n_intervals=0
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            Output("pipeline-status", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_pipeline_status(n):
            try:
                health = self.pipeline.health_check()
                status = health.get('status', 'unknown')
                
                return html.Div([
                    html.P(f"Status: {status.upper()}", className=f"text-{'success' if status == 'healthy' else 'warning'}"),
                    html.P(f"Last updated: {health.get('timestamp', 'N/A')}", className="text-muted")
                ])
            except:
                return "Status: Unknown"
        
        @self.app.callback(
            Output("simulation-results", "children"),
            Input("run-simulation-btn", "n_clicks"),
            State("intervention-variable", "value"),
            State("intervention-value", "value"),
            prevent_initial_call=True
        )
        def run_simulation(n_clicks, variable, value):
            if not variable or value is None:
                return dbc.Alert("Please provide both variable name and value", color="warning")
            
            try:
                intervention = {variable: float(value)}
                result = self.pipeline.simulator.simulate_intervention(intervention)
                
                if 'error' in result:
                    return dbc.Alert(f"Error: {result['error']}", color="danger")
                
                predictions = result.get('predictions', {})
                
                output = [html.H5("Simulation Results", className="mt-3")]
                for var, pred in list(predictions.items())[:10]:
                    output.append(
                        dbc.Card([
                            dbc.CardBody([
                                html.H6(var),
                                html.P(f"Predicted: {pred.get('predicted_value', 0):.4f}"),
                                html.P(f"Change: {pred.get('change', 0):.4f} ({pred.get('percent_change', 0):.2f}%)")
                            ])
                        ], className="mb-2")
                    )
                
                return html.Div(output)
                
            except Exception as e:
                return dbc.Alert(f"Simulation failed: {str(e)}", color="danger")
    
    def run(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
        """Run the dashboard server."""
        self.app.layout = self.build_layout()
        self.setup_callbacks()
        
        logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

def create_quick_plot(df: pd.DataFrame, plot_type: str = "correlation") -> go.Figure:
    """Create quick visualization plots."""
    if plot_type == "correlation":
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            height=600
        )
        
        return fig
    
    elif plot_type == "scatter_matrix":
        numerical_cols = df.select_dtypes(include=[np.number]).columns[:5]
        fig = px.scatter_matrix(df, dimensions=numerical_cols)
        fig.update_layout(title="Scatter Matrix", height=800)
        return fig
    
    else:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Plot type '{plot_type}' not supported",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
