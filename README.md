# ChronoSage - Predictive Intelligence Engine

A full-stack Python application for causal modeling, what-if scenario simulations, and predictive intelligence using automated data pipelines.

## Overview

ChronoSage is a sophisticated predictive intelligence engine that combines causal inference, scenario simulation, and automated data processing to help you understand and predict the effects of interventions in complex systems.

### Key Features

- **Automated Data Ingestion**: Load data from CSV, JSON, or APIs with automated preprocessing
- **Causal Graph Construction**: Build causal relationships using DoWhy and correlation analysis
- **What-If Simulations**: Run counterfactual scenarios to predict outcomes
- **FastAPI Backend**: RESTful API for programmatic access to all features
- **Interactive Dashboard**: Real-time visualization using Plotly Dash
- **Pipeline Automation**: Automated workflows with logging, error handling, and monitoring
- **Model Drift Detection**: Monitor data and model performance over time

## Project Structure

```
ChronoSage/
├── data/
│   ├── sample/              # Sample datasets for testing
│   ├── raw/                 # Raw data storage
│   └── processed/           # Processed data output
├── src/
│   ├── __init__.py
│   ├── ingestion.py         # Data ingestion & preprocessing
│   ├── causal_model.py      # Causal graph construction
│   ├── simulation.py        # Scenario simulation engine
│   ├── automation.py        # Pipeline orchestration
│   ├── api.py               # FastAPI endpoints
│   ├── visualization.py     # Dashboard & visualizations
│   └── utils.py             # Helper functions
├── tests/                   # Unit tests
├── models/                  # Saved causal graphs
├── logs/                    # Application logs
├── notebooks/               # Jupyter notebooks for analysis
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── run_pipeline.py          # Main execution script
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.11+
- pip package manager

### Setup

1. Clone or navigate to the project directory:
```bash
cd ChronoSage
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Example Pipeline

Execute the example pipeline with sample customer behavior data:

```bash
python run_pipeline.py --mode example
```

This will:
- Load sample customer behavior data
- Build a causal graph
- Run example what-if scenarios
- Generate predictions

### 2. Start API Server

Launch the FastAPI server:

```bash
python run_pipeline.py --mode api
```

The API will be available at `http://0.0.0.0:5000`

API Documentation: `http://0.0.0.0:5000/docs`

### 3. Launch Interactive Dashboard

Start the Plotly Dash dashboard:

```bash
python run_pipeline.py --mode dashboard
```

Access the dashboard at `http://0.0.0.0:8050`

### 4. Custom Pipeline

Run the pipeline with your own data:

```bash
python run_pipeline.py --mode custom --data path/to/your/data.csv
```

## Configuration

Edit `config.yaml` to customize:

- Data paths and formats
- Causal modeling parameters
- Simulation settings
- API configuration
- Logging levels
- Monitoring thresholds

## API Usage

### Example API Requests

**Run a Simulation:**
```bash
curl -X POST "http://0.0.0.0:5000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "intervention": {"marketing_spend": 3000, "product_quality": 9.0},
    "use_baseline": true
  }'
```

**Counterfactual Analysis:**
```bash
curl -X POST "http://0.0.0.0:5000/counterfactual" \
  -H "Content-Type: application/json" \
  -d '{
    "outcome_variable": "sales",
    "desired_value": 40000,
    "candidate_interventions": ["marketing_spend", "product_quality"]
  }'
```

**Get Causal Graph:**
```bash
curl "http://0.0.0.0:5000/graph/structure"
```

## Sample Data

The project includes a sample dataset (`data/sample/customer_behavior.csv`) demonstrating causal relationships:

- **marketing_spend** → website_visits → sales
- **product_quality** → customer_satisfaction → repeat_purchases
- Multiple interconnected variables showing realistic business metrics

## Core Concepts

### Causal Modeling

ChronoSage builds causal graphs to understand relationships between variables:
- Automatic discovery using correlation analysis
- Manual edge specification for domain knowledge
- Validation and cycle detection
- Treatment effect estimation

### Scenario Simulation

Run what-if scenarios to predict outcomes:
- Intervention propagation through causal graph
- Downstream effect calculation
- Baseline comparison
- Confidence scoring

### Counterfactual Analysis

Find interventions needed to achieve desired outcomes:
- Reverse causal inference
- Recommended action suggestions
- Multi-variable optimization

## Pipeline Automation

The automation module provides:
- Retry logic for failed operations
- Comprehensive logging
- Health checks
- Performance monitoring
- Data drift detection

## Architecture

### Data Flow

1. **Ingestion** → Load and validate data
2. **Preprocessing** → Clean, transform, handle missing values
3. **Modeling** → Build causal graph
4. **Simulation** → Run scenarios and predictions
5. **Output** → API responses, visualizations, exports

### Key Components

- **DataIngestion**: Handles all data loading and preprocessing
- **CausalGraphBuilder**: Constructs and manages causal graphs
- **ScenarioSimulator**: Runs what-if simulations
- **Pipeline**: Orchestrates the complete workflow
- **MonitoringService**: Tracks model and data health

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /simulate` - Run intervention simulation
- `POST /counterfactual` - Counterfactual analysis
- `POST /scenarios` - Multiple scenario comparison
- `POST /sensitivity` - Sensitivity analysis
- `GET /graph/structure` - Get causal graph
- `GET /graph/node/{name}` - Node information
- `POST /data/upload` - Upload data file
- `GET /simulation/history` - Simulation history

## Logging

Logs are stored in the `logs/` directory with timestamps. Configure log level in `config.yaml`:

```yaml
automation:
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_path: "logs"
```

## Monitoring

Built-in monitoring features:
- **Data Drift Detection**: Alerts when data distribution changes
- **Health Checks**: Validates pipeline components
- **Performance Metrics**: Tracks execution times

## Examples

### Python Code

```python
from src.automation import Pipeline

# Initialize pipeline
pipeline = Pipeline()

# Run full pipeline
results = pipeline.run_full_pipeline(
    data_path="data/sample/customer_behavior.csv",
    build_graph=True,
    run_simulations=True,
    scenarios=[
        {"marketing_spend": 3000, "product_quality": 9.0}
    ]
)

# Check results
print(results['steps'])
```

### Simulation

```python
from src.simulation import ScenarioSimulator
from src.causal_model import CausalGraphBuilder
import pandas as pd

# Load data
df = pd.read_csv("data/sample/customer_behavior.csv")

# Build causal graph
builder = CausalGraphBuilder()
graph = builder.build_graph_from_data(df)

# Create simulator
simulator = ScenarioSimulator(builder)
simulator.set_baseline(df)

# Run simulation
result = simulator.simulate_intervention({
    "marketing_spend": 3000,
    "product_quality": 9.0
})

print(result['predictions'])
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**: Change port in `config.yaml`
   ```yaml
   api:
     port: 5001  # Use different port
   ```

3. **Missing Data**: Check that sample data exists
   ```bash
   ls data/sample/customer_behavior.csv
   ```

## Development

### Adding New Features

1. Implement feature in appropriate module (`src/`)
2. Add tests in `tests/`
3. Update configuration in `config.yaml`
4. Document in README.md

### Testing

```bash
pytest tests/
```

## Performance

- Handles datasets up to 100K rows efficiently
- Graph construction: O(n²) for correlation-based
- Simulation: O(n × m) where n = nodes, m = scenarios
- API response time: <100ms for typical requests

## License

This project is provided as-is for educational and commercial use.

## Support

For issues, questions, or contributions:
1. Check logs in `logs/` directory
2. Review configuration in `config.yaml`
3. Verify data format matches expected schema

## Acknowledgments

Built with:
- **FastAPI** - Modern web framework
- **DoWhy** - Causal inference library
- **Plotly & Dash** - Interactive visualizations
- **NetworkX** - Graph operations
- **Pandas & NumPy** - Data processing

---

**ChronoSage** - Empowering predictive intelligence through causal modeling
