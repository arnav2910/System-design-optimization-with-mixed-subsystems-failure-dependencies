# System Reliability Optimization

An interactive tool for optimizing complex system designs to achieve specified availability targets while minimizing cost. Compares four state-of-the-art metaheuristic algorithms: Differential Evolution (DE), Manta Ray Foraging Optimization (MRFO), Shuffled Frog Leaping Algorithm (SFLA), and Multi-Objective Differential Evolution (MODE).

**[Try the Live Demo](https://systemdesignoptimization.streamlit.app/)** | **[View on GitHub](https://github.com/arnav2910/System-design-optimization-with-mixed-subsystems-failure-dependencies)**

## Table of Contents

- [System Reliability Optimization](#system-reliability-optimization)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Problem](#key-problem)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Quick Start](#quick-start)
  - [Usage](#usage)
    - [Interactive Web Interface](#interactive-web-interface)
    - [Command-Line Benchmarking](#command-line-benchmarking)
  - [Project Structure](#project-structure)
    - [Key Modules](#key-modules)
  - [Configuration](#configuration)
    - [System Data (`backend/data.py`)](#system-data-backenddatapy)
    - [Algorithm Parameters](#algorithm-parameters)
  - [Algorithm Details](#algorithm-details)
    - [Differential Evolution (DE)](#differential-evolution-de)
    - [Manta Ray Foraging Optimization (MRFO)](#manta-ray-foraging-optimization-mrfo)
    - [Shuffled Frog Leaping Algorithm (SFLA)](#shuffled-frog-leaping-algorithm-sfla)
    - [Multi-Objective Differential Evolution (MODE)](#multi-objective-differential-evolution-mode)
  - [Support](#support)
    - [Getting Help](#getting-help)
    - [Common Issues](#common-issues)

## Overview

This project solves a critical engineering problem: **How do you design a system that meets reliability requirements while minimizing cost?**

The system optimization tool addresses the Redundancy Allocation Problem (RAP) with availability constraints. Given a system composed of multiple subsystems, each subsystem can be made more reliable (and more expensive) by adding redundant components and/or improving repair capabilities. This tool automatically finds the optimal balance.

### Key Problem

- **Goal**: Minimize total system cost
- **Constraints**: Achieve minimum system availability threshold
- **Complexity**: Multiple subsystems with interdependent failure modes
- **Solution**: Apply evolutionary algorithms to find near-optimal designs

## Features

**Interactive Simulation**
- Web-based UI built with Streamlit for easy parameter adjustment
- Real-time visualization of algorithm performance
- Support for multiple system configurations with one click

**Multiple System Topologies**
- **Complex Bridge Network**: 5-subsystem configuration with bridge topology
- **Series-Parallel System**: 10-subsystem configuration combining series and parallel arrangements

**Four Optimization Algorithms**
- **Differential Evolution (DE)**: Population-based stochastic optimization
- **Manta Ray Foraging Optimization (MRFO)**: Bio-inspired algorithm mimicking manta ray behavior
- **Shuffled Frog Leaping Algorithm (SFLA)**: Hybrid evolutionary approach with memetic structure
- **Multi-Objective Differential Evolution (MODE)**: Explores the Pareto-optimal front for cost vs. availability trade-offs

**Comprehensive Analysis**
- Multi-run benchmarking with statistical comparison
- Box plot visualization showing solution quality distribution
- Convergence curve tracking algorithm performance over generations
- Detailed output including optimal component counts, repair parameters, and costs

**Model Flexibility**
- Configurable availability targets (0.0 to 0.9999)
- Adjustable population size and generation limits
- Multiple dependency models for subsystem failures
  - Linear dependency
  - Weak dependency
  - Strong dependency

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/arnav2910/System-design-optimization-with-mixed-subsystems-failure-dependencies
   cd System-design-optimization-with-mixed-subsystems-failure-dependencies
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

**Option 1: Try the Live Demo (No Installation Required)**

Visit the deployed app: [systemdesignoptimization.streamlit.app](https://systemdesignoptimization.streamlit.app/)

**Option 2: Launch Locally**
```bash
streamlit run app.py
```
Opens a local web browser at `http://localhost:8501` with the interactive optimization dashboard.

**Option 3: Run Benchmark Comparison**
```bash
python main.py
```
Runs a complete benchmark of all three algorithms against both system configurations with multiple availability targets. Results and visualizations are saved to the `results/` directory.

## Usage

### Interactive Web Interface

The Streamlit app (`app.py`) provides an intuitive interface for running optimizations:

1. **Select System Configuration**
   - Choose between "Complex Bridge Network" or "Series-Parallel System"

2. **Set Availability Targets**
   - Enter minimum system availability thresholds (comma-separated)
   - Example: `0.90, 0.95, 0.99`
   - Valid range: 0.0 to 0.9999

3. **Configure Algorithm Parameters**
   - **Number of Independent Runs**: How many times to run each algorithm (1-20)
   - **Population Size**: Number of candidate solutions per generation (10-500)
   - **Max Generations**: Maximum iterations before stopping (10-1000)

4. **Select Algorithms**
   - Toggle checkboxes to enable/disable DE, MRFO, or SFLA
   - Compare multiple algorithms in a single run

5. **View Results**
   - Box plots showing solution cost distribution across runs
   - Convergence curves demonstrating algorithm efficiency
   - Best solution parameters for each algorithm

### Command-Line Benchmarking

The benchmark script (`main.py`) evaluates all algorithms systematically:

```bash
python main.py
```

**Default Configuration:**
- Systems: Complex bridge network + Series-parallel system
- Availability targets: 0.90, 0.95, 0.99
- Runs per configuration: 10
- Population size: 100
- Generations: 200

**Output includes:**
- Console table with best costs, means, and standard deviations
- PNG visualizations saved to `results/` directory:
  - `complex_bridge_network_boxplots.png`
  - `complex_bridge_network_convergence.png`
  - `series_parallel_system_boxplots.png`
  - `series_parallel_system_convergence.png`

**Customizing the Benchmark**

Edit `main.py` to modify:
- System configurations
- Availability targets
- Algorithm parameters
- Number of runs

```python
systems = [
    {'name': 'Your System', 'm': 6, 'type': 'bridge'},
]
avail_targets = [0.85, 0.90, 0.95]
algorithms = {
    # specify which algorithms to run
}
```

## Project Structure

```
opti-final/
├── app.py                    # Streamlit interactive web application
├── main.py                   # Command-line benchmarking script
├── requirements.txt          # Python package dependencies
├── README.md                 # This file
│
├── backend/                  # Core optimization modules
│   ├── __init__.py
│   ├── algorithms.py         # DE, MRFO, SFLA implementations
│   ├── core_math.py          # Availability calculations and objective functions
│   └── data.py               # System topology and parameter definitions
│
├── utils/                    # Utility modules
│   ├── __init__.py
│   └── visualise.py          # Plotting and visualization functions
│
└── results/                  # Output directory (created automatically)
    └── [generated plots and data]
```

### Key Modules

| Module          | Purpose                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------- |
| `algorithms.py` | Implements DE, MRFO, and SFLA optimization algorithms                                    |
| `mo_algorithms.py` | Implements Multi-Objective Differential Evolution (MODE) for Pareto front exploration |
| `core_math.py`  | Calculates subsystem availability, system availability, and objective function penalties |
| `data.py`       | Stores system topology definitions and parameter ranges                                  |
| `visualise.py`  | Generates box plots and convergence curves                                               |

## Configuration

### System Data (`backend/data.py`)

Systems are defined with:
- **m**: Number of subsystems
- **Subsystem parameters**: Failure rate (λ), repair rate (μ), dependency type
- **Component bounds**: Minimum and maximum redundancy options

### Algorithm Parameters

**All Algorithms:**
- `pop_size`: Population size (default: 100)
- `max_gen`: Maximum generations (default: 200)
- `track_history`: Enable convergence tracking (default: False)

**SFLA-Specific:**
- `num_memeplexes`: Number of subpopulations (default: 5)
- `local_iters`: Local search iterations per memeplex (default: 10)

## Algorithm Details

### Differential Evolution (DE)

Gradient-free population-based optimizer using:
- Mutation operator: `x' = x + F(x_best - x) + F(x_a - x_b)`
- Binomial crossover with CR probability
- Parameter bounds: F ∈ [0.5, 1.0], CR = 0.9

**Best for**: Problems with complex, non-convex landscapes

### Manta Ray Foraging Optimization (MRFO)

Bio-inspired algorithm simulating manta ray foraging behavior:
- Chain foraging: population movement following a chain lead
- Cyclone foraging: local search with improved individuals
- Somersault tail: exploitation around best solution

**Best for**: Exploration-exploitation balance

### Shuffled Frog Leaping Algorithm (SFLA)

Hybrid evolutionary algorithm combining genetic algorithms with memetic structure:
- Population divided into memeplexes (subpopulations)
- Local evolution within memeplexes
- Periodic shuffling and reorganization

**Best for**: Avoiding local optima through population diversity

### Multi-Objective Differential Evolution (MODE)

Extracts the Pareto-optimal front optimizing for both constraints simultaneously:
- Minimizes cost while maximizing availability natively.
- Identifies the trade-off curve rather than a single optimal solution based on penalty factors.

**Best for**: Decision-making where the exact availability target vs budget is flexible and trade-offs need to be analyzed.

## Support

### Getting Help

- **Review examples**: See [Usage](#usage) section for detailed walkthroughs
- **Check code comments**: Source files contain inline documentation
- **Run verification**: Execute `main.py` to see the system in action
- **Adjust parameters**: Use the interactive app to experiment with different configurations

### Common Issues

**Problem**: "ModuleNotFoundError: No module named 'streamlit'"
- **Solution**: Run `pip install -r requirements.txt` to install all dependencies

**Problem**: "FileNotFoundError: results directory"
- **Solution**: The `results/` directory is created automatically; if missing, create it manually with `mkdir results`

**Problem**: Only infeasible solutions found
- **Solution**: Increase `max_gen`, `pop_size`, or decrease availability target

---

**Last Updated**: April 2026  
**Python Version**: 3.7+  
**Dependencies**: Streamlit, NumPy, Pandas, Matplotlib
