# Electric Vehicle Routing Problem (EVRP) Optimization

[![IEEE Paper](https://img.shields.io/badge/IEEE-Published-blue)](https://ieeexplore.ieee.org/document/10885593)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📋 Overview

This project implements intelligent algorithms for optimizing delivery routes in electric vehicle (EV) fleet systems. The research focuses on efficiently managing heterogeneous EV fleets for logistics and supply-chain management, considering real-world constraints such as battery limitations, charging station availability, and varying load capacities.

### 🎯 Key Features

- **Heterogeneous EV Fleet Management**: Supports multiple vehicle categories (Small, Medium, Large, XLarge) with different battery capacities and load limits
- **Intelligent Route Optimization**: Minimizes the longest travel time across all vehicles in the fleet
- **Charging Station Integration**: Incorporates strategic charging stops to extend vehicle range
- **Multiple Algorithm Implementations**:
  - Greedy heuristic for fast initial solutions
  - Simulated Annealing (SA) for near-optimal results
  - Parallel processing capabilities for large-scale problems

### 🔬 Research Publication

This work has been published in IEEE and presented at an international conference:

**"Intelligent Algorithm for Optimizing Delivery Time in Electric Vehicle Fleet Systems"**
*IEEE Conference Publication* - [View Paper](https://ieeexplore.ieee.org/document/10885593)

## 🏗️ Project Structure

```
ComsNets-EV-Routing/
├── 📁 Docs/                          # Algorithm documentation
│   ├── Greedy.md                     # Greedy algorithm details
│   ├── SA.md                         # Simulated Annealing algorithm
│   ├── InputAssumptions.md           # Problem constraints & parameters
│   └── InputGeneration.md            # Test case generation guide
├── 📁 Notebooks/                     # Jupyter notebooks for experiments
│   ├── Greedy.ipynb                  # Greedy algorithm implementation
│   ├── SA.ipynb                      # Simulated Annealing implementation
│   ├── Input.ipynb                   # Input generation and validation
│   ├── ResultAnalysis.ipynb          # Performance analysis
│   ├── TestCaseVisualizer.ipynb      # Route visualization
│   ├── GreedyParallelProcessing.py   # Parallel greedy implementation
│   └── SAParallelProcessing.py       # Parallel SA implementation
├── 📁 results/                       # Experimental results
│   ├── *.json                        # Detailed solution data
│   └── *.csv                         # Summary statistics
├── 📁 test_cases/                    # Benchmark test instances
│   ├── customers_10/                 # 10-customer scenarios
│   ├── customers_20/                 # 20-customer scenarios
│   ├── customers_30/                 # 30-customer scenarios
│   ├── customers_40/                 # 40-customer scenarios
│   └── customers_50/                 # 50-customer scenarios
├── requirements.txt                   # Python dependencies
└── readme.md                         # This file
```

## 🚗 Problem Formulation

### Electric Vehicle Categories

| Category | Battery (kWh) | Base Weight (kg) | Load Capacity (kg) | Range (km)* |
| -------- | ------------- | ---------------- | ------------------ | ----------- |
| Small    | 35            | 1,500            | 500                | 112-140     |
| Medium   | 40            | 1,800            | 600                | 118-148     |
| Large    | 45            | 2,000            | 700                | 126-158     |
| XLarge   | 50            | 2,200            | 800                | 135-175     |

*Range varies based on load and battery charge level (20%-80% operational range)

### Key Constraints

- **Battery Management**: Vehicles operate between 20%-80% battery capacity for optimal battery health
- **Charging Strategy**: Strategic charging station visits to extend operational range
- **Load Balancing**: Efficient distribution of customer demands across heterogeneous fleet
- **Time Optimization**: Minimize the maximum delivery time across all vehicles

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd ComsNets-EV-Routing
   ```
2. **Create virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation and analysis
- `matplotlib` & `seaborn` - Visualization
- `scipy` - Scientific computing
- `shapely` - Geometric operations
- `scikit-learn` - Machine learning utilities
- `tqdm` - Progress bars
- `ipykernel` & `ipywidgets` - Jupyter notebook support

## 🚀 Usage

### Quick Start

1. **Generate Test Cases**:

   ```bash
   jupyter notebook Notebooks/Input.ipynb
   ```
2. **Run Greedy Algorithm**:

   ```bash
   jupyter notebook Notebooks/Greedy.ipynb
   ```
3. **Run Simulated Annealing**:

   ```bash
   jupyter notebook Notebooks/SA.ipynb
   ```
4. **Analyze Results**:

   ```bash
   jupyter notebook Notebooks/ResultAnalysis.ipynb
   ```

### Parallel Processing

For large-scale experiments, use the parallel processing scripts:

```bash
# Parallel Greedy Algorithm
python Notebooks/GreedyParallelProcessing.py

# Parallel Simulated Annealing
python Notebooks/SAParallelProcessing.py
```

### Visualization

Visualize routes and analyze performance:

```bash
jupyter notebook Notebooks/TestCaseVisualizer.ipynb
```

## 📊 Algorithms

### 1. Greedy Heuristic

A fast construction algorithm that builds feasible routes by:

- Selecting nearest unserved customers
- Managing battery constraints with strategic charging
- Balancing load distribution across vehicle types

**Time Complexity**: O(n²) where n is the number of customers

### 2. Simulated Annealing (SA)

An advanced metaheuristic that improves upon greedy solutions through:

- Load-based neighbor generation
- Temperature-controlled acceptance criteria
- Iterative route optimization

**Performance**: Achieves near-optimal solutions with <25% deviation from optimal

### 3. Parallel Processing

Both algorithms support parallel execution for:

- Multiple test case evaluation
- Statistical analysis across different scenarios
- Scalability testing

## 📈 Experimental Results

The algorithms have been tested on various scenarios:

- **Test Cases**: 10-50 customers with varying demand distributions
- **Performance Metrics**:
  - Total delivery time
  - Vehicle utilization
  - Charging station usage
  - Solution quality vs. computation time

Results demonstrate that the SA algorithm consistently outperforms the greedy approach while maintaining reasonable computation times for practical applications.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite our IEEE paper:

```bibtex
@inproceedings{author2025intelligent,
  title={Intelligent Algorithm for Optimizing Delivery Time in Electric Vehicle Fleet Systems},
  author={[Author Names]},
  booktitle={IEEE Conference},
  year={2025},
  organization={IEEE},
  url={https://ieeexplore.ieee.org/document/10885593}
}
```

## 📞 Contact

For questions or collaboration opportunities, please reach out through:

- GitHub Issues
- Email: chanakyavasantha@gmail.com

---

**Keywords**: Electric Vehicle Routing, Fleet Optimization, Simulated Annealing, Logistics, Supply Chain Management, Sustainable Transportation

This README includes:

1. **Professional badges** linking to your IEEE paper
2. **Comprehensive overview** of the project and its features
3. **Clear project structure** with emojis for better readability
4. **Detailed problem formulation** with EV specifications
5. **Complete setup instructions** for easy reproduction
6. **Usage examples** for all major components
7. **Algorithm descriptions** with performance metrics
8. **Citation format** for academic use
9. **Professional formatting** with proper sections and styling

You can copy and paste this directly into your readme.md file!
