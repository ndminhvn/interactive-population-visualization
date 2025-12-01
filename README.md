# Interactive Population Visualization of Virginia

### Built with VTK, Trame, and Vega-Altair

### Jayachandra Sarika, Anjani Kumar Avadhanam, Minh Nguyen

This project is an interactive visualization system designed to explore county-level population trends in Virginia from 2010–2019. The application combines:

- **VTK** for rendering the Virginia county map
- **Trame** for server–client communication and UI
- **Vega-Altair** for analytical visualizations
- **Machine Learning** tools (K-Means, PCA, Linear Regression)
- **Pandas + NumPy** for data processing

Users can interactively view population trends, compare counties, explore demographic similarity, analyze yearly growth correlations, visualize clusters, and forecast future population trends.

---

## Getting Started

### Requirements

- Python 3.8 to 3.13

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the Application

You can specify a custom port (default is 8080) by using the `--port` argument:

```bash
python src/app.py --port 8000
```

Trame will start a local web server — the terminal will show a URL such as:

http://localhost:8000/

### 4. Using Docker (optional)

You can also run the application using Docker:

#### Build the Docker image:

```bash
docker build -t interactive-population-visualization .
```

#### Run the container:

```bash
# Default: access on localhost:8080
docker run -p 8080:8080 --name interactive-population-visualization interactive-population-visualization

# Or forward to a different local port (e.g., 8000)
docker run -p 8000:8080 --name interactive-population-visualization interactive-population-visualization
```

Then open http://localhost:8080/ (or http://localhost:8000/ if using custom port forwarding) in your browser.

## Data Requirements

_Note: Please contact us for the data files if you do not have them._

Your `data/` folder must contain:

- **virginia_population.csv**: 133 counties × years 2010–2019
- **virginia_counties.vtp**: VTK PolyData of Virginia counties

These files are automatically loaded by `src/data_loader.py`
