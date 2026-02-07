# 🚦 NYC Traffic Classification: ML-Powered Urban Mobility Analysis

![Project Banner](data/traffic_jam.png)

## 👋 About the Project

This repository implements a machine learning classification system that categorizes New York City boroughs as "traffic jam heavy" or "free-flowing" using publicly available traffic speed data from NYC DOT (Department of Transportation). Developed from a real estate investment perspective, the project operates on the hypothesis that traffic congestion significantly impacts neighborhood desirability, rental values, and demand dynamics.

The pipeline transforms raw traffic measurements into borough-level statistical features (average speed, coefficient of variation) and applies Logistic Regression and Random Forest algorithms to predict traffic patterns that matter for urban planning and investment decisions.

## 🎯 What Does It Do?

- **Data Collection**: Automatically pulls NYC DOT traffic speed data via API (JSON/CSV format)
- **Feature Engineering**: Aggregates individual speed readings into borough-level statistics (mean, median, std, CV)
- **Smart Labeling**: Classifies areas as "jammed" (`jam_label=1`) when average speed < 10 mph
- **Dual Models**: Trains both Logistic Regression and Random Forest classifiers
- **Performance Tracking**: Generates AUC and accuracy leaderboards with visualizations

## 🛠️ Installation

Set up your environment with the following steps:

```bash
# Clone the repository
git clone https://github.com/username/nyc-traffic-classification.git
cd nyc-traffic-classification

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate

# Install dependencies (including dev tools: ruff, pytest)
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, Pandas ≥2.0, Numpy, Requests, Scikit-learn, Matplotlib

## 🚀 Usage

Follow these steps to run the complete pipeline:

### 1. Download NYC Traffic Data

```bash
python -m src.data.pull_nyc_speed --limit 50000 --out data/raw/nyc_speeds.json
```

Fetches up to 50,000 records of real-time traffic speed measurements from NYC Open Data API. Data includes speed readings, timestamps, and location identifiers.

### 2. Build Classification Table

```bash
python -m src.data.make_table --in_csv data/raw/nyc_speeds.json --out data/processed/traffic_table.csv
```

Aggregates raw speed data by borough/area and creates binary labels:
- `jam_label = 1`: Average speed < 10 mph (heavy traffic)
- `jam_label = 0`: Average speed ≥ 10 mph (free-flowing)

### 3. Train Models

```bash
python -m src.models.train_models
```

Trains classification models (Logistic Regression, Random Forest) and saves:
- Trained models to `reports/models/`
- Performance metrics to `reports/results/leaderboard.csv`

### 4. Evaluate Performance

```bash
python -m src.models.eval_models
```

Generates evaluation outputs:
- AUC comparison chart: `reports/figures/model_auc.png`
- Detailed classification reports with precision, recall, and F1 scores

## 🧠 Model Architecture

The system uses borough-level aggregated features to classify traffic patterns:

### Feature Engineering

| Feature | Description | Formula/Details |
|---------|-------------|-----------------|
| **avg_speed** | Mean speed across all readings | Simple average of mph values |
| **median_speed** | Median speed (robust to outliers) | 50th percentile |
| **std_speed** | Speed variability | Standard deviation |
| **speed_cv** | Coefficient of variation | `std_speed / avg_speed` |
| **log_count** | Log-transformed reading count | `log(1 + count)` |

### Target Variable

- **jam_label**: Binary classification
  - `1` → Heavy traffic (avg_speed < 10 mph)
  - `0` → Free-flowing (avg_speed ≥ 10 mph)

### Models

| Model | Algorithm | Use Case |
|-------|-----------|----------|
| **Logistic Regression** | Linear probabilistic classifier | Interpretable baseline, feature importance |
| **Random Forest** | Ensemble decision trees | Captures non-linear patterns, robust to outliers |

## 📊 Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── pull_nyc_speed.py     # NYC DOT API data fetcher
│   │   └── make_table.py         # Borough aggregation & labeling
│   ├── feats/
│   │   └── build.py              # Feature engineering utilities
│   └── models/
│       ├── train_models.py       # Model training pipeline
│       └── eval_models.py        # Evaluation & visualization
├── reports/
│   ├── results/
│   │   └── leaderboard.csv       # Model performance metrics
│   ├── models/                   # Saved .pkl/.joblib files
│   └── figures/
│       └── model_auc.png         # AUC comparison chart
├── data/
│   ├── raw/                      # Downloaded traffic JSON/CSV
│   └── processed/                # Aggregated classification tables
└── README.md
```

**Note**: `reports/results/` and `reports/figures/` are tracked by git to preserve model benchmarks.

## 📈 Sample Output

After training, expect performance metrics like:

```
Classification Leaderboard:
Model                    AUC     Accuracy
logistic_regression      0.87    0.82
random_forest            0.91    0.86

Classification Report (Random Forest):
              precision    recall  f1-score   support
           0       0.88      0.84      0.86       145
           1       0.85      0.89      0.87       155

    accuracy                           0.86       300
   macro avg       0.87      0.87      0.87       300
weighted avg       0.86      0.86      0.86       300
```

**Note**: Results vary based on data collection time and API response.

## 🔬 Technical Details

### Data Source

- **API**: NYC Open Data (DOT Real-Time Traffic Speed Data)
- **Format**: JSON/CSV with fields: `speed`, `data_as_of`, `borough`, `link_id`
- **Rate Limits**: Default 50,000 records per pull (configurable via `--limit`)

### Classification Logic

The 10 mph threshold for traffic jam classification is based on:
- Urban planning standards (Level of Service F)
- NYC street grid average free-flow speeds (~25-30 mph)
- Empirical observation of severe congestion patterns

### Model Training

- **Train/Test Split**: 80/20 stratified split (preserves class balance)
- **Hyperparameters**: Scikit-learn defaults (tune via GridSearchCV if needed)
- **Evaluation Metrics**: AUC-ROC, Accuracy, Precision, Recall, F1

## 🎨 Visualization

The evaluation script generates an AUC comparison chart showing model performance:

```python
# Example: Model AUC scores visualized as bar chart
Logistic Regression: 0.87
Random Forest:       0.91 ⭐ (Best)
```

## 📝 TODO

Future enhancements and improvements:

- [ ] Add temporal features (time of day, day of week, rush hour indicators)
- [ ] Incorporate weather data (rain/snow impact on traffic)
- [ ] Experiment with XGBoost/LightGBM for better performance
- [ ] Build real-time prediction API (FastAPI/Flask)
- [ ] Create interactive dashboard (Streamlit/Dash) for live traffic maps
- [ ] Add cross-validation for more robust evaluation
- [ ] Integrate geospatial features (distance to subway, highway proximity)
- [ ] Expand to multi-class classification (light/moderate/heavy/severe)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or suggest features via Issues
- Submit pull requests for improvements
- Share insights on traffic pattern analysis
- Propose new data sources (MTA, Citibike, etc.)

## 📄 License

This project is open source and for educational purposes only. Traffic predictions are not intended for critical transportation or safety decisions—always use official NYC DOT resources for real-time navigation. 🚗

---

**Disclaimer**: This is an experimental ML project for urban analytics research. Traffic patterns are dynamic and influenced by numerous factors beyond speed measurements. Use predictions as informational insights, not authoritative guidance.
