
# 🍷 Wine Quality Classification Case Study

## 📌 Overview
This case study focuses on wine quality classification using the K-Nearest Neighbors (KNN) algorithm. The project analyzes various chemical properties of wine to predict its quality class, providing insights into wine production and quality assessment.

---

## 🎯 Problem Statement
Classify wine quality based on various chemical properties and measurements to assist in wine production quality control and help consumers make informed purchasing decisions.

---

## 📂 Dataset
- **File**: `WinePredictor.csv`
- **Size**: 178 samples
- **Target**: `Class` (1, 2, 3 - Wine quality classes)

### 🔬 Features
| Feature                  | Description                                  |
|--------------------------|----------------------------------------------|
| Alcohol                  | Alcohol content percentage                   |
| Malic acid               | Malic acid content                           |
| Ash                      | Ash content                                  |
| Alkalinity of ash        | Alkalinity of ash                            |
| Magnesium                | Magnesium content                            |
| Total phenols            | Total phenols content                        |
| Flavanoids               | Flavanoids content                           |
| Nonflavanoid phenols     | Nonflavanoid phenols content                 |
| Proanthocyanins          | Proanthocyanins content                      |
| Color intensity          | Color intensity                              |
| Hue                      | Hue measurement                              |
| OD280/OD315              | OD280/OD315 of diluted wines                 |
| Proline                  | Proline content                              |

---

## ⚙️ Features

### 🔄 Data Preprocessing
- Missing value handling (`dropna`)
- Feature scaling with `StandardScaler`

### 🔍 Hyperparameter Tuning
- K-value optimization (1–19)
- Accuracy vs K plot for optimal parameter selection

### 📊 Visualization
- Accuracy vs K value plot

### 🤖 Model
- K-Nearest Neighbors Classifier with optimal K

### 📈 Evaluation
- Accuracy
- Confusion Matrix
- Classification Report

### 📦 Artifacts
- Automated model saving with timestamps

---

## 🧪 Technical Implementation

| Component              | Details                                      |
|------------------------|----------------------------------------------|
| Algorithm              | K-Nearest Neighbors (KNN)                    |
| Preprocessing          | `StandardScaler` for feature normalization   |
| Hyperparameter Tuning  | Grid search for optimal K (1–19)             |
| Validation             | 80/20 train-test split (random_state=42)     |
| Distance Metric        | Euclidean distance (default)                 |

---

## 🚀 Usage

### 📋 Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ Running the Application
```bash
python WinePredictorParameterTuningvisual.py --data WinePredictor.csv
```

### 🧾 Command Line Arguments
- `--data`: Path to the dataset CSV file (default: `WinePredictor.csv`)

---

## 📤 Output

- **Model Performance Metrics**: Accuracy, confusion matrix, classification report
- **Visualizations**: Saved in `artifacts_wine/plots/`
  - `accuracy_vs_k.png`: Accuracy for different K values
- **Trained Model**: Saved in `artifacts_wine/models/` with timestamp

---

## 📊 Model Performance

- **Accuracy**: 90–95% on test data with optimal K
- **Precision & Recall**: Good across all wine quality classes
- **Optimal K**: Typically between 3–7 neighbors

---

## 🔍 Key Insights

- **Alcohol Content**: Higher alcohol often correlates with better quality
- **Phenolic Compounds**: Flavanoids and total phenols are strong indicators
- **Color Properties**: Hue and color intensity influence classification
- **KNN Effectiveness**: Performs well on small, well-separated datasets

---

## 🍷 Wine Quality Classes

| Class | Description            |
|-------|------------------------|
| 1     | Lower quality wines    |
| 2     | Medium quality wines   |
| 3     | Higher quality wines   |

---

## 🧪 Dataset Feature Ranges

| Feature                  | Range             |
|--------------------------|-------------------|
| Alcohol                  | 11.03–14.83       |
| Malic acid               | 0.74–5.80         |
| Ash                      | 1.36–3.23         |
| Alkalinity of ash        | 10.6–30.0         |
| Magnesium                | 70–162            |
| Total phenols            | 0.98–3.88         |
| Flavanoids               | 0.34–5.08         |
| Nonflavanoid phenols     | 0.13–0.66         |
| Proanthocyanins          | 0.41–3.58         |
| Color intensity          | 1.28–13.0         |
| Hue                      | 0.48–1.71         |
| OD280/OD315              | 1.27–4.00         |
| Proline                  | 278–1680          |

---

## ✅ KNN Algorithm Benefits

- Non-parametric: No assumptions about data distribution
- Simple: Easy to understand and implement
- Effective: Works well on small to medium datasets
- Interpretable: Results are easy to explain
- Robust: Handles noise well

---

## 🔧 Hyperparameter Tuning Process

- **K Range**: Test values from 1 to 19
- **Cross-validation**: Train-test split evaluation
- **Accuracy Plot**: Visualize performance across K values
- **Optimal Selection**: Choose K with highest accuracy
- **Final Model**: Train with optimal K value

---

## 📁 File Structure

```
Wine_Predictor_Case_Study/
├── WinePredictorParameterTuningvisual.py
├── WinePredictor.csv
├── requirements.txt
├── README.md
└── artifacts_wine/
    ├── models/
    │   └── wine_knn_model_*.joblib
    └── plots/
        └── accuracy_vs_k.png
```

---

## 🍇 Wine Industry Applications

- Quality Control: Automated wine quality assessment
- Production Optimization: Identify key factors for better wine
- Consumer Guidance: Help consumers choose quality wines
- Research: Study chemical properties affecting wine quality
- Competition Judging: Assist in wine competition evaluations

---

## 🍷 Wine Quality Factors

- Alcohol Content: Balance is crucial for quality
- Acidity: Malic acid affects taste and preservation
- Phenolic Compounds: Influence color, taste, and health benefits
- Color Properties: Visual appeal and quality indicators
- Mineral Content: Ash and magnesium affect taste profile

---

## 📦 Dependencies

```
pandas >= 2.1.0  
numpy >= 1.25.0  
matplotlib >= 3.8.0  
seaborn >= 0.12.2  
scikit-learn >= 1.3.0  
joblib >= 1.3.2
```

---

## 👨‍💻 Author
# sakshi prakash khadke 
#3-10-2025