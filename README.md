# ✈️ Airline Delay Prediction

A machine learning project to predict whether a US domestic flight will be delayed by 10 or more minutes, using 2017 flight operations data.

---

## 📌 Project Overview

Flight delays cost airlines and passengers billions of dollars annually. This project builds a binary classification model to predict the likelihood of a flight delay based on pre-departure information such as airline, route, scheduled departure time, and month of travel.

- **Target variable:** Whether a flight is delayed by 10+ minutes (binary: Yes/No)
- **Dataset:** 2017 US Airline On-Time Performance Data (1M+ records)
- **Approach:** Data cleaning → Feature Engineering → EDA → Model Building → Evaluation

---

## 📊 Dataset

| File | Description |
|------|-------------|
| `flights.csv` | Flight-level records including departure time, airline, origin, destination, delay info |
| `airports.csv` | Airport metadata including IATA codes and locations |
| `airlines.csv` | Airline IATA code reference |

**Scale:** Over 1 million flight records processed and validated.

---

## 🔧 Tools & Libraries

| Category | Tools |
|----------|-------|
| Language | Python 2.7 |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib |
| Machine Learning | H2O.ai (Random Forest, GBM), Scikit-learn |
| Other | Regular Expressions (re), OS, DateTime |

---

## 🧹 Data Preprocessing

- Validated origin and destination airport codes using regular expressions against the IATA airport reference dataset
- Removed records with incorrect or missing airport codes
- Dropped columns with entirely null values (e.g. cancellation reason)
- Merged delay sub-categories into a single `OTHER_DELAYS` composite column
- Removed all remaining null entries to produce a clean modelling dataset

---

## 🛠️ Feature Engineering

- Extracted **hour of day** from scheduled departure time to capture time-of-day delay patterns
- Created binary target variable `Delayed` (1 = delayed 10+ minutes, 0 = on time)
- Analysed delay patterns by **month**, **hour of day**, and **airline** to inform feature selection

**Key findings from EDA:**
- Delay rates vary significantly by month — summer and holiday periods show higher delay rates
- Evening departures are consistently more likely to be delayed than morning flights
- Delay rates vary meaningfully across airlines

---

## 🤖 Models Built

Two models were trained and evaluated using H2O.ai:

### 1. H2O Random Forest (`rf_user_df_v1`)
- Ensemble of decision trees with bootstrapped sampling
- Evaluated on held-out test data

### 2. H2O Gradient Boosting Machine — GBM (`model3`) ✅ Best Model
- Sequential boosting approach optimising for classification performance
- Outperformed Random Forest on key metrics

---

## 📈 Model Performance (GBM — Test Set)

| Metric | Value |
|--------|-------|
| **AUC** | **0.873** |
| Max Accuracy | 84.7% |
| Max Mean Per-Class Accuracy | 80.0% |
| Log Loss | 0.374 |
| RMSE | 0.338 |
| Gini Coefficient | 0.746 |

**Confusion Matrix (at max F1 threshold = 0.345):**

| | Predicted: No Delay | Predicted: Delayed |
|---|---|---|
| **Actual: No Delay** | 1,693 | 206 |
| **Actual: Delayed** | 222 | 513 |

- **Overall Error Rate:** 16.25%
- **Precision at max threshold:** 100% (conservative prediction)
- **AUC of 0.873** indicates strong discriminatory power between delayed and on-time flights

---

## 🔍 Key Takeaways

- An AUC of **0.873** demonstrates the model's strong ability to distinguish delayed from on-time flights
- GBM outperformed Random Forest, consistent with its effectiveness on structured tabular data
- Time of day and airline carrier are among the most informative features for predicting delays
- The model provides a practical foundation for passenger-facing delay risk tools or airline operational planning systems

---

## 📁 Repository Structure

```
airline-delay-prediction/
│
├── Capstone_F.ipynb          # Main Jupyter notebook (full analysis)
├── README.md                 # Project documentation
├── airlines.csv              # Airline IATA code reference
├── airports.csv              # Airport metadata
└── requirements.txt          # Python dependencies
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/rohanmg91/airline-delay-prediction.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook Capstone_F.ipynb
   ```

> **Note:** H2O.ai requires Java to be installed. Download from [h2o.ai](https://h2o.ai).

---

## 👤 Author

**Rohan Mathew George**  
Senior Data Scientist | FMCG & Retail Analytics  
📧 rohanmg91@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/rohanmg91)
