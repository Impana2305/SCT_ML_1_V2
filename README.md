# 🏡 House Price Prediction using Linear Regression

### **SkillCraft Technology - Machine Learning Internship (Task 01)**

This repository contains an end-to-end Machine Learning project to predict housing prices in Indian Rupees (₹) using multivariate Linear Regression. It includes a comprehensive Jupyter Notebook detailing the entire data science workflow, the cleaned housing dataset, and a premium React-based dashboard web application integrating the model for real-time client-side valuations.

---

## 📋 Table of Contents
1. [Objective & Feature Selection](#-objective--feature-selection)
2. [Machine Learning Workflow](#-machine-learning-workflow)
3. [Evaluation Metrics & Performance](#-evaluation-metrics--performance)
4. [Project Visualizations & Insights](#-project-visualizations--insights)
5. [Interactive Web Dashboard](#-interactive-web-dashboard)
6. [Repository Structure](#-repository-structure)
7. [Getting Started & Local Execution](#-getting-started--local-execution)

---

## 🎯 Objective & Feature Selection

The primary goal of this task is to construct a regression model that accurately estimates market values for residential properties based on three main spatial specifications:
- **Square Footage (`area`)**: Total interior living space in square feet.
- **Number of Bedrooms (`bedrooms`)**: Count of bedrooms in the property.
- **Number of Bathrooms (`bathrooms`)**: Count of bathrooms in the property.

---

## 🛠️ Machine Learning Workflow

The model was built using the following step-by-step pipeline in the Jupyter Notebook:
1. **Data Collection & Load**: Loaded the raw house price dataset (`Housing.csv`) and validated rows, shapes, and features.
2. **Data Preprocessing & Cleaning**:
   - Checked and confirmed zero null values (`df.isnull().sum()`).
   - Performed outlier detection and cleaning using the **Interquartile Range (IQR)** method on the predictor variables (`area`, `bedrooms`, `bathrooms`) to prevent regression line skewing.
3. **Feature Engineering**:
   - Engineered a custom **Spatial Layout Ratio** (`area_per_bedroom`) representing the average square footage allocated per bedroom. This metric controls for density and spatial layout premium.
4. **Feature Scaling & Split**:
   - Conducted an **80-20 train-test split** (random state 42) for validation stability.
   - Scaled the training features using **Standard Scaling (Z-score normalization)** to ensure even weights training across varying dimensional scales.
5. **Model Building & Training**:
   - Fitted a multivariate Scikit-Learn `LinearRegression` model.
   - Extracted standard model intercept and feature impact coefficients.

---

## 📈 Evaluation Metrics & Performance

The model's performance on the unseen validation test set is summarized below:

| Metric | Value | Description |
| :--- | :---: | :--- |
| **R-squared Score ($R^2$)** | **0.4551** | Explains 45.5% of variance in validation house prices. |
| **Mean Absolute Error (MAE)** | **₹1,180,967.16** | The average absolute Rupee prediction deviation (~₹11.8 Lakhs). |
| **Root Mean Squared Error (RMSE)** | **₹1,567,544.86** | Standard deviation of prediction residuals (~₹15.7 Lakhs). |
| **Mean Squared Error (MSE)** | **2,457,196,879,839.07** | Average squared difference between actual and predicted prices. |

---

## 📊 Project Visualizations & Insights

The Jupyter Notebook embeds comprehensive seaborn and matplotlib charts representing key house price findings:
- **Price Distribution Histogram**: Visualizes the density distribution of prices across all market records.
- **Feature Boxplots**: Shows the range and distribution of prices grouped by the bedroom and bathroom counts.
- **Bivariate Scatter Plots**: Maps Area vs. Price, with color indicators showing the positive price trend of higher bedroom/bathroom counts.
- **Correlation Heatmap**: Inspects collinearity and direct correlation coefficients.
- **Actual vs. Predicted Price Fit**: Scatter plot contrasting validation targets against regression lines along the ideal $y=x$ dashed line.
- **Impact Coefficients Chart**: Shows that bathroom counts have the strongest positive Z-score coefficient weight driving up prices, whereas high room densities (more bedrooms in smaller spaces) yield a negative coefficient adjustment.

---

## 💻 Interactive Web Dashboard

To make the predictive model accessible, a beautiful **React & Vite web application** was built. 
- Features glassmorphism panels, customized slider inputs, and dynamic charts using Recharts.
- Performs client-side Z-score scaling and regression inference using the pre-computed JSON model weights, removing python server lag entirely.
- Includes three dedicated navigation panels: **Valuation Estimator**, **Market Explorer** (with data statistics and histogram), and **Diagnostics & Analytics** (featuring real-time Actual vs. Predicted scatter fits and feature coefficients).
- Displays all prices formatted in the standard **Indian Rupee (₹)** numbering system (Lakhs and Crores).

---

## 📂 Repository Structure

```directory
├── data/
│   └── Housing.csv           # Raw dataset
├── src/                      # React frontend client source code
│   ├── assets/               # Aesthetic design assets
│   ├── App.css               # Vanilla CSS variables & layout styles
│   ├── App.jsx               # Dashboard application code (dynamic evaluation)
│   ├── index.css             # Main styling rules
│   ├── main.jsx              # React runtime entrypoint
│   └── model_params.json     # Saved scaler Z-scores & linear coefficients
├── backup/                   # Python scripts & model configuration backups
│   ├── app.py                # Alternate Streamlit python dashboard
│   ├── train_model.py        # Core model training script
│   └── model_params.json     # Backup parameters
├── Housing.csv               # Root level dataset for Jupyter loading
├── SCT_ML_1_House_Price_Prediction.ipynb  # Pre-executed Jupyter Notebook
├── README.md                 # Project portfolio documentation (this file)
├── package.json              # Client dependencies config
├── vite.config.js            # Build server configuration
└── eslint.config.js          # Code linter rules
```

---

## 🚀 Getting Started & Local Execution

### **1. Running the Jupyter Notebook (.ipynb)**
To explore the machine learning model training steps and visualizations:
1. Ensure Python 3.10+ is installed.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn notebook ipykernel
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open [SCT_ML_1_House_Price_Prediction.ipynb](file:///d:/SCT_ML_1_V2/SCT_ML_1_House_Price_Prediction.ipynb) and run the cells.

### **2. Running the React Web Application**
To run the live interactive dashboard local server:
1. Install Node.js dependencies:
   ```bash
   npm install
   ```
2. Start the local Vite development server:
   ```bash
   npm run dev
   ```
3. Open the browser to the address shown in the terminal (typically `http://localhost:5173`).
