This project aims to predict the presence of heart disease in patients using various machine learning algorithms. It uses a dataset of patient medical records and applies classification models to detect heart disease based on key health indicators.
- **Goal**: Predict whether a patient has heart disease (binary classification: 0 = No, 1 = Yes)
- **Dataset**: UCI Heart Disease Dataset
- **Algorithms Used**: 
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)

 **Feature Used**:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Rest ECG results
- Max heart rate achieved
- Exercise-induced angina
- ST depression
- Slope of the peak exercise ST segment
- Number of major vessels (0–3) colored by fluoroscopy
- Thalassemia
Metrics used to evaluate models:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Curve

**Result**:
| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | 85.2%    |
| Random Forest       | 88.6%    |
| KNN                 | 83.4%    |
| SVM                 | 86.0%    |

**Diployment**:
This project can be deployed using:
- **Streamlit** for an interactive UI
- **Flask** for a backend API
