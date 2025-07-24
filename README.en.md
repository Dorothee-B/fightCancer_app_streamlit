# fightCancer_app_streamlit

An interactive **Streamlit application** designed to assess the risk of developing cancer based on personal health data and lifestyle habits.

The model uses a **Random Forest classifier** trained with **SMOTE resampling** to correct class imbalance.

---

## Try the App via Streamlit Cloud

Click below to launch the app directly in your browser (no installation needed): 👉 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fightcancerappapp-mq3mhixvyhxr5jne567rt6.streamlit.app/)

https://fightcancerappapp-mq3mhixvyhxr5jne567rt6.streamlit.app/

### Quick access via QR code

Scan this QR code with your phone to open the app instantly:
<img src="images/qr_FightCancer_OncoSisters_streamlit.png" alt="Streamlit form - part 1" width="300"/>
---

## Features

- Interactive input of personal and lifestyle information  
- Risk score prediction using a pre-trained machine learning model  
- Risk displayed as a percentage (0%: low risk → 100%: very high risk)  
- Personalized health advice based on the predicted score  
- Clear, user-friendly results interface  

---

## Application Preview

### Input Form
<img src="images/App_home_screenshot.png" alt="Streamlit form - part 1" width="300"/>
<br/>
<img src="images/App_home_screenshot1.png" alt="Streamlit form - part 2" width="300"/>

### Personalized Message Based on Risk Score
<img src="images/Result_app_screenshot.png" alt="Message pour un risque modéré / haut" width="300"/>


---

## Installation (optional)

1. Clone this repository:
```bash
git clone https://github.com/Dorothee-B/fightCancer_app_streamlit.git
cd fightCancer_app_streamlit
```

2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
````

3. Install the dependencies:
```bash
pip install -r requirements.txt
````

## 🚀 Deployment
This app is deployed on Streamlit Cloud and can be accessed at:
https://fightcancerappapp-mq3mhixvyhxr5jne567rt6.streamlit.app/

📂 Project Structure
```bash
fightCancer_app_streamlit/
│
├── Cancer_app_smote_resample_rf.py   # Main Streamlit script
├── model_cancer_resample_rf.pkl      # Pre-trained ML model
├── requirements.txt                  # Python dependencies
├── images/                           # Dossier des images (screenshots_app, logo, smiley)
│   ├── App_home_screenshot.png
│   ├── App_home_screenshot1.png
│   └── Result_app_screenshot.png
│   └── Logo_fight_cancer_app.png
├── data/                             # Dataset and feature importance report
└── README.en.md                      # This file
```

## Dataset
This application is based on the **HINTS 7 (2024)** (Health Information National Trends Survey) dataset, containing **77 questions** and conducted by the **National Cancer Institute (NCI)** in the United States.

- **Data collection period**: March – September 2024
- **Modes**: paper and web, with financial incentives ($2 + $10)
- **Sampling method**: 2-stage random sampling (address + one adult per household)
- **Initial sample size:** 7278 respondents
- **After data cleaning, filtering, resampling, the final modeling dataset was reduced to:**

  - **Training set:** 709 individuals
  - **Test set:** 178 individuals, including 87 high-risk cases

- **Goal**: to assess health behaviors, lifestyle, and access to medical information among U.S. adults
- **Built-in experiments**:
  - *Commitment to answer truthfully* to improve data quality
  - Additional $10 incentive in high-minority strata to improve representation
- **Official source**: https://hints.cancer.gov
- **License**: public data available for research and non-commercial use

## 🧠 Machine Learning Model

The prediction algorithm is based on a **Random Forest Classifier**, selected after automated comparison of several models using the **PyCaret** library.

---

## Modeling Process

- **Preprocessing:**
  - Data cleaning from the HINTS 7 survey
  - Encoding of categorical variables
  - Manualy Resampling and Class balancing using **SMOTE** (Synthetic Minority Over-sampling Technique)

- **Model comparison:**
  - Automated benchmarking via **PyCaret** with models such as Random Forest, Gradient Boosting, XGBoost, Logistic Regression, etc.
  - Evaluation metrics: **Accuracy**, **Recall**, **F1-score**, **AUC**, **MCC**

- **Final selection:**
  - **Random Forest** was chosen for its balance of **performance**, **robustness**, and **interpretability**
  - **Recall on the “high-risk” class** was prioritized due to the health-critical nature of the task

---

### 📈 Random Forest Model Results

| Class             | Precision | Recall | F1-score | Support |
|-------------------|-----------|--------|----------|---------|
| 0 (low risk)      | 0.76      | 0.70   | 0.73     | 91      |
| 1 (high risk)     | 0.71      | 0.77   | 0.74     | 87      |
| **Overall Accuracy** |         |        | **0.74** | 178     |

The model **maximizes recall for the "high-risk" class (0.77)** to avoid missing potentially serious cases.

- The trained model was saved and integrated into the app using `joblib`.


---

⚠️ This application does not provide a medical diagnosis. It is an estimation based on self-reported data. **For medical concerns, please consult a healthcare professional.**

---

## Authors  
- Aasiyah B.  
- Dorothée B.

<img src="images/Onco-sisters_logo.png" alt="Onco-sisters_logo" width="100"/>

## Contact

For any questions or suggestions, feel free to reach out:  
📧 Aasiyah B. - aasiyah.bhewa@gmail.com
📧 Dorothée B. – busierdorothee@gmail.com  


## ❗ License

This project is protected by copyright.  
**Any use, reproduction, modification, or redistribution is strictly prohibited without prior written permission.**
