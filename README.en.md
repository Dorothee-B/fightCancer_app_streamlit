# fightCancer_app_streamlit

An interactive **Streamlit application** designed to assess the risk of developing cancer based on personal health data and lifestyle habits.

The model uses a **Random Forest classifier** trained with **SMOTE resampling** to correct class imbalance.

---

## Features

- Interactive input of personal and lifestyle information  
- Risk score prediction using a pre-trained machine learning model  
- Risk displayed as a percentage (0%: low risk → 100%: very high risk)  
- Personalized health advice based on the predicted score  
- Clear, user-friendly results interface  

---

## Installation

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
├── images/                           # Image assets (emoji, logos)
├── data/                             # Dataset, features_importance.txt
└── README.en.md                      # This file
```

## Dataset
This application is based on the **HINTS 7 (2024)** (Health Information National Trends Survey) dataset, containing **77 questions** and conducted by the **National Cancer Institute (NCI)** in the United States.

- **Data collection period**: March – September 2024
- **Modes**: paper and web, with financial incentives ($2 + $10)
- **Sampling method**: 2-stage random sampling (address + one adult per household)
- **Goal**: to assess health behaviors, lifestyle, and access to medical information among U.S. adults
- **Built-in experiments**:
  - *Commitment to answer truthfully* to improve data quality
  - Additional $10 incentive in high-minority strata to improve representation
- **Official source**: https://hints.cancer.gov
- **License**: public data available for research and non-commercial use

The data were cleaned and **balanced** using **SMOTE**, then a **Random Forest**classifier was trained to predict cancer risk based on lifestyle features.

⚠️ This tool is **not a medical diagnostic**, but a predictive aid based on self-reported data. **Always consult a healthcare professional for medical concerns**.

## Authors
Aassyiah B.
Dorothée B.

## Contact
For questions or feedback, feel free to reach out:
Dorothée B. – busierdorothee@gmail.com
Aassyiah B.