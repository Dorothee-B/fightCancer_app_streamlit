# fightCancer_app_streamlit

Une application **Streamlit interactive** permettant d’évaluer le **risque de développer un cancer** à partir des habitudes de vie et des données personnelles de santé. 

Le modèle utilise un classifieur **Random Forest**  entraîné avec un **réchantillonnage et correctif SMOTE** pour corriger le déséquilibrexs des classes.


## Tester l'application en ligne sur Streamlit Cloud

Tester l'application en ligne sur Streamlit Cloud (aucune installation nécessaire)  👉 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fightcancerappapp-mq3mhixvyhxr5jne567rt6.streamlit.app/)

https://fightcancerappapp-mq3mhixvyhxr5jne567rt6.streamlit.app/

### Accès rapide via QR code

Scannez ce QR code avec votre téléphone pour accéder directement à l'application : <br/>

<img src="images/qr_FightCancer_OncoSisters_streamlit.png" alt="Streamlit form - part 1" width="200"/>



## Fonctionnalités

- Saisie interactive des informations personnelles et des habitudes de vie
- Prédiction du score de risque avec un modèle pré-entraîné
- Affichage d'un risque de développer un cancer sous forme de pourcentage (0% = faible risque à 100% = risque très élevé)
- Message personnalisé selon le score
- Conseils santé adaptés selon le niveau de risque
- Interface utilisateur claire et intuitive



## Aperçu de l'application

### Formulaire d’entrée
<img src="images/App_home_screenshot.png" alt="Streamlit form - part 1" width="300"/>
<br/>
<img src="images/App_home_screenshot1.png" alt="Streamlit form - part 2" width="300"/>
<br/>
<img src="images/App_home_screenshot2.png" alt="Streamlit form - part 3" width="300"/>
<br/>
<img src="images/App_home_screenshot3.png" alt="Streamlit form - part 3" width="300"/>

### Message personnalisé selon le score
<img src="images/Result_app_screenshot.png" alt="Message pour un risque modéré / haut" width="300"/>

---

## Installation locale (optionnelle)

1. Cloner ce dépôt :
```bash
git clone https://github.com/Dorothee-B/fightCancer_app_streamlit.git
cd fightCancer_app_streamlit
```

2. Créer un environnement virtuel (optionnel mais recommandé) :
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Lancer l'application : 
```
streamlit run Cancer_app_smote_resample_rf.py
```
---

##  📂 Structure du projet
```bash
fightCancer_app_streamlit/
│
├── Cancer_app_smote_resample_rf.py  # Script principal Streamlit
├── model_cancer_resample_rf.pkl      # Modèle ML pré-entraîné
├── requirements.txt            # Dépendances Python
├── images/                     # Dossier des images (screenshots_app, logo, smiley)
│   ├── App_home_screenshot.png
│   ├── App_home_screenshot1.png
│   └── Result_app_screenshot.png
│   └── Logo_fight_cancer_app.png
├── data/                       # dataset, features_importance.txt
└── README.md                   # Ce fichier
```
---

## 📊 Jeu de données

Cette application repose sur les données de l’enquête **HINTS 7 (2024)** (*Health Information National Trends Survey*), composé de 77 questions et réalisée par le **National Cancer Institute (NCI)** aux États-Unis.

- **Période de collecte** : mars à septembre 2024
- **Modes de réponse** : papier et en ligne, avec incitations financières ($2 + $10)
- **Méthodologie** : échantillonnage aléatoire en deux étapes (adresse + individu)
- **Objectif** : mesurer les comportements de santé, les habitudes de vie, et l'accès à l'information médicale dans la population américaine adulte
- **Taille initiale de l’échantillon** : 7278 participants (enquête HINTS 7)
- **Taille finale après nettoyage et rééchantillonnage** :
  - **Entraînement** : 708 individus (dont classes équilibrées via SMOTE)
  - **Test** : 177 individus (dont 90 cas à risque élevé)
  Ce traitement permet d’assurer la qualité et la robustesse du modèle malgré le déséquilibre initial.

- **Expériences intégrées** : 
  - *Engagement de sincérité* sur la qualité des réponses
  - Incitation spécifique ($10) dans les zones à forte minorité pour améliorer la représentativité
- **Lien officiel** : [https://hints.cancer.gov](https://hints.cancer.gov)
- **Licence / Accès** : données publiques accessibles gratuitement pour usage non commercial


## 🧠 Modèle de Machine Learning
L’algorithme de prédiction repose sur un **Random Forest Classifier**, sélectionné après comparaison automatisée de plusieurs modèles avec la **librairie PyCaret**.

## Processus de modélisation 

- **Prétraitement :**
  - Nettoyage des données issues de l’enquête HINTS 7
  - Encodage des variables catégorielles
  - Rééchantillonage manuel et rééquilibrage des classes avec **SMOTE** (Synthetic Minority Over-sampling Technique)

- **Comparaison des modèles :**
  - Utilisation de **PyCaret** pour tester plusieurs algorithmes : Random Forest, Gradient Boosting, XGBoost, Logistic Regression, etc.
  - Évaluation selon plusieurs métriques : **Accuracy**, **Recall**, **F1-score**, **AUC**, **MCC**

- **Sélection finale :**
  - Le modèle **Random Forest** a été retenu pour son bon compromis entre **performance**, **robustesse** et **interprétabilité**
  - Le **Recall pour la classe “à risque”** étant prioritaire dans cette problématique de santé, il a été utilisé comme critère principal

## Hyperparamètres du modèle final Random Forest
Le modèle a été optimisé par recherche d'hyperparamètres. Les meilleurs paramètres trouvés :
- max_depth: 20  
- min_samples_leaf: 2  
- min_samples_split: 5  
- n_estimators: 200  

Ces réglages ont permis d'améliorer la précision et la robustesse du modèle.

## 📈 Résultats du modèle Random Forest

| Classe         | Précision | Rappel | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| 0 (faible risque) | 0.76   | 0.70   | 0.73     | 87      |
| 1 (haut risque)   | 0.73   | 0.79   | 0.76     | 90      |
| **Accuracy globale** |         |        | **0.75** | 177     |

Le modèle **maximise le rappel de la classe "à risque" (0.79)** pour ne pas rater de cas potentiellement graves.

- Le modèle a été sauvegardé et intégré dans l’application avec joblib.

## Validation croisée (5 plis)
Une validation croisée à 5 plis a été effectuée pour évaluer la performance généralisée du modèle.

| Métrique      | Plis 1 | Plis 2 | Plis 3 | Plis 4 | Plis 5 | Moyenne   | Écart-type |
| ------------- | ------ | ------ | ------ | ------ | ------ | --------- | ---------- |
| **Accuracy**  | 0.695  | 0.723  | 0.712  | 0.761  | 0.670  | **0.712** | ±0.030     |
| **Précision** | 0.657  | 0.680  | 0.693  | 0.747  | 0.633  | **0.682** | ±0.039     |
| **Rappel**    | 0.765  | 0.800  | 0.718  | 0.765  | 0.738  | **0.757** | ±0.028     |
| **F1-score**  | 0.707  | 0.735  | 0.705  | 0.756  | 0.681  | **0.717** | ±0.026     |


Ces résultats confirment la stabilité et la fiabilité du modèle sur des échantillons variés, notamment en ce qui concerne le rappel sur la classe "à risque".

---

⚠️ Cette application ne fournit pas un diagnostic médical mais une estimation basée sur des données déclaratives. **Consultez un professionnel de santé pour tout avis médical.**

---

## Auteurs 
- Aasiyah B.
- Dorothée B.

<img src="images/Onco-sisters_logo.png" alt="Onco-sisters_logo" width="100"/>

## Contacts
Pour toute question ou suggestion, n’hésitez pas à me contacter : 
- Aasiyah B. - aasiyah.bhewa@gmail.com
- Dorothée B. - busierdorothee@gmail.com

## ❗ Licence

Ce projet est protégé par le droit d’auteur.  
**Toute utilisation, reproduction, modification ou redistribution est strictement interdite sans autorisation écrite préalable.**
