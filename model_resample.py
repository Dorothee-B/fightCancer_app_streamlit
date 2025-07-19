# Script d'entraînement du Random Forest pour l'application Streamlit de détection du cancer

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib
import os
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Importer SMOTE pour le rééchantillonnage
from imblearn.over_sampling import SMOTE # NÉCESSITE: pip install imbalanced-learn!! il faudrait le rajouter sur le fichier requirements txt 
from imblearn.pipeline import Pipeline
# Charger les données
try:
    df = pd.read_csv('data/df2.csv')
except FileNotFoundError:
    print("Erreur: Le fichier 'df2.csv' n'a pas été trouvé. Veuillez vous assurer qu'il est dans le même répertoire.")
    exit()

cancer_oui= df[df['EverHadCancer'] == 1]  # Filtrer les lignes des personnes qui ONT le cancer
cancer_non = df[df['EverHadCancer'] == 0]

# Sous-échantillonner le groupe avec 1073 lignes, df[df['EverHadCancer'] == 1]  # Filtrer(personnes qui ont un Cancer dans le fichier de base)
resample_cancer_oui = cancer_oui.sample(n=1073, random_state=142)
resample_cancer_non = cancer_non.sample(n=1073, random_state=142)

# Combiner les deux sous-échantillons
balanced_df = pd.concat([resample_cancer_oui, resample_cancer_non])

# Mélanger les lignes pour éviter l'ordre
balanced_df = balanced_df.sample(frac=1, random_state=142).reset_index(drop=True)
print(balanced_df['EverHadCancer'].value_counts())

# Variables (définies selon l'ordre logique pour le preprocessor)
binary_vars = [
    'CutSkipMeals2',
    'DiffPayMedBills', 'SmokeNow',
    'MedConditions_Diabetes', 'MedConditions_HighBP', 'MedConditions_HeartCondition',
    'MedConditions_LungDisease', 'MedConditions_Depression'
]
ordinal_categorical_vars = [
    'GeneralHealth', 'HealthLimits_Pain', 'Nervous',
    'IncomeRanges', 'Education'
]
string_categorical_vars = ['Birthcountry']  # La variable qui sera one-hot encodée

continuous_vars = [
    'Fruit2', 'Vegetables2', 'TimesStrengthTraining', 'Drink_nb_PerMonth',
    'ChildrenInHH', 'TotalHousehold', 'TimesSunburned', 'BMI', 'Age', 'SleepWeekdayHr'
]

target = 'EverHadCancer'

# Features utilisées pour l'entraînement (avant toute transformation)
# Cet ordre correspond à l'ordre des colonnes du DataFrame X d'entrée
features_raw = binary_vars + ordinal_categorical_vars + string_categorical_vars + continuous_vars

df_clean = balanced_df[features_raw + [target]].dropna() # Assurez-vous que toutes les colonnes existent et qu'il n'y a pas de NaN pour ces features

X = df_clean[features_raw]
y = df_clean[target]

# Préprocesseur avec ColumnTransformer
# L'ordre des transformations ici DÉFINIT l'ordre de sortie des features dans le pipeline
# et donc l'ordre attendu par le modèle final.
transformers = [
    ('ord', OrdinalEncoder(), ordinal_categorical_vars),
    ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), string_categorical_vars),
    ('scale', StandardScaler(), continuous_vars)
]

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='passthrough'  # les variables binaires passent sans transformation
)

# Créer le pipeline complet, y compris SMOTE
# SMOTE est appliqué APRÈS le préprocesseur mais AVANT le classificateur.
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=142)), # Ajout de l'étape SMOTE
    ('model', RandomForestClassifier(n_estimators=100, random_state=142))
])

# Split des données et entraînement
# Le pipeline entier, y compris preprocessing et SMOTE, sera appliqué sur X_train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)
pipeline.fit(X_train, y_train)

# Évaluation du modèle
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Récupération des noms de features FINAUX après preprocessing pour les enregistrer ---
# L'ordre est crucial et doit correspondre à l'ordre du ColumnTransformer.
# SMOTE ne change pas le nom des features, donc cette logique reste la même.
# 1. Features ordinales
ord_features = ordinal_categorical_vars

# 2. Features OneHotEncoded
ohe_transformer = pipeline.named_steps['preprocess'].named_transformers_['ohe']
# get_feature_names_out donne les noms corrects comme 'Birthcountry_Mexican'
ohe_features = list(ohe_transformer.get_feature_names_out(string_categorical_vars))

# 3. Features continues
cont_features = continuous_vars

# 4. Features binaires (passées directement, 'remainder' du ColumnTransformer)
bin_features = binary_vars

# L'ordre des features dans la matrice finale (cet ordre doit être respecté par l'application)
final_feature_names_for_model_input = (
    ord_features +
    ohe_features +
    cont_features +
    bin_features
)

# Calcul des importances des features (peut être affecté par SMOTE, mais la logique reste la même)
try:
    # Récupérer le classificateur après SMOTE et preprocessing
    model_trained = pipeline.named_steps['model']
    # Si le modèle est un RandomForestClassifier, il aura feature_importances_
    if hasattr(model_trained, 'feature_importances_'):
        importances = model_trained.feature_importances_
        importance_df = pd.DataFrame({
            'Variable': final_feature_names_for_model_input,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Visualisation Top 20
        top_20 = importance_df.head(20)
        plt.figure(figsize=(12, 10))
        plt.barh(top_20['Variable'][::-1], top_20['Importance'][::-1], color='teal')
        plt.xlabel("Importance")
        plt.title(f"Top 20 Variables (Random Forest - {target})")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print("Le modèle final n'a pas d'attribut 'feature_importances_'. Impossible d'afficher les importances.")

except Exception as e:
    print(f"Erreur lors du calcul/affichage des importances des features: {e}")


# Sauvegarde du pipeline entraîné (incluant preprocessing et SMOTE)
joblib.dump(pipeline, "modele_cancer_resample_rf.pkl")
print("Pipeline entraîné (avec SMOTE) sauvegardé sous 'modele_cancer_resample_rf.pkl'")

# Sauvegarde des importances (facultatif)
if 'importance_df' in locals():
    importance_df.to_csv("data/importances_cancer_resample_rf.csv", index=False)
    print("Importances des features sauvegardées sous 'data/importances_cancer_resample_rf.csv'")

# Sauvegarde des noms de features FINALES (crucial pour l'application Streamlit)
with open("data/features_cancer_resample_rf.txt", "w") as f:
    for feature in final_feature_names_for_model_input:
        f.write(feature + "\n")
print("Noms des features finales (encodées) sauvegardés sous 'data/features_cancer_resample_rf.txt'")

