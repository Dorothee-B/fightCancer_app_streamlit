# Random Forest training script for the Streamlit cancer risk detection application

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib
import os
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# SMOTE importation for resampling
from imblearn.over_sampling import SMOTE # to do: pip install imbalanced-learn!!
from imblearn.pipeline import Pipeline
# data
try:
    df = pd.read_csv('data/df2.csv')
except FileNotFoundError:
    print("Erreur: Le fichier 'df2.csv' n'a pas été trouvé. Veuillez vous assurer qu'il est dans le même répertoire.")
    exit()

cancer_oui= df[df['EverHadCancer'] == 1]  # people who had cancer
cancer_non = df[df['EverHadCancer'] == 0]

# manual resampling to balance classes , 1073 = df[df['EverHadCancer'] == 1
resample_cancer_oui = cancer_oui.sample(n=1073, random_state=142)
resample_cancer_non = cancer_non.sample(n=1073, random_state=142)

# Combine the two subsamples
balanced_df = pd.concat([resample_cancer_oui, resample_cancer_non])


# Shuffle lines to avoid order
balanced_df = balanced_df.sample(frac=1, random_state=142).reset_index(drop=True)
print(balanced_df['EverHadCancer'].value_counts())

# Variables (defined in logical order for the preprocessor)
binary_vars = [
    'MedConditions_Diabetes', 'MedConditions_HighBP', 'MedConditions_HeartCondition',
    'MedConditions_LungDisease', 'MedConditions_Depression','FamilyEverHadCancer2', 
    'HealthLimits_Pain' 
]
ordinal_categorical_vars = [ 
    'SmokeNow','GeneralHealth', 'Nervous',
    'IncomeRanges', 'Education', 'Fruit2', 'Vegetables2', 'CutSkipMeals2',
    'DiffPayMedBills'   
]
birthsex_var = ['BirthSex']  # categorial encoded [0,1,2] for ordinal_ohe (int64)

string_categorical_vars = ['Birthcountry']  # string

continuous_vars = ['BMI', 'Age', 
                   'SleepWeekdayHr', 'TimesSunburned', 'TimesStrengthTraining',
                   'ChildrenInHH', 'TotalHousehold', 'Drink_nb_PerMonth'
                   ]

target = 'EverHadCancer'


features_raw = ordinal_categorical_vars + birthsex_var + string_categorical_vars + continuous_vars + binary_vars

df_clean = balanced_df[features_raw + [target]].dropna()


X = df_clean[features_raw]
y = df_clean[target]


# Mapping ordinal cat var (same as Streamlit app)
smoke_categories = ["Jamais", "Quelques fois", "Tous les jours"]
sante_categories = ["Faible", "Moyen", "Bon", "Très bon : On va danser ce soir ?", "Excellent : Je pète la forme !"]
nervous_categories = ["Très faible, je suis relax", "Faible, quelques fois", "Modéré, sous pression la moitié du temps", "Élevé, stressé(e) tous les jours"]
revenu_categories = [" 0 à 730€ mensuel", "730€ à 1099€ mensuel", "1100€ à 1469€ mensuel", "1470€ à 2569€ mensuel", "2570€ à 3669€ mensuel", "3670 à 5499€ mensuel", "5500€ à 7339€ mensuel", "7340€ à 14669€ mensuel", "14670€ mensuel et plus"]
etude_categories = ["Primaire", "Collège / brevet", "Lycée / BAC", "Universitaire : BTS / DUT / filière technique", "Universitaire : Licence / Maîtrise / DEUG", "Universitaire : Master / DEA / DESS", "Doctorat ou plus"]
fruit_veg_categories = ["0", "1/2 portion ou moins", "1/2 à 1 portion", "1 à 2 portions", "2 à 3 portions", "3 à 4 portions", "plus de 4"]
diff_med_categories = ["Jamais", "Un peu", "Souvent"]
cut_skip_categories = ["Jamais", "Un peu", "Souvent"]

# Mapping
ordinal_categories = [
    smoke_categories,
    sante_categories,
    nervous_categories,
    revenu_categories,
    etude_categories,
    fruit_veg_categories, # Fruit2
    fruit_veg_categories, # Vegetables2
    cut_skip_categories,
    diff_med_categories
]

# Preprocessor with ColumnTransformer
# The order of transformations here defines the output order of features in the pipeline
# and therefore the order expected by the final model


transformers = [
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_categorical_vars),
    ('ohe_birthsex', OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False, handle_unknown='ignore'), birthsex_var),
    ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), string_categorical_vars),
    ('scale', StandardScaler(), continuous_vars)
]

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='passthrough'  # binary variables pass without transformation
)



# Create the complete pipeline, including SMOTE
# SMOTE is applied AFTER the preprocessor but BEFORE the classifier

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=142)), # SMOTE
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=142))
])

# Split et train
# The entire pipeline, including preprocessing and SMOTE, will be applied to X_train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)
pipeline.fit(X_train, y_train)

# model evaluation
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Feature names extraction
# This is important for the Streamlit app to know which features are used in the model
# Note: The order of features in the final matrix must match the order expected by the model

# 1. Features ordinales
ord_features = ordinal_categorical_vars

# 2. Features OneHotEncoded
ohe_transformer = pipeline.named_steps['preprocess'].named_transformers_['ohe']
# get_feature_names_out donne les noms corrects comme 'Birthcountry_Mexican'
ohe_features = list(ohe_transformer.get_feature_names_out(string_categorical_vars))

# 3. Features OneHotEncoded pour BirthSex
ohe_birthsex = pipeline.named_steps['preprocess'].named_transformers_['ohe_birthsex']
ohe_birthsex_features = list(ohe_birthsex.get_feature_names_out(birthsex_var))

# 4. Features continues
cont_features = continuous_vars

# 5. Binary Features (passed directly, 'remainder' of the ColumnTransformer)
bin_features = binary_vars

# # The order of the features in the final matrix (this order must be respected by the application)
final_feature_names_for_model_input = (
    ord_features +
    ohe_birthsex_features+
    ohe_features +
    cont_features +
    bin_features
)

# Features importance
try:
    
    # Retrieve the classifier after SMOTE and preprocessing
    model_trained = pipeline.named_steps['model']
    # RandomForestClassifier, il aura feature_importances_
    if hasattr(model_trained, 'feature_importances_'):
        importances = model_trained.feature_importances_
        importance_df = pd.DataFrame({
            'Variable': final_feature_names_for_model_input,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Visualization Top 20
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


# Save the train pipeline (including preprocessing and SMOTE)
joblib.dump(pipeline, "modele_cancer_resample_rf.pkl")
print("Pipeline entraîné (avec SMOTE) sauvegardé sous 'modele_cancer_resample_rf.pkl'")

# Save features importance (optional)
if 'importance_df' in locals():
    importance_df.to_csv("data/importances_cancer_resample_rf.csv", index=False)
    print("Importances des features sauvegardées sous 'data/importances_cancer_resample_rf.csv'")

# Save final features name (important for streamlit app)
with open("data/features_cancer_resample_rf.txt", "w") as f:
    for feature in final_feature_names_for_model_input:
        f.write(feature + "\n")
print("Noms des features finales (encodées) sauvegardés sous 'data/features_cancer_resample_rf.txt'")

