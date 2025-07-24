import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import base64

# chemin absolu du répertoire du script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Note: You have two logo_path definitions, one used and one commented out.
# Ensure 'images/Logo_fight_cancer_app.png' exists in your app's directory.
# logo_path = os.path.join(current_dir, "Logo_fight_cancer_app.png") # This path seems unused

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Évaluation du risque de Cancer",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load and display main logo
try:
    with open("images/Logo_fight_cancer_app.png", "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    # HTML to center image
    image_html = f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-top: 10px; margin-bottom: 20px;">
            <img src="data:image/png;base64,{encoded}" alt="Logo" width="500">
        </div>
    """
    with st.container():
        st.markdown(image_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("Error: 'images/Logo_fight_cancer_app.png' not found. Please ensure the image exists in the 'images' directory.")

# --- Custom CSS for styling ---
st.markdown("""
<style>
    body {
                background: linear-gradient(to bottom, #fcd2e2, #fff8cc);
            }
            .stApp {
                background: linear-gradient(to bottom, #fcd2e2, #fff8cc);
            }

    .header-bar img {
        max-height: 80px; /* Adjust logo size */
    }
    .header-bar h1 {
        color: #8B4513; 
        margin: 0;
        font-size: 2.5em;
        text-align: center;
    }
    .stButton {
    display: flex;
    justify-content: center;
    margin: 10px 0; 
    }

    .stButton > button {
        background-color: #FF69B4; 
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 1em;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #E05BA2;
        color: white;
    }
    .stAlert {
        border-radius: 5px;
    }
    .stSuccess {
        background-color: #e6ffe6;
        color: #2e8b57;
        font-size: 28px;
    }
    .stWarning {
        background-color: #fffacd;
        color: #daa520;
    }
    .stError {
        background-color: #ffe6e6;
        color: #dc143c;
    }
    h2 {
        color: #8B4513; 
        text-align: center;
    }
    .stSlider .st-bb {
        background-color: #FF69B4; 
    }
    .stRadio div[role="radiogroup"] label {
        color: #8B4513; 
    }
    .stSelectbox div[data-baseweb="select"] div {
        color: #8B4513; 
    }
</style>
""", unsafe_allow_html=True)

# --- Header with Logo and Title ---
st.markdown('<div class="header-bar"> <h1>Votre risque de cancer en quelques questions</h1> </div>', unsafe_allow_html=True)

st.markdown("""
    Bienvenue,<br> <br> 
    Ce questionnaire vise à estimer votre **risque potentiel de développer un cancer** en se basant sur divers facteurs de mode de vie et de santé. <br> <br>
    **Ce n'est pas un diagnostic médical.** Pour toute préoccupation de santé, veuillez consulter un professionnel.
""", unsafe_allow_html=True)

raw_features_for_pipeline_input = [
    'CutSkipMeals2', 'DiffPayMedBills', 'SmokeNow',
    'MedConditions_Diabetes', 'MedConditions_HighBP', 'MedConditions_HeartCondition',
    'MedConditions_LungDisease', 'MedConditions_Depression',
    'GeneralHealth', 'HealthLimits_Pain', 'Nervous',
    'IncomeRanges', 'Education', 'Birthcountry',
    'Fruit2', 'Vegetables2', 'TimesStrengthTraining', 'Drink_nb_PerMonth',
    'ChildrenInHH', 'TotalHousehold', 'TimesSunburned', 'BMI', 'Age', 'SleepWeekdayHr'
]

# Load the model
pipeline = None
model_output_features = raw_features_for_pipeline_input
try:
    pipeline = joblib.load("modele_cancer_resample_rf.pkl")
    st.success("✅ Pipeline chargé avec succès.")
    with open("data/features_cancer_resample_rf.txt", "r") as f:
        model_output_features = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("❌ Erreur : 'modele_cancer_resample_rf.pkl' ou 'data/features_cancer_resample_rf.txt' non trouvé. Assurez-vous que les fichiers du modèle sont dans le bon répertoire.")
    # Fallback mock pipeline if model is not found
    class MockPipeline:
        def predict(self, X): return [0] * len(X)
        def predict_proba(self, X): return [[0.9, 0.1] for _ in range(X.shape[0])]
    pipeline = MockPipeline()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du pipeline : {e}")
    # Fallback mock pipeline
    class MockPipeline:
        def predict(self, X): return [0] * len(X)
        def predict_proba(self, X): return [[0.9, 0.1] for _ in range(X.shape[0])]
    pipeline = MockPipeline()


# Initialize session state for steps and inputs
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.inputs = {}
    st.session_state.form_submitted = False # Track if the final form is submitted

# --- Form Steps ---

if st.session_state.step == 0:
    with st.form("step_0_form"):
        st.markdown("<h2>Un peu de vous </h2>", unsafe_allow_html=True)

        col_age, col_bmi = st.columns(2)
        with col_age:
            prenom = st.text_input("Quel est votre prénom ou surnom ?", value=st.session_state.inputs.get("prenom", ""), help="facultatif, il sera utilisé pour personnaliser les résultats")
            poids = st.number_input("Quel est votre poids (kg) ?", 30.0, 300.0, float(st.session_state.inputs.get("poids", 70.0)))
            
        with col_bmi:
            Age = st.number_input("Quel est votre âge ?", 18, 120, int(st.session_state.inputs.get("Age", 30)))
            taille_m = st.number_input("Quelle est votre taille (en mètres) ?", 1.0, 2.4, float(st.session_state.inputs.get("taille_m", 1.70)))
        
        submit_step0 = st.form_submit_button("Continuer")

        if submit_step0:
            # Store inputs for this step
            st.session_state.inputs["prenom"] = prenom
            st.session_state.inputs["Age"] = Age
            st.session_state.inputs["poids"] = poids
            st.session_state.inputs["taille_m"] = taille_m
            st.session_state.step = 1 # Move to next step
            st.rerun() # Rerun to display the next step

elif st.session_state.step == 1:       
    with st.form("step_1_form"):
        st.markdown("<h2> Votre mode de vie </h2>", unsafe_allow_html=True)
        col_smoke, col_alcohol = st.columns(2)
        with col_smoke:
            fumeur = st.selectbox("Fumez-vous actuellement ?", ["Jamais", "Quelques fois", "Tous les jours"], index=["Jamais", "Quelques fois", "Tous les jours"].index(st.session_state.inputs.get("fumeur", "Jamais")), help="Indiquez si vous êtes un fumeur actuel.")
        with col_alcohol:
            alcohol = st.number_input("Nombre de verres d'alcool par mois ?", 0, 500, int(st.session_state.inputs.get("alcohol", 7)), help="Estimation du nombre de verres standards d'alcool consommés par mois.")
        
        col_sun, col_sleep = st.columns(2)
        with col_sun:
            soleil = st.number_input("Les 12 derniers mois, avez-vous eu des coups de soleil ? Si oui, combien ?", 0, 300, int(st.session_state.inputs.get("soleil", 0)), help="Le nombre de fois où votre peau a été brûlée par le soleil, causant rougeur et douleur.")
        with col_sleep:
            sommeil_moy = st.number_input("Durée moyenne de sommeil par jour (en heures) ?", 0.0, 24.0, float(st.session_state.inputs.get("sommeil_moy", 7.0)), help="Nombre moyen d'heures de sommeil par 24 heures.")

        submit_step1 = st.form_submit_button("Continuer")

        if submit_step1:
            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 2:
    with st.form("step_2_form"):
        st.markdown("<h2>Votre santé</h2>", unsafe_allow_html=True)
        col_health1, col_health2, col_health3 = st.columns(3)
        with col_health1:
            diabete = st.radio("Avez-vous du diabète ?", ["Oui", "Non"], index=["Oui", "Non"].index(st.session_state.inputs.get("diabete", "Non")))
            cardiaque = st.radio("Avez-vous des problèmes cardiaques ?", ["Oui", "Non"], index=["Oui", "Non"].index(st.session_state.inputs.get("cardiaque", "Non")))
            
        with col_health2:
            hypertension = st.radio("Avez-vous de l'hypertension ?", ["Oui", "Non"], index=["Oui", "Non"].index(st.session_state.inputs.get("hypertension", "Non")))
            poumon = st.radio("Avez-vous des problèmes pulmonaires ?", ["Oui", "Non"], index=["Oui", "Non"].index(st.session_state.inputs.get("poumon", "Non")))
        with col_health3:
            depression = st.radio("Souffrez-vous de dépression ?", ["Oui", "Non"], index=["Oui", "Non"].index(st.session_state.inputs.get("depression", "Non")))
            douleur = st.radio("Souffrez-vous de douleur chronique ?", ["Oui", "Non"], index=["Oui", "Non"].index(st.session_state.inputs.get("douleur", "Non")))
        nervous_options = ["Très faible, je suis relax", "Faible, quelques fois", "Modéré, sous pression la moitié du temps", "Élevé, stressé(e) tous les jours"]
        nervous = st.selectbox("Quel est votre niveau de stress ?", nervous_options, index=nervous_options.index(st.session_state.inputs.get("nervous", "Très faible, je suis relax")))
        sante_general_options = ["Faible", "Moyen", "Bon", "Très bon : On va danser ce soir ?", "Excellent : Je pète la forme !"]
        Sante_general = st.selectbox("Comment évaluez-vous votre santé générale ?", sante_general_options, index=sante_general_options.index(st.session_state.inputs.get("Sante_general", "Bon")))
        
        submit_step2 = st.form_submit_button("Continuer")

        if submit_step2:
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
        st.markdown("<h2>Nutrition et Activités</h2>", unsafe_allow_html=True)
        col_fruit, col_veg, col_sport = st.columns(3)
        fruit_veg_options = ["0", "1/2 portion ou moins", "1/2 à 1 portion", "1 à 2 portions", "2 à 3 portions", "3 à 4 portions", "plus de 4"]
        with col_fruit:
            fruits = st.select_slider("Combien de portions de fruits mangez-vous par jour ?", fruit_veg_options, index=fruit_veg_options.index(st.session_state.inputs.get("fruits", "2 à 3 portions")), help="Une portion correspond à une pomme moyenne, une banane, ou une tasse de petits fruits.")
        with col_veg:
            legumes = st.select_slider("Combien de portions de légumes mangez-vous par jour ?", fruit_veg_options, index=fruit_veg_options.index(st.session_state.inputs.get("legumes", "2 à 3 portions")), help="Une portion correspond à une tasse de légumes verts à feuilles ou une demi-tasse de légumes coupés.")
        with col_sport:
            sport = st.select_slider("Combien de jours par semaine faites-vous de l'exercice intense ?", ["0", "1", "2", "3", "4", "5", "6", "7"], index=["0", "1", "2", "3", "4", "5", "6", "7"].index(str(st.session_state.inputs.get("sport", "3"))), help="Nombre de séance de cardio, renforcement musculaire par semaine.")

        if submit_step3:
            st.session_state.step = 4
            st.rerun()

elif st.session_state.step == 4:
        st.markdown("<h2>Home, sweet home </h2>", unsafe_allow_html=True)
        revenu_options = [
            " 0 à 730€ mensuel", "730€ à 1099€ mensuel", "1100€ à 1469€ mensuel",
            "1470€ à 2569€ mensuel", "2570€ à 3669€ mensuel", "3670 à 5499€ mensuel",
            "5500€ à 7339€ mensuel", "7340€ à 14669€ mensuel", "14670€ mensuel et plus"]
        revenu = st.select_slider("Quel est votre revenu annuel net approximatif ?", revenu_options, index=revenu_options.index(st.session_state.inputs.get("revenu", "1470€ à 2569€ mensuel")), help="Veuillez sélectionner la tranche qui correspond le mieux à votre revenu annuel net.")
        etude_options = [
            "Primaire", "Collège / brevet", "Lycée / BAC", "Universitaire : BTS / DUT / filière technique",
            "Universitaire : Licence / Maîtrise / DEUG", "Universitaire : Master / DEA / DESS",
            "Doctorat ou plus"]
        etude = st.select_slider("Quel est votre niveau d'études le plus élevé atteint ?", etude_options, index=etude_options.index(st.session_state.inputs.get("etude", "Lycée / BAC")), help="Votre plus haut diplôme ou niveau de scolarité atteint.")
        
        col_children, col_household = st.columns(2)
        with col_children:
            enfants = st.number_input("Combien d'enfants avez-vous?", 0, 30, int(st.session_state.inputs.get("enfants", 0)))
        with col_household:
            foyer = st.number_input("Combien de personnes vivent avec vous?", 0, 30, int(st.session_state.inputs.get("foyer", 1)), help="Adultes, enfants et vous compris")

        diff_options = ["Jamais", "Un peu", "Souvent"]
        Diff_financiere = st.selectbox("Avez-vous des difficultés à payer vos factures médicales ?", diff_options, index=diff_options.index(st.session_state.inputs.get("Diff_financiere", "Jamais")), help="Indiquez la fréquence de vos difficultés à couvrir les frais médicaux.")
        Skip_meal = st.selectbox("Avez-vous déjà sauté des repas en raison de difficultés financières au cours des 12 derniers mois ?", diff_options, index=diff_options.index(st.session_state.inputs.get("Skip_meal", "Jamais")), help="Indiquez si vous avez dû sauter des repas en raison de contraintes financières.")
        ethnie_options = [
            "Blanc", "Noir Africain ou Noir Americain", "Indien Américain, Américain du nord",
            "Indien d'Asie", "Chinois", "Philippin", "Japonais", "Coréen", "Vietnamien",
            "Autre Asiatique", "Autre île du Pacifique", "Autre origine"]
        ethnie = st.selectbox("Quelle est votre origine ethnique ?", ethnie_options, index=ethnie_options.index(st.session_state.inputs.get("ethnie", "Blanc")), help="Cette information est utilisée à des fins statistiques et d'amélioration du modèle.")

        submit_step3 = st.form_submit_button("Calculer mon risque de cancer")

        if submit_step3:

            st.session_state.form_submitted = True
            st.rerun() # Rerun to display results

# --- Display Results ---
if st.session_state.form_submitted:
    data = st.session_state.inputs
    imc = round(data["poids"] / (data["taille_m"] ** 2), 2) if data["taille_m"] > 0 else 0.0
    nom = data["prenom"].strip() or "Cher utilisateur" # More professional default

    # Mappings for model input
    fumeur_num = {"Jamais": 0, "Quelques fois": 1, "Tous les jours": 2}
    stress_map = {"Très faible, je suis relax": 0, "Faible, quelques fois": 1, "Modéré, sous pression la moitié du temps": 2, "Élevé, stressé(e) tous les jours": 3}
    revenu_map = {
        " 0 à 730€ mensuel": 1, "730€ à 1099€ mensuel": 2, "1100€ à 1469€ mensuel": 3,
        "1470€ à 2569€ mensuel": 4, "2570€ à 3669€ mensuel": 5, "3670 à 5499€ mensuel": 6,
        "5500€ à 7339€ mensuel": 7, "7340€ à 14669€ mensuel": 8, "14670€ mensuel et plus": 9
    }
    etude_map = {
        "Primaire": 1, "Collège / brevet": 2, "Lycée / BAC": 3, "Universitaire : BTS / DUT / filière technique": 4,
        "Universitaire : Licence / Maîtrise / DEUG": 5, "Universitaire : Master / DEA / DESS": 6,
        "Doctorat ou plus": 7
    }
    bool_map = {"Oui": 1, "Non": 0}
    sante_general_map = {"Faible": 0, "Moyen": 1, "Bon": 2, "Très bon : On va danser ce soir ?": 3, "Excellent : Je pète la forme !": 4}
    fruits_map = {"0": 0, "1/2 portion ou moins": 1, "1/2 à 1 portion": 2, "1 à 2 portions": 3,
                  "2 à 3 portions": 4, "3 à 4 portions": 5, "plus de 4": 6}
    legumes_map = fruits_map
    diff_map = {"Jamais": 0, "Un peu": 1, "Souvent": 2}
    ethnie_map = { # change english to french 
        "Blanc": "White", "Noir Africain ou Noir Americain": "Black", "Indien Américain, Américain du nord": "AmerInd",
        "Indien d'Asie": "AsInd", "Chinois": "Chinese", "Philippin": "Filipino", "Japonais": "Japanese",
        "Coréen": "Korean", "Vietnamien": "Vietnamese", "Autre Asiatique": "OthAsian",
        "Autre île du Pacifique": "OthPacIsl", "Autre origine": "Other"
    }

    input_data_for_pipeline = {
        "Age": data["Age"],
        "BMI": imc,
        "SmokeNow": fumeur_num[data["fumeur"]],
        "SleepWeekdayHr": data["sommeil_moy"],
        "Nervous": stress_map[data["nervous"]],
        "IncomeRanges": revenu_map[data["revenu"]],
        "Education": etude_map[data["etude"]],
        "MedConditions_Diabetes": bool_map[data["diabete"]],
        "MedConditions_HighBP": bool_map[data["hypertension"]],
        "MedConditions_HeartCondition": bool_map[data["cardiaque"]],
        "MedConditions_LungDisease": bool_map[data["poumon"]],
        "MedConditions_Depression": bool_map[data["depression"]],
        "HealthLimits_Pain": bool_map[data["douleur"]],
        "Fruit2": fruits_map[data["fruits"]],
        "Vegetables2": legumes_map[data["legumes"]],
        "TimesStrengthTraining": int(data["sport"]),
        "Drink_nb_PerMonth": data["alcohol"],
        "ChildrenInHH": data["enfants"],
        "TotalHousehold": data["foyer"],
        "TimesSunburned": data["soleil"],
        "GeneralHealth": sante_general_map[data["Sante_general"]],
        "Birthcountry": ethnie_map[data["ethnie"]],
        "CutSkipMeals2": diff_map[data["Skip_meal"]],
        "DiffPayMedBills": diff_map[data["Diff_financiere"]]
    }

    df_form = pd.DataFrame([input_data_for_pipeline])
    df_form = df_form.reindex(columns=raw_features_for_pipeline_input)

    try:
        if pipeline: # Ensure pipeline is loaded before predicting
            Y_prediction_proba = pipeline.predict_proba(df_form)
            score = round(Y_prediction_proba[0][1] * 100, 0)

            st.markdown(f"<h2> {nom}, résultat de votre test</h2>", unsafe_allow_html=True)
            
            # Result slider
            st.markdown(f"""
                <style>
                    .progress-bar-container {{
                        background: #edebeb;
                        border-radius: 20px;
                        height: 30px;
                        width: 100%;
                        max-width: 100%;
                        box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
                        position: relative;
                        margin: 15px 0;
                    }}
                    .progress-bar-fill {{
                        height: 100%;
                        width: {score}%;
                        background: linear-gradient(90deg, #c6f1c6 0%, #f7c6c7 100%);
                        border-radius: 20px 0 0 20px;
                        box-shadow: 0 0 4px rgba(247, 198, 199, 0.3);
                    }}
                    .progress-bar-text {{
                        position: absolute;
                        width: 100%;
                        text-align: center;
                        top: 0;
                        line-height: 30px;
                        font-weight: 600;
                        color: #333;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        user-select: none;
                    }}
                    .risk-level-labels {{
                    display: flex;
                    justify-content: space-between;
                    font-weight: 600;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    color: #555;
                    margin: 0 5px 15px 5px;
                    user-select: none;
                    font-size: 0.9rem;
                </style>

                <div class="progress-bar-container">
                    <div class="progress-bar-fill"></div>
                    <div class="progress-bar-text">{score} %</div>
                </div>

                <div class="risk-level-labels">
                <span>Faible</span>
                <span>Modéré</span>
                <span>Fort</span>
                </div>
            """, unsafe_allow_html=True)


            if score <= 20 :
                st.success(f"**Félicitations {nom} !** Votre score de risque est faible. Cela suggère que vos habitudes actuelles sont globalement favorables à une bonne santé. Continuez à prendre soin de vous et à maintenir ces pratiques saines.")
                st.image("images/Smile.png", caption="Continuez sur cette voie de bien-être !")
            elif score <= 38 :
                st.warning(f"**Attention {nom},** Votre score indique un risque modéré. Ce n'est pas une fatalité, mais un signal pour envisager quelques ajustements dans votre mode de vie. De petits changements peuvent faire une grande différence pour votre bien-être futur. Nous vous encourageons à explorer les facteurs qui pourraient contribuer à ce risque et à discuter de ces points avec un professionnel de la santé.")
                st.image("images/soso_smiley.png", caption="De petits pas peuvent mener à de grands changements.")
            else:
                st.error(f"**Important {nom} :** Votre score est élevé. Il est crucial de comprendre que ceci n’est pas un diagnostic médical, mais un indicateur d'un risque potentiellement plus élevé. Nous vous recommandons vivement de consulter un professionnel de la santé pour une évaluation approfondie et des conseils personnalisés. Un examen médical permettra de mieux comprendre votre situation et de discuter des mesures préventives ou de suivi appropriées.")
                st.image("images/sad-emoji.png", caption="Prenez votre santé en main, consultez un professionnel")
        else:
            st.warning("Le modèle n'a pas pu être chargé. Impossible de calculer le risque.")

    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul de la prédiction. Veuillez vérifier vos données et réessayer. Détails de l'erreur : {e}")

    st.markdown("---")
    if st.button(" Recommencer l'évaluation"):
        st.session_state.form_submitted = False
        st.session_state.inputs = {}
        st.session_state.step = 0 # Reset to the first step
        st.rerun()

# --- Footer Logo ---
try:
    with open("images/Onco-sisters_logo-nobackground.png", "rb") as image_file:
        logo_encoded = base64.b64encode(image_file.read()).decode()

    logo_html = f"""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_encoded}" alt="Onco-Sisters Logo" width="200">
        <p style="margin-top: 5px; font-size: 14px; color: gray;">Powered by Onco-Sisters</p>
    </div>
"""
    st.markdown(logo_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("Error: 'images/Onco-sisters_logo-nobackground.png' not found. Please ensure the image exists in the 'images' directory.")