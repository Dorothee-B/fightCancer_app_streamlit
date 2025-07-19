
# 🩺 Application Streamlit : Prédiction du Risque de Cancer

## ✅ Lancer l'application

1. Assurez-vous d’avoir Python 3 et `streamlit` installé :
   ```bash
   pip install streamlit pandas
   ```

2. Lancez l'application :
   ```bash
   streamlit run app.py
   ```

## ⚙️ Personnalisation

- Le score est simulé avec `random.uniform(0, 1)`.
- Pour brancher votre modèle :
  - Remplacez cette ligne :
    ```python
    score = random.uniform(0, 1)
    ```
  - Par votre propre prédiction, par exemple :
    ```python
    import joblib
    model = joblib.load("votre_modele.pkl")
    score = model.predict_proba(df)[0][1]
    ```

## 📁 Fichiers

- `app.py` → Application principale
- `README.txt` → Ce fichier

---

🎉 Bonne chance avec votre projet !
