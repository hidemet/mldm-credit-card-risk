from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # ATTENZIONE: pipeline di imblearn
import pandas as pd
import numpy as np

# ... (le tue funzioni plot_confusion_matrix, evaluate_model, restano invariate) ...

def run_grid_search(sampler, model_type, param_grid, X_train, y_train, X_test, y_test, cv, class_names):
    """ Esegue la grid search con cross-validation, addestra e valuta il modello.

    Args:
        sampler:  La strategia di resampling (può essere None, un oggetto, o una stringa 'class_weight').
        model_type: Stringa, "DecisionTree" o "RandomForest".
        param_grid: Dizionario con i parametri da esplorare per il modello E per il sampler.
        X_train, y_train, X_test, y_test: Dati.
        cv: Oggetto cross-validation.
    """
    if model_type == "DecisionTree":
        classifier = DecisionTreeClassifier(random_state=42)
    elif model_type == "RandomForest":
        classifier = RandomForestClassifier(random_state=42)  # Aggiunto!
    else:
        raise ValueError("Invalid model_type. Choose 'DecisionTree' or 'RandomForest'.")


    # --- Crea la pipeline ---
    pipeline_steps = []

    if isinstance(sampler, str) and sampler == "class_weight":
      #Se la strategia è la pesatura delle classi, non abbiamo bisogno di un sampler.
      #Impostiamo class_weight nel classificatore
      classifier.set_params(class_weight="balanced")
    elif sampler is not None: # Se abbiamo una strategia di resampling
        pipeline_steps.append(('sampling', sampler))

    pipeline_steps.append(('classifier', classifier))
    pipeline = ImbPipeline(pipeline_steps) #Pipeline di imblearn

    # --- GridSearchCV ---
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_weighted',  # F1-score pesato
        cv=cv,
        n_jobs=-1,  # Usa tutti i core
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("Migliori iperparametri:", grid_search.best_params_)
    print("Miglior F1-score (media cross-validation):", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    # --- Valutazione sul Test Set ---
    test_f1 = evaluate_model(best_model, X_test, y_test, str(sampler), class_names)

    # Restituisci i risultati (opzionale, ma utile per analisi successive)
    return {
        'best_params': grid_search.best_params_,
        'best_f1_weighted_cv': grid_search.best_score_,
        'best_f1_weighted_test': test_f1,
        'best_model': best_model  # Aggiunto per comodità
    }

# --- Griglia di iperparametri (Decision Tree) ---
# NOTA:  Ora includiamo i parametri *sia* del classificatore *sia* del sampler!
tree_param_grid = {
     'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [5, 10, 20],
    'classifier__min_samples_leaf': [5, 10, 20],
    'classifier__max_features': [0.7, 'sqrt', None],
    'classifier__criterion': ['gini', 'entropy'],
    # Iperparametri per RandomOverSampler (se usato)
    'sampling__sampling_strategy': [0.5, 0.7, 1.0],  # Controlla il rapporto tra le classi
    # Iperparametri per SMOTE (se usato)
    'sampling__k_neighbors': [3, 5, 7], # Numero di vicini per SMOTE
    # Iperparametri per ADASYN (se usato)
     'sampling__n_neighbors': [3, 5, 7]
}

# --- Griglia di iperparametri (Random Forest) ---
rf_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7, None],
    'classifier__min_samples_split': [5, 10, 20],
    'classifier__min_samples_leaf': [5, 10, 20],
    'classifier__max_features': [0.7, 'sqrt', None],
    'classifier__criterion': ['gini', 'entropy'],
      # Iperparametri per RandomOverSampler (se usato)
    'sampling__sampling_strategy': [0.5, 0.7, 1.0],  # Controlla il rapporto tra le classi
    # Iperparametri per SMOTE (se usato)
    'sampling__k_neighbors': [3, 5, 7], # Numero di vicini per SMOTE
    # Iperparametri per ADASYN (se usato)
    'sampling__n_neighbors': [3, 5, 7]
}

# --- Esecuzione (Decision Tree) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
class_names = ['bad', 'good']  # Importante per il classification report

# Definisci i sampler *SENZA* random_state (o con uno diverso per ogni esecuzione)
samplers = {
    'None': None,
    'RandomOver': RandomOverSampler(), #No random state!
    'SMOTE': SMOTE(), #No random state!
    'ClassWeight': "class_weight" #Caso speciale per class_weight
}

results_dtree = {}
for sampler_name, sampler in samplers.items():
    print(f"\n{'='*20} GridSearch (Decision Tree): {sampler_name} {'='*20}")
    results_dtree[(sampler_name, "DecisionTree")] = run_grid_search(
        sampler, "DecisionTree", tree_param_grid, X_train_processed, y_train_encoded,
        X_test_processed, y_test_encoded, cv, class_names
    )

# --- Esecuzione (Random Forest) ---

results_rf = {}
for sampler_name, sampler in samplers.items():
    print(f"\n{'='*20} GridSearch (Random Forest): {sampler_name} {'='*20}")
    results_rf[(sampler_name, "RandomForest")] = run_grid_search(
        sampler, "RandomForest", rf_param_grid, X_train_processed, y_train_encoded,
        X_test_processed, y_test_encoded, cv, class_names
     )


# --- Confronto Finale e Visualizzazione (opzionale, ma consigliato) ---
# Trova la combinazione migliore (modello + strategia)
all_results = {**results_dtree, **results_rf} #Unisce i dizionari
best_combination = max(all_results, key=lambda k: all_results[k]['best_f1_weighted_test'] if all_results[k] is not None else -1)
print(f"\nMiglior Metodo: {best_combination[0]} con {best_combination[1]}")
print("Miglior F1-score (pesato) sul test set:", all_results[best_combination]['best_f1_weighted_test'])
print("Migliori iperparametri:", all_results[best_combination]['best_params_'])

# --- Visualizzazione (Opzionale) ---
# Importanza delle feature (se applicabile)
best_model = all_results[best_combination]['best_model']


# -----------------------------------------------------------------------------------------------------
def run_randomized_search(model, sampler, base_param_grid, X_train, y_train, X_test, y_test, cv, class_names, n_iter):
    """
    Esegue RandomizedSearchCV, simile a run_grid_search.
    """
    pipeline_steps = []
    if sampler:
        pipeline_steps.append(('sampling', sampler))
    pipeline_steps.append(('classifier', model))
    pipeline = ImbPipeline(pipeline_steps)

    param_distributions = base_param_grid.copy()
    if sampler:
        param_distributions['sampling__sampling_strategy'] = [0.7, 0.9, 1.0]

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        scoring='f1_weighted',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        n_iter=n_iter,  # Numero di iterazioni!
        random_state=42, # Aggiunto random_state per riproducibilità
        error_score=np.nan
    )

    random_search.fit(X_train, y_train)
  
    print("Migliori iperparametri:", random_search.best_params_)
    print("Miglior F1-score (cv):", random_search.best_score_)

    best_model = random_search.best_estimator_
    sampler_name = sampler.__class__.__name__ if sampler else "NoSampling"
    test_f1 = evaluate_model(best_model, X_test, y_test, sampler_name, class_names)

    return {
        'best_params': random_search.best_params_,
        'best_f1_weighted_cv': random_search.best_score_,
        'best_f1_weighted_test': test_f1,
        'best_model': best_model
    }