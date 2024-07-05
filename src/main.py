import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import os

# Créer le dossier ./doc/ s'il n'existe pas déjà
os.makedirs('./doc/', exist_ok=True)

# Charger les données
df = pd.read_csv('./data/movies.csv')

# Pré-traitement
df = df.drop(columns=['Movie'])
label_encoders = {}
for column in ['LeadStudio', 'Story', 'Genre']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Traiter les valeurs manquantes
df = df.dropna(subset=['DomesticGross', 'ForeignGross', 'WorldGross'])

# Sélectionner les caractéristiques et les cibles
X = df.drop(columns=['DomesticGross', 'ForeignGross', 'WorldGross'])
y_domestic = df['DomesticGross']
y_foreign = df['ForeignGross']
y_world = df['WorldGross']

# Diviser les données pour chaque cible
X_train_domestic, X_test_domestic, y_train_domestic, y_test_domestic = train_test_split(X, y_domestic, test_size=0.2, random_state=42)
X_train_foreign, X_test_foreign, y_train_foreign, y_test_foreign = train_test_split(X, y_foreign, test_size=0.2, random_state=42)
X_train_world, X_test_world, y_train_world, y_test_world = train_test_split(X, y_world, test_size=0.2, random_state=42)

# Fonction pour entraîner et évaluer un modèle de régression
def train_and_evaluate(X_train, X_test, y_train, y_test, target_name):
    regressor = DecisionTreeRegressor(random_state=42)
    
    # Hyperparameter tuning avec GridSearchCV et k-folds cross-validation
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Meilleurs hyperparamètres
    print(f'Best parameters for {target_name}: {grid_search.best_params_}')
    best_regressor = grid_search.best_estimator_
    
    # Évaluation sur l'ensemble de test
    y_pred = best_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{target_name} - Mean Squared Error: {mse}')
    print(f'{target_name} - R^2 Score: {r2}')
    
    # Visualisation des résultats
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Valeurs Réelles', fontsize=14)
    plt.ylabel('Valeurs Prédites', fontsize=14)
    plt.title(f'{target_name} - Valeurs Réelles vs. Prédites', fontsize=16)
    plt.savefig(f'./doc/{target_name}_real_vs_pred.png')
    plt.show()

    # Visualisation de l'arbre de décision
    plt.figure(figsize=(20, 15))
    plot_tree(best_regressor, feature_names=X.columns, filled=True, fontsize=10)
    plt.title(f'{target_name} Decision Tree')
    plt.savefig(f'./doc/{target_name}_decision_tree.png')
    plt.show()
    
    return best_regressor

# Entraîner et évaluer les modèles pour chaque cible
best_regressor_domestic = train_and_evaluate(X_train_domestic, X_test_domestic, y_train_domestic, y_test_domestic, 'DomesticGross')
best_regressor_foreign = train_and_evaluate(X_train_foreign, X_test_foreign, y_train_foreign, y_test_foreign, 'ForeignGross')
best_regressor_world = train_and_evaluate(X_train_world, X_test_world, y_train_world, y_test_world, 'WorldGross')

# Sauvegarder les métriques dans des fichiers
metrics = pd.DataFrame({
    'Target': ['DomesticGross', 'ForeignGross', 'WorldGross'],
    'MSE': [
        mean_squared_error(y_test_domestic, best_regressor_domestic.predict(X_test_domestic)),
        mean_squared_error(y_test_foreign, best_regressor_foreign.predict(X_test_foreign)),
        mean_squared_error(y_test_world, best_regressor_world.predict(X_test_world))
    ],
    'R2': [
        r2_score(y_test_domestic, best_regressor_domestic.predict(X_test_domestic)),
        r2_score(y_test_foreign, best_regressor_foreign.predict(X_test_foreign)),
        r2_score(y_test_world, best_regressor_world.predict(X_test_world))
    ]
})
metrics.to_csv('./doc/metrics_regression.csv', index=False)
