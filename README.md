# Prédiction du Succès des Films au Box-Office Local, Étranger et Mondial

Ce projet vise à prédire les revenus domestiques, étrangers et mondiaux des films à l'aide de modèles de régression d'arbres de décision.

## Auteurs
ITJI Amine

## Prérequis

- Python 3.7 ou supérieur
- `pip` pour la gestion des packages

## Instructions pour l'installation et l'exécution du projet

### Windows

1. **Créer un environnement virtuel**
   ```bash
   python -m venv env
   ```

2. **Activer l'environnement virtuel**
   ```bash
   .\env\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Exécuter le script principal**
   ```bash
   python .\src\main.py
   ```

### Ubuntu

1. **Créer un environnement virtuel**
   ```bash
   python3 -m venv env
   ```

2. **Activer l'environnement virtuel**
   ```bash
   source env/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Exécuter le script principal**
   ```bash
   python3 src/main.py
   ```

## Structure du Projet

- **data/**: Contient les fichiers de données CSV
- **doc/**: Contient les figures et les visualisations générées
- **src/**: Contient le script principal (`main.py`)
- **render/**: Contient le cahier des charges, le diagramme de Gentt et le rapport
- **requirements.txt**: Liste des dépendances nécessaires au projet

## Notes

- Assurez-vous que le fichier `movies.csv` est présent dans le répertoire `data/` avant d'exécuter le script.
- Les figures et visualisations générées seront enregistrées dans le répertoire `doc/`.

## Exécution

Après avoir exécuté le script `main.py`, les résultats des prédictions et les visualisations seront disponibles dans le répertoire `doc/`.
