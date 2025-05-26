#!/bin/bash

# Script pour exécuter tous les tests du projet

# Se placer dans le répertoire racine du projet (si nécessaire)
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# cd "$SCRIPT_DIR"

echo "Début de l'exécution des tests..."

# Exécuter les tests unitaires
echo "Exécution des tests unitaires..."
python -m unittest discover -s tests.unit -p "test_*.py"

# Exécuter les tests d'intégration
echo "Exécution des tests d'intégration..."
python -m unittest discover -s tests.integration -p "test_*.py"

# Exécuter les tests de performance
echo "Exécution des tests de performance..."
python -m unittest discover -s tests.performance -p "test_*.py"

# Exécuter les tests de robustesse des données
echo "Exécution des tests de robustesse des données..."
python -m unittest discover -s tests.data_robustness -p "test_*.py"

echo "Fin de l'exécution des tests."

# Pour utiliser avec pytest (alternative, si pytest est installé et configuré):
# echo "Exécution des tests avec pytest..."
# pytest tests/