#!/bin/bash
# Script de lancement de l'interface web AlphaBeta808

echo "🌐 Démarrage de l'Interface Web AlphaBeta808"
echo "============================================="

# Vérifier si on est dans le bon répertoire
if [ ! -f "app_enhanced.py" ]; then
    echo "❌ Erreur: app_enhanced.py non trouvé"
    echo "   Assurez-vous d'être dans le répertoire web_interface/"
    exit 1
fi

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

# Vérifier et installer les dépendances
echo "📦 Vérification des dépendances..."
if [ -f "requirements.txt" ]; then
    echo "📥 Installation des dépendances Python..."
    pip3 install -r requirements.txt
else
    echo "📥 Installation des dépendances de base..."
    pip3 install flask flask-socketio flask-cors
fi

# Vérifier que le fichier de configuration du bot existe
if [ ! -f "../trader_config.json" ]; then
    echo "⚠️  Attention: trader_config.json non trouvé dans le répertoire parent"
    echo "   Le bot de trading doit être configuré pour utiliser l'interface web"
fi

# Créer le répertoire de logs s'il n'existe pas
if [ ! -d "../logs" ]; then
    echo "📁 Création du répertoire logs..."
    mkdir -p ../logs
fi

echo ""
echo "🚀 Lancement de l'interface web (app_enhanced.py)..."
echo "📊 Tableau de bord disponible sur: http://localhost:5000"
echo "🔄 Mise à jour automatique toutes les 5 secondes"
echo "🛑 Ctrl+C pour arrêter"
echo ""

# Lancer l'application
python3 app_enhanced.py
