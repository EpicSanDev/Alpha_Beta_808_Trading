#!/bin/bash

# AlphaBeta808 Trading Bot - Production Setup Script
# Ce script configure l'environnement pour la production

set -e

echo "🚀 AlphaBeta808 Trading Bot - Production Setup"
echo "=============================================="

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "trader_config.json" ]; then
    log_error "Veuillez exécuter ce script depuis le répertoire racine d'AlphaBeta808Trading"
    exit 1
fi

log_info "Configuration de l'environnement de production..."

# 1. Créer les répertoires nécessaires
log_info "Création des répertoires..."
mkdir -p logs
mkdir -p backtest_results
mkdir -p models_store
mkdir -p optimized_models
mkdir -p reports
mkdir -p results

# 2. Configuration des variables d'environnement
log_info "Configuration des variables d'environnement..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        log_warning "Fichier .env créé depuis .env.example. Veuillez le configurer avec vos clés API."
    else
        log_info "Création du fichier .env..."
        cat > .env << EOF
# AlphaBeta808 Trading Bot - Production Environment Variables
# IMPORTANT: Configurez ces variables avec vos vraies clés API

# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Web Interface Authentication
WEB_ADMIN_USER=admin
WEB_ADMIN_PASSWORD=change_this_password_in_production

# Webhook Security
WEBHOOK_SECRET=your_webhook_secret_here

# Monitoring Configuration
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin123

# Docker Registry (if using custom registry)
DOCKER_REGISTRY=docker.io

# Application Settings
LOG_LEVEL=INFO
FLASK_ENV=production
PYTHONPATH=/app/src:/app

# Database Configuration
DATABASE_URL=sqlite:///trading_web.db

# Redis Configuration (for caching and queues)
REDIS_URL=redis://redis:6379/0

# Risk Management
MAX_TRADE_AMOUNT=1000
MAX_PORTFOLIO_EXPOSURE=10000
DAILY_LOSS_LIMIT=500

# Notification Settings
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
EOF
        log_success "Fichier .env créé avec les variables par défaut"
        log_warning "IMPORTANT: Configurez le fichier .env avec vos vraies clés API avant le déploiement!"
    fi
else
    log_success "Fichier .env existant trouvé"
fi

# 3. Vérification de la configuration Python
log_info "Vérification de l'environnement Python..."
if [ -d "trading_env" ]; then
    log_success "Environnement virtuel trading_env trouvé"
    source trading_env/bin/activate
    
    # Vérifier les dépendances critiques
    log_info "Vérification des dépendances..."
    python -c "
import sys
missing = []
try:
    import pandas
    import numpy
    import sklearn
    import xgboost
    import yfinance
    print('✅ Dépendances principales: OK')
except ImportError as e:
    missing.append(str(e))
    print(f'❌ Dépendances manquantes: {e}')

try:
    from src.modeling.tensorflow_compat import TENSORFLOW_AVAILABLE
    if TENSORFLOW_AVAILABLE:
        print('✅ TensorFlow: Disponible')
    else:
        print('⚠️  TensorFlow: Non disponible (mode compatibilité)')
except ImportError:
    print('❌ Module de compatibilité TensorFlow non trouvé')

if missing:
    print(f'ERREUR: {len(missing)} dépendances manquantes')
    sys.exit(1)
else:
    print('✅ Toutes les dépendances sont satisfaites')
"
    if [ $? -eq 0 ]; then
        log_success "Toutes les dépendances Python sont satisfaites"
    else
        log_error "Des dépendances Python sont manquantes"
        log_info "Exécution de: pip install -r requirements.txt"
        pip install -r requirements.txt
    fi
else
    log_warning "Environnement virtuel non trouvé, création en cours..."
    python3 -m venv trading_env
    source trading_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    log_success "Environnement virtuel créé et configuré"
fi

# 4. Test du système complet
log_info "Test du système complet..."
python system_verification.py
if [ $? -eq 0 ]; then
    log_success "Système vérifié avec succès - prêt pour la production"
else
    log_error "Le système a échoué aux tests de vérification"
    log_info "Veuillez corriger les erreurs avant de continuer"
    exit 1
fi

# 5. Configuration Docker
log_info "Vérification de Docker..."
if command -v docker &> /dev/null; then
    log_success "Docker trouvé"
    
    # Construction de l'image Docker
    log_info "Construction de l'image Docker..."
    docker build -t alphabeta808/trading-bot:latest .
    if [ $? -eq 0 ]; then
        log_success "Image Docker construite avec succès"
    else
        log_error "Échec de la construction de l'image Docker"
        exit 1
    fi
else
    log_warning "Docker non trouvé. Installation manuelle requise pour le déploiement conteneurisé"
fi

# 6. Configuration Kubernetes (optionnel)
log_info "Vérification de kubectl..."
if command -v kubectl &> /dev/null; then
    log_success "kubectl trouvé"
    
    # Vérifier la connexion au cluster
    if kubectl cluster-info &> /dev/null; then
        log_success "Connexion au cluster Kubernetes établie"
        log_info "Fichiers de déploiement Kubernetes disponibles dans k8s/"
        log_info "Exécutez 'k8s/deploy.sh' pour déployer sur Kubernetes"
    else
        log_warning "kubectl trouvé mais pas de connexion au cluster"
        log_info "Configurez kubectl pour vous connecter à votre cluster"
    fi
else
    log_warning "kubectl non trouvé. Déploiement Kubernetes non disponible"
fi

# 7. Configuration de la surveillance
log_info "Configuration des outils de surveillance..."
if [ -f "monitoring/prometheus.yml" ]; then
    log_success "Configuration Prometheus trouvée"
else
    log_warning "Configuration Prometheus manquante"
fi

# 8. Vérification des modèles ML
log_info "Vérification des modèles ML..."
if [ -d "models_store" ] && [ "$(ls -A models_store)" ]; then
    model_count=$(ls -1 models_store/*.joblib 2>/dev/null | wc -l)
    log_success "$model_count modèle(s) ML trouvé(s) dans models_store/"
else
    log_warning "Aucun modèle ML pré-entraîné trouvé"
    log_info "Les modèles seront entraînés lors du premier lancement"
fi

# 9. Permissions de sécurité
log_info "Configuration des permissions de sécurité..."
chmod 600 .env
chmod 755 *.sh
chmod 755 k8s/*.sh
log_success "Permissions de sécurité configurées"

echo ""
echo "🎉 Configuration de production terminée!"
echo "======================================="
echo ""
log_info "Prochaines étapes:"
echo "  1. Configurez vos clés API dans le fichier .env"
echo "  2. Testez le système avec: python system_verification.py"
echo "  3. Pour Docker: docker-compose up -d"
echo "  4. Pour Kubernetes: cd k8s && ./deploy.sh"
echo "  5. Accédez à l'interface web: http://localhost:5000"
echo "  6. Surveillez avec Grafana: http://localhost:3000"
echo ""
log_warning "IMPORTANT: Sauvegardez régulièrement les modèles et les données de trading!"
echo ""
if [ ! -f ".env" ]; then
    log_info "Création du fichier .env depuis .env.example..."
    cp .env.example .env
    log_warning "IMPORTANT: Éditez le fichier .env avec vos vraies clés API !"
else
    log_warning "Le fichier .env existe déjà"
fi

# 3. Vérifier Python et les dépendances
log_info "Vérification de Python..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 n'est pas installé"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_success "Python $PYTHON_VERSION détecté"

# 4. Créer un environnement virtuel si nécessaire
if [ ! -d "venv" ]; then
    log_info "Création de l'environnement virtuel..."
    python3 -m venv venv
    log_success "Environnement virtuel créé"
fi

# 5. Activer l'environnement virtuel et installer les dépendances
log_info "Installation des dépendances..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6. Vérifier la configuration
log_info "Vérification de la configuration..."
python3 -c "
import json
try:
    with open('trader_config.json', 'r') as f:
        config = json.load(f)
    print('✅ Configuration trader_config.json valide')
    
    # Vérifier le mode testnet
    if config.get('trading', {}).get('testnet', True):
        print('⚠️  Mode TESTNET activé (sécurisé pour les tests)')
    else:
        print('🚨 Mode LIVE TRADING activé (attention !)')
        
except Exception as e:
    print(f'❌ Erreur dans trader_config.json: {e}')
    exit(1)
"

# 7. Test des importations critiques
log_info "Test des importations Python..."
python3 -c "
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import flask
    import binance
    print('✅ Toutes les dépendances critiques sont installées')
except ImportError as e:
    print(f'❌ Dépendance manquante: {e}')
    exit(1)
"

# 8. Créer les scripts de service
log_info "Création des scripts de service..."

# Script de démarrage
cat > start_production.sh << 'EOF'
#!/bin/bash
# Script de démarrage pour la production
cd "$(dirname "$0")"
source venv/bin/activate
export FLASK_ENV=production
python3 bot_manager.py start --background
python3 web_interface/app_enhanced.py &
echo "✅ AlphaBeta808 démarré en mode production"
echo "📊 Interface web: http://localhost:5000"
echo "📝 Logs: tail -f logs/trading_bot.log"
EOF

chmod +x start_production.sh

# Script d'arrêt
cat > stop_production.sh << 'EOF'
#!/bin/bash
# Script d'arrêt pour la production
cd "$(dirname "$0")"
source venv/bin/activate
python3 bot_manager.py stop
pkill -f "app_enhanced.py"
echo "✅ AlphaBeta808 arrêté"
EOF

chmod +x stop_production.sh

# 9. Vérifications de sécurité
log_info "Vérifications de sécurité..."

# Vérifier les permissions des fichiers sensibles
if [ -f ".env" ]; then
    chmod 600 .env
    log_success "Permissions .env sécurisées (600)"
fi

# Vérifier la configuration de production dans trader_config.json
TESTNET_MODE=$(python3 -c "import json; print(json.load(open('trader_config.json'))['trading']['testnet'])")
if [ "$TESTNET_MODE" = "True" ]; then
    log_success "Mode TESTNET activé (recommandé pour les premiers tests)"
else
    log_warning "Mode LIVE TRADING activé - Assurez-vous d'avoir testé en mode testnet"
fi

# 10. Instructions finales
echo ""
echo "🎉 Configuration terminée avec succès !"
echo "======================================="
echo ""
echo "📋 Prochaines étapes :"
echo "1. Éditez .env avec vos vraies clés API Binance"
echo "2. Testez d'abord avec testnet: true dans trader_config.json"
echo "3. Démarrage : ./start_production.sh"
echo "4. Monitoring : ./bot_manager.py status"
echo "5. Interface web : http://localhost:5000"
echo "6. Arrêt : ./stop_production.sh"
echo ""
echo "🔒 Sécurité :"
echo "- Gardez vos clés API secrètes"
echo "- Commencez toujours par le testnet"
echo "- Surveillez les logs régulièrement"
echo "- Définissez des limites de risque appropriées"
echo ""
echo "📚 Documentation : Consultez README.md et KUBERNETES_DEPLOYMENT_GUIDE.md"
