#!/bin/bash

# AlphaBeta808 Trading Bot - Railway Deployment Helper
# Ce script aide à préparer et déployer sur Railway

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}🚂 AlphaBeta808 Trading Bot - Railway Deployment Helper${NC}"
echo "================================================================"

# Fonction pour générer des clés sécurisées
generate_keys() {
    echo -e "\n${PURPLE}🔐 Génération de clés sécurisées${NC}"
    
    if command -v openssl &> /dev/null; then
        SECRET_KEY=$(openssl rand -base64 32)
        WEBHOOK_SECRET=$(openssl rand -base64 32)
        
        echo -e "${GREEN}✅ Clés générées avec succès !${NC}"
        echo ""
        echo -e "${YELLOW}📋 Copiez ces valeurs dans Railway Dashboard → Variables:${NC}"
        echo ""
        echo -e "${BLUE}SECRET_KEY=${NC}${SECRET_KEY}"
        echo -e "${BLUE}WEBHOOK_SECRET=${NC}${WEBHOOK_SECRET}"
        echo ""
        echo -e "${YELLOW}💾 Sauvegardez ces clés dans un endroit sûr !${NC}"
    else
        echo -e "${RED}❌ OpenSSL non trouvé. Installez OpenSSL pour générer des clés sécurisées.${NC}"
        echo -e "${YELLOW}Alternative: utilisez un générateur de mots de passe en ligne${NC}"
    fi
}

# Fonction pour vérifier les prérequis
check_prerequisites() {
    echo -e "\n${PURPLE}📋 Vérification des prérequis${NC}"
    
    # Vérifier Git
    if command -v git &> /dev/null; then
        echo -e "${GREEN}✅ Git installé${NC}"
    else
        echo -e "${RED}❌ Git non installé${NC}"
        exit 1
    fi
    
    # Vérifier les fichiers Railway
    if [ -f "Dockerfile.railway" ]; then
        echo -e "${GREEN}✅ Dockerfile.railway présent${NC}"
    else
        echo -e "${RED}❌ Dockerfile.railway manquant${NC}"
        exit 1
    fi
    
    if [ -f ".railwayignore" ]; then
        echo -e "${GREEN}✅ .railwayignore présent${NC}"
    else
        echo -e "${YELLOW}⚠️  .railwayignore manquant (optionnel)${NC}"
    fi
    
    if [ -f "Procfile" ]; then
        echo -e "${GREEN}✅ Procfile présent${NC}"
    else
        echo -e "${RED}❌ Procfile manquant${NC}"
        exit 1
    fi
}

# Fonction pour vérifier le repository Git
check_git_status() {
    echo -e "\n${PURPLE}📚 Vérification du repository Git${NC}"
    
    if [ ! -d ".git" ]; then
        echo -e "${RED}❌ Pas un repository Git${NC}"
        echo -e "${YELLOW}Initialisez avec: git init${NC}"
        return 1
    fi
    
    # Vérifier s'il y a des changes non commitées
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW}⚠️  Changements non commitées détectés${NC}"
        echo -e "${BLUE}Voulez-vous committer les changements ? (y/n)${NC}"
        read -p "Commit: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git add .
            git commit -m "Préparation pour déploiement Railway"
            echo -e "${GREEN}✅ Changements commitées${NC}"
        fi
    else
        echo -e "${GREEN}✅ Repository à jour${NC}"
    fi
    
    # Vérifier remote origin
    if git remote get-url origin &> /dev/null; then
        ORIGIN_URL=$(git remote get-url origin)
        echo -e "${GREEN}✅ Remote origin configuré: ${ORIGIN_URL}${NC}"
    else
        echo -e "${YELLOW}⚠️  Pas de remote origin configuré${NC}"
        echo -e "${BLUE}Ajoutez votre repository GitHub:${NC}"
        echo -e "${BLUE}git remote add origin https://github.com/username/AlphaBeta808Trading.git${NC}"
    fi
}

# Fonction principale de déploiement
deploy_guide() {
    echo -e "\n${PURPLE}🚀 Guide de déploiement Railway${NC}"
    echo ""
    echo -e "${BLUE}1. Aller sur Railway${NC}"
    echo "   → Visitez https://railway.app"
    echo "   → Connectez-vous avec GitHub"
    echo ""
    echo -e "${BLUE}2. Créer un nouveau projet${NC}"
    echo "   → Cliquez 'New Project'"
    echo "   → Sélectionnez 'Deploy from GitHub repo'"
    echo "   → Choisissez votre repository AlphaBeta808Trading"
    echo "   → Railway détectera automatiquement le Dockerfile.railway"
    echo ""
    echo -e "${BLUE}3. Configurer les variables d'environnement${NC}"
    echo "   → Allez dans Settings → Variables"
    echo "   → Ajoutez les variables suivantes:"
    echo ""
    echo -e "${YELLOW}   Variables OBLIGATOIRES:${NC}"
    echo "   - SECRET_KEY (généré ci-dessus)"
    echo "   - WEBHOOK_SECRET (généré ci-dessus)"
    echo "   - WEB_ADMIN_USER (ex: admin)"
    echo "   - WEB_ADMIN_PASSWORD (choisissez un mot de passe fort)"
    echo "   - BINANCE_API_KEY (votre clé API Binance)"
    echo "   - BINANCE_API_SECRET (votre secret API Binance)"
    echo ""
    echo -e "${YELLOW}   Variables OPTIONNELLES:${NC}"
    echo "   - EMAIL_ENABLED=false"
    echo "   - TELEGRAM_ENABLED=false"
    echo ""
    echo -e "${BLUE}4. Déployer${NC}"
    echo "   → Railway détecte automatiquement les projets Docker"
    echo "   → Aucun fichier de configuration supplémentaire nécessaire"
    echo "   → Le déploiement commencera automatiquement"
    echo "   → Surveillez les logs de build"
    echo ""
    echo -e "${BLUE}5. Vérifier le déploiement${NC}"
    echo "   → Votre app sera disponible à: your-app.railway.app"
    echo "   → Testez: your-app.railway.app/health"
    echo "   → Connectez-vous avec vos identifiants admin"
    echo ""
    echo -e "${GREEN}🎯 Déploiement terminé !${NC}"
}

# Fonction pour afficher les conseils post-déploiement
post_deployment_tips() {
    echo -e "\n${PURPLE}💡 Conseils post-déploiement${NC}"
    echo ""
    echo -e "${YELLOW}Sécurité:${NC}"
    echo "• Changez immédiatement le mot de passe admin par défaut"
    echo "• Utilisez le mode sandbox Binance pour les tests"
    echo "• Activez l'authentification à deux facteurs sur Binance"
    echo ""
    echo -e "${YELLOW}Monitoring:${NC}"
    echo "• Surveillez les logs Railway pour détecter les erreurs"
    echo "• Vérifiez régulièrement l'endpoint /health"
    echo "• Configurez les alertes email/Telegram si nécessaire"
    echo ""
    echo -e "${YELLOW}Trading:${NC}"
    echo "• Commencez avec le mode paper trading"
    echo "• Testez avec de petits montants d'abord"
    echo "• Surveillez les performances et ajustez les paramètres"
    echo ""
    echo -e "${YELLOW}Maintenance:${NC}"
    echo "• Sauvegardez régulièrement vos configurations"
    echo "• Mettez à jour les dépendances périodiquement"
    echo "• Surveillez l'utilisation des ressources Railway"
}

# Menu principal
show_menu() {
    echo -e "\n${BLUE}Que voulez-vous faire ?${NC}"
    echo "1. Vérifier les prérequis"
    echo "2. Générer des clés sécurisées"
    echo "3. Vérifier le statut Git"
    echo "4. Afficher le guide de déploiement"
    echo "5. Conseils post-déploiement"
    echo "6. Tout faire (recommandé)"
    echo "0. Quitter"
    echo ""
    read -p "Votre choix: " choice
    
    case $choice in
        1)
            check_prerequisites
            ;;
        2)
            generate_keys
            ;;
        3)
            check_git_status
            ;;
        4)
            deploy_guide
            ;;
        5)
            post_deployment_tips
            ;;
        6)
            check_prerequisites
            generate_keys
            check_git_status
            deploy_guide
            post_deployment_tips
            ;;
        0)
            echo -e "${GREEN}Au revoir !${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Choix invalide${NC}"
            show_menu
            ;;
    esac
}

# Boucle principale
while true; do
    show_menu
    echo ""
    read -p "Appuyez sur Entrée pour continuer ou Ctrl+C pour quitter..."
done
