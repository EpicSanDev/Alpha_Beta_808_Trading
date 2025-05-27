# 🚀 Guide de déploiement Kubernetes - AlphaBeta808 Trading Bot

Ce guide vous accompagne pour déployer votre bot de trading sur Kubernetes de manière complète et sécurisée.

## 📋 Vue d'ensemble du déploiement

Votre infrastructure Kubernetes comprendra :

- **2 applications principales** : Bot de trading + Interface web
- **Stockage persistant** : Données, modèles ML, logs
- **Gestion des secrets** : Clés API sécurisées
- **Monitoring** : Health checks et métriques
- **Auto-scaling** : Adaptation automatique de charge
- **Load balancing** : Distribution de charge avec Nginx

## 🏗️ Architecture déployée

```
Internet
    ↓
[LoadBalancer] ← Nginx Proxy
    ↓
[Trading Web Interface] ← Flask App + Dashboard
    ↓
[Trading Bot] ← Live Trading + ML Models
    ↓
[Persistent Storage] ← Data + Models + Logs
```

## 🔧 Étapes de déploiement

### 1. Préparation de l'environnement

```bash
# Vérifier que kubectl fonctionne
kubectl cluster-info

# Cloner ou naviguer vers votre projet
cd /Users/bastienjavaux/Desktop/AlphaBeta808Trading
```

### 2. Configuration des clés API

Créez un fichier `.env` avec vos vraies clés :

```bash
cat > .env << 'EOF'
BINANCE_API_KEY=your_real_binance_api_key
BINANCE_API_SECRET=your_real_binance_secret
WEBHOOK_SECRET=your_webhook_secret_key
WEB_ADMIN_USER=admin
WEB_ADMIN_PASSWORD=your_secure_password
EOF
```

⚠️ **Important** : Gardez `testnet: true` dans la configuration pour les tests !

### 3. Déploiement automatique

```bash
# Déploiement complet en une commande
./k8s/deploy.sh latest

# Ou avec plus d'options
./k8s/deploy.sh v1.0 --deploy-ingress --skip-push
```

Le script va :
1. ✅ Construire l'image Docker
2. ✅ Créer le namespace Kubernetes 
3. ✅ Configurer les secrets
4. ✅ Déployer tous les composants
5. ✅ Attendre que tout soit prêt

### 4. Vérification du déploiement

```bash
# Voir le statut général
./k8s/manage.sh status

# Vérifier que les pods sont prêts
kubectl get pods -n alphabeta808-trading
```

Vous devriez voir quelque chose comme :
```
NAME                                      READY   STATUS    RESTARTS   AGE
trading-bot-xxx-xxx                       1/1     Running   0          2m
trading-web-interface-xxx-xxx             1/1     Running   0          2m
```

## 🌐 Accès à l'interface web

### Option 1 : Port forwarding (recommandé pour les tests)

```bash
# Créer un tunnel vers l'interface web
./k8s/manage.sh port-forward web 8080

# Ouvrir dans votre navigateur
open http://localhost:8080
```

### Option 2 : LoadBalancer (si supporté par votre cluster)

```bash
# Voir l'IP externe
kubectl get svc trading-web-loadbalancer -n alphabeta808-trading

# Utiliser l'EXTERNAL-IP affichée
```

### Option 3 : Ingress avec domaine personnalisé

```bash
# Modifier k8s/ingress.yaml avec votre domaine
# Puis déployer l'ingress
./k8s/deploy.sh --deploy-ingress
```

## 📊 Surveillance et gestion

### Voir les logs en temps réel

```bash
# Logs du bot de trading
./k8s/manage.sh logs bot

# Logs de l'interface web
./k8s/manage.sh logs web
```

### Se connecter aux pods pour debugging

```bash
# Connexion au bot
./k8s/manage.sh shell bot

# Connexion à l'interface web  
./k8s/manage.sh shell web
```

### Redémarrer les services

```bash
# Redémarrer tout
./k8s/manage.sh restart all

# Redémarrer juste le bot
./k8s/manage.sh restart bot
```

### Scaler selon la charge

```bash
# Augmenter le nombre d'instances web
./k8s/manage.sh scale web 3

# L'HPA gère l'auto-scaling automatiquement
kubectl get hpa -n alphabeta808-trading
```

## 🔄 Mise à jour de l'application

### Mise à jour du code

```bash
# Après avoir modifié le code
./k8s/deploy.sh v1.1

# Ou mise à jour forcée
./k8s/deploy.sh latest --skip-push
./k8s/manage.sh restart all
```

### Mise à jour de la configuration

```bash
# Modifier k8s/configmap.yaml
kubectl apply -f k8s/configmap.yaml

# Redémarrer pour appliquer les changements
./k8s/manage.sh restart all
```

## 🛡️ Sécurité et bonnes pratiques

### Secrets et mots de passe

- ✅ Les clés API sont stockées dans des secrets Kubernetes
- ✅ Utilisateur non-root dans les containers
- ✅ Permissions RBAC minimales
- ✅ Isolation des namespaces

### Monitoring de sécurité

```bash
# Vérifier les secrets
kubectl get secrets -n alphabeta808-trading

# Vérifier les permissions
kubectl auth can-i --list --as=system:serviceaccount:alphabeta808-trading:trading-bot-sa -n alphabeta808-trading
```

## 🎯 Optimisations de performance

### Ressources CPU/Mémoire

Les limites sont pré-configurées :
- **Bot de trading** : 1-2 CPU, 1-2Gi RAM
- **Interface web** : 0.25-0.5 CPU, 512Mi-1Gi RAM

### Auto-scaling intelligent

```bash
# L'HPA scale automatiquement basé sur :
# - CPU > 70%
# - Mémoire > 80%
kubectl describe hpa trading-web-hpa -n alphabeta808-trading
```

### Stockage persistant

- **Données de trading** : 10Gi
- **Modèles ML** : 5Gi  
- **Logs** : 2Gi

## 🚨 Résolution de problèmes

### Problème : Pod en erreur

```bash
# Voir les détails de l'erreur
kubectl describe pod <pod-name> -n alphabeta808-trading

# Voir les logs d'erreur
kubectl logs <pod-name> -n alphabeta808-trading --previous
```

### Problème : Connexion API Binance

```bash
# Vérifier les secrets
kubectl get secret trading-secrets -n alphabeta808-trading -o yaml

# Tester la connectivité depuis un pod
./k8s/manage.sh shell bot
# Dans le pod : python -c "import os; print(os.getenv('BINANCE_API_KEY'))"
```

### Problème : Interface web inaccessible

```bash
# Vérifier les services
kubectl get svc -n alphabeta808-trading

# Port forward manuel
kubectl port-forward svc/trading-web-service 8080:5000 -n alphabeta808-trading
```

### Problème : Manque d'espace disque

```bash
# Voir l'utilisation des volumes
kubectl get pvc -n alphabeta808-trading
df -h # dans les pods via shell
```

## 🎯 Environnements (Dev/Staging/Prod)

### Déploiement pour différents environnements

```bash
# Développement (testnet)
./k8s/deploy.sh dev --skip-push

# Staging (testnet avec données réelles)  
./k8s/deploy.sh staging

# Production (live trading) - ATTENTION !
./k8s/deploy.sh prod --deploy-ingress
```

Modifiez `configmap.yaml` pour chaque environnement :
- `testnet: true/false`
- `symbols: [...]` (différentes paires)
- `risk_management.*` (limites différentes)

## 📈 Monitoring avancé (optionnel)

### Prometheus et Grafana

```bash
# Déployer le monitoring
kubectl apply -f k8s/monitoring.yaml

# Ajouter des métriques custom dans votre code Flask
```

### Alertes importantes

- Bot arrêté > 1 minute
- Utilisation mémoire > 90%
- Utilisation CPU > 80%
- Erreurs API Binance

## 🗑️ Suppression complète

```bash
# Suppression sécurisée avec confirmations
./k8s/undeploy.sh

# Suppression forcée (DANGER !)
kubectl delete namespace alphabeta808-trading --force
```

## ✅ Checklist finale

Avant de mettre en production :

- [ ] Tests complets en mode testnet
- [ ] Clés API de production configurées
- [ ] Limites de risque appropriées
- [ ] Monitoring configuré
- [ ] Sauvegardes des données importantes
- [ ] Plan de rollback préparé
- [ ] Alertes configurées
- [ ] Documentation d'équipe mise à jour

## 📞 Support

En cas de problème :

1. **Logs** : `./k8s/manage.sh logs bot`
2. **Statut** : `./k8s/manage.sh status`  
3. **Events** : `kubectl get events -n alphabeta808-trading`
4. **Documentation** : `k8s/README.md`

---

**🎉 Félicitations !** Votre bot de trading AlphaBeta808 est maintenant déployé sur Kubernetes avec une infrastructure robuste, scalable et sécurisée !

**⚠️ Rappel de sécurité** : Commencez toujours par le testnet Binance avant de passer en live trading.
