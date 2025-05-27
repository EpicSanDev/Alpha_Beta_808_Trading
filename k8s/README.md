# AlphaBeta808 Trading Bot - Kubernetes Deployment

Ce dossier contient tous les fichiers nécessaires pour déployer le bot de trading AlphaBeta808 sur Kubernetes.

## 🏗️ Architecture du déploiement

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────┤
│  Namespace: alphabeta808-trading                            │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │   Web Interface │  │   Trading Bot   │                 │
│  │                 │  │                 │                 │
│  │  - Flask App    │  │  - Live Trading │                 │
│  │  - Dashboard    │  │  - ML Models    │                 │
│  │  - API          │  │  - Risk Mgmt    │                 │
│  └─────────────────┘  └─────────────────┘                 │
│           │                     │                          │
│  ┌─────────────────────────────────────────┐               │
│  │        Persistent Storage               │               │
│  │                                         │               │
│  │  ├─ Trading Data (10Gi)                │               │
│  │  ├─ ML Models (5Gi)                    │               │
│  │  └─ Logs (2Gi)                         │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Structure des fichiers

```
k8s/
├── deploy.sh              # Script de déploiement automatique
├── undeploy.sh            # Script de suppression
├── manage.sh              # Script de gestion et monitoring
├── namespace.yaml         # Namespace Kubernetes
├── configmap.yaml         # Configuration de l'application
├── secrets.yaml           # Secrets (clés API, mots de passe)
├── pvc.yaml              # Volumes persistants
├── rbac.yaml             # Permissions et rôles
├── web-deployment.yaml    # Déploiement interface web
├── bot-deployment.yaml    # Déploiement bot de trading
├── services.yaml          # Services Kubernetes
├── ingress.yaml           # Exposition externe (optionnel)
├── hpa.yaml              # Auto-scaling horizontal
├── pdb.yaml              # Pod Disruption Budget
└── monitoring.yaml        # Monitoring Prometheus (optionnel)
```

## 🚀 Déploiement rapide

### Prérequis

1. **Cluster Kubernetes** fonctionnel
2. **kubectl** configuré
3. **Docker** installé
4. **Clés API Binance** (testnet recommandé)

### Installation en une commande

```bash
# Déploiement complet
./k8s/deploy.sh

# Ou avec options
./k8s/deploy.sh v1.0 --deploy-ingress
```

### Configuration des secrets

1. **Option 1: Utiliser le fichier .env**
```bash
# Créer/modifier le fichier .env
cat > .env << EOF
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
WEBHOOK_SECRET=your_webhook_secret
WEB_ADMIN_USER=admin
WEB_ADMIN_PASSWORD=secure_password_123
EOF

# Le script deploy.sh détectera automatiquement le fichier
./k8s/deploy.sh
```

2. **Option 2: Modifier secrets.yaml**
```bash
# Encoder vos clés en base64
echo -n "your_api_key" | base64

# Modifier k8s/secrets.yaml avec vos valeurs encodées
# Puis déployer
./k8s/deploy.sh
```

## 📊 Gestion du déploiement

### Commandes utiles

```bash
# Voir le statut
./k8s/manage.sh status

# Voir les logs du bot
./k8s/manage.sh logs bot

# Voir les logs de l'interface web
./k8s/manage.sh logs web

# Se connecter à un pod
./k8s/manage.sh shell bot

# Port forwarding pour accès local
./k8s/manage.sh port-forward web 8080

# Redémarrer les services
./k8s/manage.sh restart all

# Scaler l'interface web
./k8s/manage.sh scale web 3
```

### Accès à l'interface web

1. **Via LoadBalancer** (si supporté par votre cluster)
```bash
kubectl get svc trading-web-loadbalancer -n alphabeta808-trading
# Utiliser l'EXTERNAL-IP affichée
```

2. **Via Port Forward**
```bash
./k8s/manage.sh port-forward web 8080
# Ouvrir http://localhost:8080
```

3. **Via Ingress** (si déployé)
```bash
# Modifier trading.yourdomain.com dans ingress.yaml
# Puis accéder via votre domaine
```

## 🔧 Configuration

### Variables d'environnement importantes

- `BINANCE_API_KEY`: Clé API Binance
- `BINANCE_API_SECRET`: Secret API Binance  
- `WEBHOOK_SECRET`: Secret pour les webhooks
- `WEB_ADMIN_USER`: Utilisateur admin interface web
- `WEB_ADMIN_PASSWORD`: Mot de passe admin

### Configuration du trading

La configuration est centralisée dans `configmap.yaml`. Paramètres clés:

```yaml
trading:
  testnet: true              # Utiliser le testnet Binance
  symbols: ["BTCUSDT", ...]  # Paires à trader
  max_concurrent_positions: 5 # Positions max simultanées

risk_management:
  max_position_size: 0.10    # 10% max par position
  max_daily_loss: 0.02       # 2% perte max par jour
  stop_loss_percentage: 0.05 # Stop loss à 5%
```

## 🔄 Mise à jour

### Mise à jour de l'application

```bash
# Reconstruire et redéployer
./k8s/deploy.sh v1.1

# Ou juste redémarrer avec la nouvelle image
kubectl set image deployment/trading-bot trading-bot=alphabeta808/trading-bot:v1.1 -n alphabeta808-trading
kubectl set image deployment/trading-web-interface web-interface=alphabeta808/trading-bot:v1.1 -n alphabeta808-trading
```

### Mise à jour de la configuration

```bash
# Modifier configmap.yaml puis:
kubectl apply -f k8s/configmap.yaml

# Redémarrer pour prendre en compte les changements
./k8s/manage.sh restart all
```

## 📈 Monitoring

### Logs

```bash
# Logs en temps réel
kubectl logs -f deployment/trading-bot -n alphabeta808-trading
kubectl logs -f deployment/trading-web-interface -n alphabeta808-trading

# Logs précédents (si le pod a redémarré)
kubectl logs deployment/trading-bot -n alphabeta808-trading --previous
```

### Métriques

```bash
# Ressources utilisées
kubectl top pods -n alphabeta808-trading
kubectl top nodes

# Statut des déploiements
kubectl get deployments -n alphabeta808-trading
kubectl describe deployment trading-bot -n alphabeta808-trading
```

### Auto-scaling

L'HPA (Horizontal Pod Autoscaler) est configuré pour l'interface web:
- Scale automatique basé sur CPU (70%) et mémoire (80%)
- Min: 1 réplique, Max: 3 répliques

```bash
# Voir le statut HPA
kubectl get hpa -n alphabeta808-trading
kubectl describe hpa trading-web-hpa -n alphabeta808-trading
```

## 🛡️ Sécurité

### Bonnes pratiques implémentées

1. **Utilisateur non-root** dans les containers
2. **Secrets Kubernetes** pour les données sensibles
3. **RBAC** avec permissions minimales
4. **Resource limits** pour éviter la surconsommation
5. **Health checks** pour la fiabilité
6. **Pod Disruption Budget** pour la haute disponibilité

### Recommandations

1. **Utilisez le testnet** en production jusqu'à validation complète
2. **Monitorer les logs** régulièrement
3. **Sauvegarder** les données persistantes
4. **Tester** les mises à jour sur un environnement de staging
5. **Configurer** des alertes Prometheus/Grafana

## 🔍 Troubleshooting

### Problèmes courants

1. **Pod en CrashLoopBackOff**
```bash
# Voir les logs pour identifier le problème
kubectl logs deployment/trading-bot -n alphabeta808-trading
kubectl describe pod -l app=trading-bot -n alphabeta808-trading
```

2. **Impossible d'accéder à l'interface web**
```bash
# Vérifier que le service est exposé
kubectl get svc -n alphabeta808-trading
./k8s/manage.sh port-forward web 8080
```

3. **Erreurs de connexion API Binance**
```bash
# Vérifier les secrets
kubectl get secrets trading-secrets -n alphabeta808-trading -o yaml
# Vérifier la configuration
kubectl get configmap trading-config -n alphabeta808-trading -o yaml
```

4. **Manque d'espace disque**
```bash
# Voir l'utilisation des PVC
kubectl get pvc -n alphabeta808-trading
kubectl describe pvc trading-data-pvc -n alphabeta808-trading
```

### Commandes de debug

```bash
# État général
./k8s/manage.sh status

# Événements récents
kubectl get events -n alphabeta808-trading --sort-by=.metadata.creationTimestamp

# Description détaillée d'un pod
kubectl describe pod <pod-name> -n alphabeta808-trading

# Exécuter des commandes dans un pod
./k8s/manage.sh shell bot
./k8s/manage.sh shell web
```

## 🗑️ Suppression

```bash
# Suppression complète (ATTENTION: supprime tout!)
./k8s/undeploy.sh

# Suppression sélective
kubectl delete deployment trading-bot -n alphabeta808-trading
kubectl delete pvc trading-data-pvc -n alphabeta808-trading
```

## 📞 Support

Pour obtenir de l'aide:

1. Vérifier les logs: `./k8s/manage.sh logs bot`
2. Vérifier le statut: `./k8s/manage.sh status`
3. Consulter la documentation du projet principal
4. Créer une issue sur le repository GitHub

---

**⚠️ Important**: Ce déploiement est configuré par défaut en mode testnet. Assurez-vous de bien comprendre les risques avant de passer en mode live trading.
