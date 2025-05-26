import numpy as np
from sklearn.model_selection import KFold

class PurgedTimeSeriesSplit(KFold):
    """
    Validation croisée temporelle avec purge et embargo.

    Cette classe implémente une stratégie de validation croisée spécifique aux séries temporelles,
    inspirée de `TimeSeriesSplit` de scikit-learn, mais avec l'ajout de périodes de purge et d'embargo
    pour éviter les fuites d'information et réduire le surapprentissage, comme décrit par
    Marcos Lopez de Prado.

    Args:
        n_splits (int): Nombre de plis (splits) de validation. Doit être au moins 1.
        purge_duration (int): Nombre de périodes à purger à la fin de chaque ensemble d'entraînement.
                              Ces périodes sont retirées pour éviter la fuite d'information due aux
                              dépendances temporelles (par exemple, features décalées) qui pourraient
                              "voir" dans l'ensemble de test.
        embargo_duration (int): Nombre de périodes à embarguer (ignorer) au début de chaque ensemble de test,
                                immédiatement après la fin de la période d'entraînement (avant purge).
                                Cela aide à réduire le surapprentissage sur la transition entre
                                l'entraînement et le test, en créant un "no man's land".
        max_train_size (int, optional): Taille maximale pour chaque ensemble d'entraînement.
                                        Si None (par défaut), l'ensemble d'entraînement croît de manière expansive.
                                        Si une valeur est fournie, cela crée une fenêtre d'entraînement glissante.
    """
    def __init__(self, n_splits=5, purge_duration=0, embargo_duration=0, max_train_size=None):
        # KFold n'est pas un parent approprié car sa logique de split est différente.
        # On implémente notre propre logique de split.
        if not isinstance(n_splits, int) or n_splits <= 0:
            raise ValueError("n_splits doit être un entier positif.")
        if not isinstance(purge_duration, int) or purge_duration < 0:
            raise ValueError("purge_duration doit être un entier non-négatif.")
        if not isinstance(embargo_duration, int) or embargo_duration < 0:
            raise ValueError("embargo_duration doit être un entier non-négatif.")
        if max_train_size is not None and (not isinstance(max_train_size, int) or max_train_size <= 0):
            raise ValueError("max_train_size doit être un entier positif ou None.")

        self.n_splits = n_splits
        self.purge_duration = purge_duration
        self.embargo_duration = embargo_duration
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """
        Génère les indices pour séparer les données en ensembles d'entraînement et de test.

        Args:
            X (array-like): Données à séparer (features). La forme doit être (n_samples, n_features).
                            Seul n_samples est utilisé.
            y (array-like, optional): Variable cible. Ignoré.
            groups (array-like, optional): Informations de groupe pour les échantillons. Ignoré.

        Yields:
            tuple: (train_indices, test_indices)
                train_indices (ndarray): Indices de l'ensemble d'entraînement pour ce pli.
                test_indices (ndarray): Indices de l'ensemble de test pour ce pli.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if self.n_splits <= 0:
            # raise ValueError("n_splits doit être positif.") # Déjà vérifié dans __init__
            return # Ne rien céder si n_splits n'est pas valide (devrait être attrapé par __init__)

        if n_samples == 0:
            # Si pas d'échantillons, ne rien céder.
            # On pourrait aussi lever une erreur si n_splits > 0.
            return


        # Taille de base pour chaque pli de test et pour le premier ensemble d'entraînement.
        # On divise n_samples en (n_splits + 1) parts : 1 part pour l'entraînement initial,
        # et n_splits parts pour les n_splits ensembles de test.
        base_fold_size = n_samples // (self.n_splits + 1)

        if base_fold_size == 0:
            # Cela arrive si n_samples <= n_splits.
            # Par exemple, n_samples=5, n_splits=5 => base_fold_size=0.
            # Dans ce cas, on ne peut pas former des plis significatifs avec un entraînement initial.
            # On pourrait essayer de forcer test_size=1 si n_samples > n_splits,
            # mais la logique de TimeSeriesSplit est que l'entraînement initial doit exister.
            if n_samples > self.n_splits : # Assez d'échantillons pour au moins 1 par test et 1 pour train initial
                 base_fold_size = 1 # Chaque test aura 1 échantillon, le reste pour l'entraînement initial.
            else:
                 # Pas assez d'échantillons pour tous les splits et un entraînement initial.
                 # On pourrait choisir de ne générer aucun pli ou de lever une erreur.
                 # Pour être cohérent avec TimeSeriesSplit qui peut générer moins de plis,
                 # on pourrait continuer et voir si des plis valides sont formés.
                 # Cependant, une erreur est plus informative ici.
                 raise ValueError(
                    f"n_samples={n_samples} est trop petit par rapport à n_splits={self.n_splits} "
                    f"pour créer des plis de validation temporelle significatifs avec un entraînement initial, "
                    f"purge ({self.purge_duration}) et embargo ({self.embargo_duration}). "
                    f"Réduisez n_splits ou augmentez n_samples."
                )
        
        # `test_potential_starts` sont les indices où un nouveau bloc de données commence,
        # qui pourrait être utilisé soit pour étendre l'entraînement, soit pour commencer un test.
        # Le premier bloc est `0` à `base_fold_size-1`.
        # Le premier test commencera après ce premier bloc.
        # `test_boundary_before_embargo` est la fin de la période d'entraînement (avant purge)
        # et le début de la période d'embargo pour un pli donné.
        
        current_train_start_idx = 0 # Pour la fenêtre glissante

        for i in range(self.n_splits):
            # Fin de la période d'entraînement (avant purge) pour ce pli.
            # L'entraînement se fait sur les `i+1` premiers blocs.
            train_embargo_boundary = (i + 1) * base_fold_size
            
            # Début de la période de test (après embargo).
            test_actual_start = train_embargo_boundary + self.embargo_duration
            
            # Fin de la période de test. Le test est le `(i+2)`-ème bloc.
            test_actual_end = test_actual_start + base_fold_size
            
            # Ajustements pour s'assurer que les indices sont dans les limites de n_samples
            if test_actual_end > n_samples:
                test_actual_end = n_samples # Le dernier test peut être plus petit.
            
            # Vérifier si le pli est valide avant de continuer
            if test_actual_start >= n_samples or test_actual_start >= test_actual_end:
                # Plus de données disponibles pour former un ensemble de test valide.
                # Cela peut arriver si embargo_duration est grand ou si n_samples est juste.
                # print(f"Debug: Split {i}: Test set non viable. test_actual_start={test_actual_start}, test_actual_end={test_actual_end}, n_samples={n_samples}")
                continue # Passer au pli suivant (ou terminer si c'est le dernier)

            test_indices = indices[test_actual_start:test_actual_end]

            # Déterminer l'ensemble d'entraînement.
            # Il se termine à `train_embargo_boundary` (exclusif).
            train_indices_end_unpurged = train_embargo_boundary
            
            # Appliquer `max_train_size` pour déterminer le début de l'entraînement.
            # Si `max_train_size` est None, `train_actual_start` reste `0` (fenêtre expansive).
            # Sinon, c'est une fenêtre glissante.
            train_actual_start = 0 # Par défaut, fenêtre expansive
            if self.max_train_size is not None:
                train_actual_start = max(0, train_indices_end_unpurged - self.max_train_size)
            
            train_indices_unpurged = indices[train_actual_start : train_indices_end_unpurged]
            
            # Appliquer la purge à la fin de l'ensemble d'entraînement.
            train_indices_purged = train_indices_unpurged
            if self.purge_duration > 0 and len(train_indices_unpurged) > 0:
                # Ne pas purger plus que la taille de l'ensemble d'entraînement.
                effective_purge = min(self.purge_duration, len(train_indices_unpurged))
                train_indices_purged = train_indices_unpurged[:-effective_purge]
            
            # Vérifier que les deux ensembles (entraînement purgé et test) sont non vides.
            if len(train_indices_purged) > 0 and len(test_indices) > 0:
                # Vérification de non-chevauchement (devrait être garantie par la construction)
                # L'entraînement purgé se termine au plus tard à `train_embargo_boundary - 1 - purge_duration`.
                # Le test commence à `train_embargo_boundary + embargo_duration`.
                # Si purge_duration et embargo_duration sont >= 0, il n'y a pas de chevauchement.
                # La condition critique est que `train_actual_start` < `fin de train_indices_purged`.
                yield train_indices_purged, test_indices
            else:
                # Soit l'entraînement, soit le test (ou les deux) est vide après les ajustements.
                # print(f"Debug: Split {i}: Train or Test set empty. Train len: {len(train_indices_purged)}, Test len: {len(test_indices)}")
                # print(f"  train_actual_start={train_actual_start}, train_indices_end_unpurged={train_indices_end_unpurged}, effective_purge={self.purge_duration if len(train_indices_unpurged) > 0 else 0}")
                # print(f"  test_actual_start={test_actual_start}, test_actual_end={test_actual_end}")
                # On ne cède pas ce pli s'il n'est pas valide.
                pass
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Retourne le nombre de plis de division."""
        return self.n_splits