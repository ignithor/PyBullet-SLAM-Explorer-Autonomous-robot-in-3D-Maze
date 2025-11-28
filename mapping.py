import numpy as np

# --- Constantes ---
# Doivent correspondre aux valeurs physiques du robot
LIDAR_RANGE = 5.0    # Portée max du Lidar en mètres
LIDAR_RAYS = 36      # Nombre de rayons (doit correspondre au robot.py)

class Slam:
    """
    Implémente un algorithme de SLAM par Grille d'Occupation (Occupancy Grid).
    
    Le principe est de diviser le monde en une grille. Chaque case contient
    une valeur 'log-odds' qui représente la probabilité d'être un obstacle.
    """
    
    # --- Constantes Log-Odds (Probabilités) ---
    # On utilise des logs pour pouvoir additionner les probabilités simplement.
    # > 0 : Probablement un obstacle
    # < 0 : Probablement libre
    LOG_ODDS_HIT = 0.9    # Valeur ajoutée si le laser touche un obstacle (Bonus)
    LOG_ODDS_FREE = 0.4   # Valeur soustraite si le laser traverse la case (Malus)
    LOG_ODDS_CLAMP = 20.0 # Valeur min/max pour éviter l'infini (saturation)

    def __init__(self, map_size_m=40.0, map_resolution=0.1):
        """
        Initialise la grille.
        
        Args:
            map_size_m (float): Taille du monde en mètres (ex: 40x40m).
            map_resolution (float): Taille d'une case en mètres (ex: 0.1 = 10cm).
        """
        self.map_size_m = map_size_m
        self.resolution = map_resolution
        
        # Calcul du nombre de cellules (ex: 40m / 0.1m = 400 cellules de large)
        self.size_cells = int(self.map_size_m / self.resolution)
        
        # On centre la carte : le robot démarre au milieu (0,0 dans le monde)
        # origin_m est le coin bas-gauche en coordonnées monde
        self.origin_m = -self.map_size_m / 2.0
        
        # Création de la matrice de la carte (remplie de 0 = inconnu)
        self.map = np.zeros((self.size_cells, self.size_cells), dtype=np.float32)
        
        # Pré-calcul des angles du Lidar (0 à 2*Pi) pour gagner du temps
        # 0° = Droite du robot, 90° = Devant
        self.lidar_angles = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False)

        print(f"INFO: SLAM initialisé. Carte: {self.size_cells}x{self.size_cells} cellules ({map_size_m}x{map_size_m}m).")

    def world_to_grid(self, x_world, y_world):
        """
        Étape 2 : Conversion Coordonnées Monde (mètres) -> Coordonnées Grille (indices).
        """
        # Formule : (Position - Origine) / Résolution
        x_grid = int((x_world - self.origin_m) / self.resolution)
        y_grid = int((y_world - self.origin_m) / self.resolution)

        # Vérification si on est hors de la carte
        if 0 <= x_grid < self.size_cells and 0 <= y_grid < self.size_cells:
            return (x_grid, y_grid)
        return None

    def update(self, robot_pos, robot_yaw, lidar_data):
        """
        Étape 3 & 4 : Met à jour la carte avec une nouvelle lecture du Lidar.
        
        Args:
            robot_pos (list): [x, y] position actuelle du robot.
            robot_yaw (float): Angle actuel du robot (orientation).
            lidar_data (array): Liste des distances mesurées par le Lidar.
        """
        robot_x, robot_y = robot_pos
        
        # On utilise des 'set' pour stocker les cases uniques à mettre à jour
        # (évite de mettre à jour 10 fois la même case pour un seul scan)
        free_cells = set()
        occupied_cells = set()

        # Pour chaque rayon du Lidar...
        for i, dist in enumerate(lidar_data):
            # 1. Calculer l'angle réel du rayon dans le monde
            world_angle = robot_yaw + self.lidar_angles[i]
            
            # 2. Calculer la position exacte de l'impact (ou fin de rayon)
            hit_x = robot_x + dist * np.cos(world_angle)
            hit_y = robot_y + dist * np.sin(world_angle)
            
            # Vérifier si c'est une détection réelle ou juste la portée max (ciel)
            is_max_range = (dist >= LIDAR_RANGE - 0.1)

            # --- Ray Tracing (Algorithme géométrique) ---
            # On parcourt la ligne entre le robot et l'impact
            vec_x = hit_x - robot_x
            vec_y = hit_y - robot_y
            num_steps = int(dist / self.resolution) # Nombre de cases à traverser

            for step in range(num_steps):
                # On avance petit à petit sur la ligne
                t = step / num_steps if num_steps > 0 else 0
                p_x = robot_x + (t * vec_x)
                p_y = robot_y + (t * vec_y)
                
                cell = self.world_to_grid(p_x, p_y)
                if cell:
                    free_cells.add(cell) # Cette case est LIBRE

            # --- Marquage de l'obstacle ---
            if not is_max_range:
                hit_cell = self.world_to_grid(hit_x, hit_y)
                if hit_cell:
                    occupied_cells.add(hit_cell) # Cette case est OCCUPÉE

        # --- Mise à jour de la matrice (Log-Odds) ---
        
        # Marquer les cases libres (Soustraction)
        for (gx, gy) in free_cells:
            # On ne marque pas libre une case qu'on vient de détecter comme occupée
            if (gx, gy) not in occupied_cells:
                 self.map[gy, gx] -= self.LOG_ODDS_FREE

        # Marquer les cases occupées (Addition)
        for (gx, gy) in occupied_cells:
            self.map[gy, gx] += self.LOG_ODDS_HIT

        # Saturation (Clamp) pour ne pas avoir des valeurs infinies
        np.clip(self.map, -self.LOG_ODDS_CLAMP, self.LOG_ODDS_CLAMP, out=self.map)

    def get_map_probabilities(self):
        """
        Étape 5 : Convertit la carte log-odds en probabilités (0.0 à 1.0) pour l'affichage.
        0.0 = Blanc (Libre), 1.0 = Noir (Mur), 0.5 = Gris (Inconnu)
        """
        # Formule Sigmoïde : p = exp(val) / (1 + exp(val))
        exp_map = np.exp(self.map)
        prob_map = exp_map / (1.0 + exp_map)
        return prob_map