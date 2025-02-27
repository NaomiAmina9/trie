import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Fonction pour segmenter la plante
def segment_plant(image):
    # Conversion en HSV pour faciliter la segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Définir les limites pour détecter la plante (vert)
    lower_green = np.array([35, 100, 100])# Seuil bas pour le vert
    upper_green = np.array([85, 255, 255]) # Seuil haut pour le vert
    
    # Créer un masque pour isoler la plante
    mask = cv2.inRange(hsv, lower_green, upper_green)# Crée un masque binaire pour le vert
    
    # Appliquer des opérations morphologiques pour nettoyer le masque
    kernel = np.ones((5, 5), np.uint8)# Définition d'un noyau pour le traitement morphologique
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Nettoyage du masque avec une fermeture morphologique
    
    return mask

# Fonction pour calculer la hauteur de la tige
def calculate_stem_height(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    
    # Trouver le contour le plus grand (la plante principale)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Encadrer le contour avec un rectangle englobant
    x, y, w, h = cv2.boundingRect(largest_contour)
    return h

# Fonction pour analyser la couleur des feuilles (indice de chlorophylle)
def analyze_leaf_color(image, mask):
    # Appliquer le masque sur l'image originale
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Convertir en espace de couleur Lab pour analyser la teinte verte
    lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2Lab)
    green_channel = lab[:, :, 1]
    
    # Calculer la moyenne de la teinte verte
    mean_green = np.mean(green_channel[green_channel > 0])
    return mean_green

# Initialisation de la capture vidéo
video_path = "pgrowing.mp4"
cap = cv2.VideoCapture(video_path)

# Liste pour stocker les données
data = []

# Boucle pour traiter chaque frame de la vidéo
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Segmenter la plante
    mask = segment_plant(frame)
    
    # Calculer la hauteur de la tige
    stem_height = calculate_stem_height(mask)
    
    # Analyser la couleur des feuilles
    mean_green = analyze_leaf_color(frame, mask)
    
    # Enregistrer les données
    data.append({
        "frame": frame_count,
        "stem_height": stem_height,
        "mean_green": mean_green
    })
    
    # Afficher les résultats (optionnel)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    
    frame_count += 1

# Libérer les ressources
cap.release()
#cv2.destroyAllWindows()

# Convertir les données en DataFrame
df = pd.DataFrame(data)

# Sauvegarder les données dans un fichier CSV
df.to_csv("pomme_de_terre_data.csv", index=False)



if "frame" in df.columns:
    plt.plot(df["frame"], df["stem_height"])
else:
    print("La colonne 'frame' n'existe pas dans df !")



# Visualiser les résultats
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df["frame"], df["stem_height"])
print(df.columns)
plt.title("Hauteur de la tige vs Temps")
plt.xlabel("Frame")
plt.ylabel("Hauteur (pixels)")





