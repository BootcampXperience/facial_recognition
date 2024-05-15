#pip install imutils
#pip install opencv-python
#pip install face-recognition

import imutils
from imutils import paths
import os
import pickle
import cv2
import face_recognition

#Busca todos los folder dentro del folder Imagenes
folders = list(paths.list_images('Imagenes'))
knownEncodings = []
knownNames = []

#Recorre imagenes en cada folder
for (i, imagePath) in enumerate(folders):
    #Extraye el nombre del folder
    nombre = imagePath.split(os.path.sep)[-2]
    imagen = cv2.imread(imagePath)
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    #Usamos Face_recognition para localizar el rostro
    cajas = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, cajas)
    #Después de convertir cada imagen a encoding lo guardamos en listas
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(nombre)
#Convertimos las listas en diccionario
data = {"encodings": knownEncodings, "names": knownNames}
#Guardamos el diccionario en un archivo para posteriormente comparar las imagenes recibidas de la cámara
f = open("rostros_enc", "wb")
f.write(pickle.dumps(data))
f.close()

#Cargamos el Modelo haarcascade para detección de rostro frontal
faceCascade = cv2.CascadeClassifier('/Users/alejandro/miniforge3/pkgs/libopencv-4.5.5-py39h86e1ac9_9/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml')
#Cargamos nuestra base de rostros conocidos
data = pickle.loads(open('rostros_enc', "rb").read())

print("Video Iniciado")
video_capture = cv2.VideoCapture(0)

#Iniciamos la transmisión
while True:
    ret, frame = video_capture.read()
    height,width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = faceCascade.detectMultiScale(gray,
             scaleFactor=1.1,
             minNeighbors=5,
             minSize=(60, 60),
             flags=cv2.CASCADE_SCALE_IMAGE)
 
    #Recibimos la imagen de la cámara y la transformamos a encoding
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    nombres = []
    
    #Recorremos para cada rostro detectado en cámara
    for encoding in encodings:
       #Comparamos con la base de rostros conocidos
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        nombre = "Desconocido"
        #Si encontramos un rostro conocido
        if True in matches:
            #Guardamos la posición donde encontramos el rostro en nuestra base de rostros conocidos
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            #Recorremos todos los rostros que coinciden de nuestra base
            for i in matchedIdxs:
                nombre = data["names"][i]
                counts[nombre] = counts.get(nombre, 0) + 1
            #Actualizamos el nombre con el rostro que tuvo más coincidencias
            nombre = max(counts, key=counts.get)
 
        #Adicionamos nombre a la lista
        nombres.append(nombre)

        #Recorriendo los rostros detectados
        for ((x, y, w, h), nombre) in zip(rostros, nombres):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, nombre, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                
    cv2.imshow("Frame", frame)
    
    #Tecla de salida - acaba la transmisión
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video_capture.release()
        cv2.destroyAllWindows()
        break