import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Inicializa MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Cargar el scaler desde un archivo local
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Cargar tu modelo .h5 desde un archivo local
modelo = load_model('modelo_seed29_5C_5U.h5')

# Diccionario de etiquetas (número -> emoción)
label_dict = {0: 'Enojado', 1: 'Feliz', 2: 'Neutral', 3: 'Triste', 4: 'Sorprendido'}

# Función para calcular distancias entre landmarks
def calcular_distancias(landmarks_per_image):
    distancias = []
    for name, points in landmarks_per_image.items():
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1 = np.array(points[i])
                p2 = np.array(points[j])
                dist = np.linalg.norm(p1 - p2)
                distancias.append(dist)
    return distancias

# Función para extraer distancias de landmarks y devolverlas como numpy array
def extraer_distancias(landmarks_per_image):
    distancias = calcular_distancias(landmarks_per_image)
    distancias_array = np.array(distancias)
    return distancias_array

# Función para extraer landmarks en tiempo real desde el frame de la cámara
def extraer_landmarks_de_frame(frame, landmarks_of_interest):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    
    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        landmarks_per_image = {}
        for name, points in landmarks_of_interest.items():
            landmarks_per_image[name] = []
            for point in points:
                landmark = face_landmarks.landmark[point]
                landmarks_per_image[name].append((landmark.x, landmark.y, landmark.z))
                
                # Dibujar los puntos faciales en el frame
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
        return landmarks_per_image, frame
    else:
        return None, frame

# Landmarks de interés
LANDMARKS_OF_INTEREST = {
    "left_eye": [33, 160, 158, 133, 153, 144],  
    "right_eye": [362, 385, 387, 263, 373, 380],  
    "eyebrows": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],  
    "nose": [1, 2, 4, 5, 9, 197, 195, 49, 279, 278, 344, 309],  
    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61],  
    "lips_upper": [185, 40, 39, 37, 0, 267, 269, 270, 409],  
    "lips_lower": [18, 84, 181, 321, 314],  
    "lips_inner_upper": [78, 95, 88, 178, 87],  
    "lips_inner_lower": [317, 402, 318, 324, 308],  
    "chin": [152],  
    "face_contour": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                     152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],  
}

# Inicializar el contador de sonrisas en st.session_state para la segunda parte
if 'contador_sonrisas' not in st.session_state:
    st.session_state.contador_sonrisas = 0

# Descripción del proyecto
st.title("¡Prueba EmotiScan! V1")

st.markdown("<h5>Este proyecto te permite escanear tus emociones en tiempo real. Prueba encendiendo la cámara  para interactuar con el modelo</h5>", unsafe_allow_html=True)
st.write("""-------------------------------------------------------------------------------------------------------------------- """)

st.write(""" **Espejito Espejito...** 🪞✨  
          Activando esta primera parte podrás visualizar cómo el modelo detecta tus emociones (Feliz, Triste, Sorprendido, Neutral, etc..)""")
st.write("""PSS... 🤫 Es la versión 1, para mejores resultados: entre más cerca de la cámara estés, mejor la predicción...""")

# PRIMERA PARTE: Prueba de la cámara
st.subheader("Prueba la cámara")

# Checkbox para activar la cámara y probar las emociones sin guardar la información
run_camera_test = st.checkbox('Activar cámara para probar tus emociones')

if run_camera_test:
    camera_test = cv2.VideoCapture(0)
    FRAME_WINDOW_TEST = st.image([])

    while run_camera_test:
        ret_test, frame_test = camera_test.read()
        if ret_test:
            # Invertir la imagen
            frame_test = cv2.flip(frame_test, 1)
            # Extraer landmarks
            landmarks_test, frame_test = extraer_landmarks_de_frame(frame_test, LANDMARKS_OF_INTEREST)
            
            if landmarks_test:
                # Mostrar las predicciones, pero no guardar nada
                distancias_test = extraer_distancias(landmarks_test)
                distancias_normalizadas_test = scaler.transform([distancias_test])
                prediccion_test = modelo.predict(distancias_normalizadas_test)
                emocion_predicha_test = label_dict[np.argmax(prediccion_test)]
                # Mostrar la predicción en la imagen
                cv2.putText(frame_test, f'Prediccion: {emocion_predicha_test}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Mostrar la imagen con la predicción
            FRAME_WINDOW_TEST.image(cv2.cvtColor(frame_test, cv2.COLOR_BGR2RGB))
        time.sleep(0.1)
    camera_test.release()

st.write(""" """)
st.write(""" """)
st.write(""" """)
st.write(""" """)
st.write(""" """)
st.write(""" """)
st.write("""-------------------------------------------------------------------------------------------------------------------- """)

# SEGUNDA PARTE: Prueba con el contador de emociones por segundo
st.subheader("Ahora prueba el contador de emociones con el siguiente video")
st.write("""
🎬 **¡Prepárate para la acción!** 🎬

El contador se activa en cuanto marques el check. Una vez habilitado por favor reproduce el video. ¡Espero que te guste! 😄

Vamos a ver cuántas veces te saca una sonrisa este pequeño clip. ¡A por ello! 🚀
""")
st.subheader(""" """)

# Mostrar el video en una columna diferente para que no desaparezca
col1, col2 = st.columns([1, 2])

# Mostrar el video pre cargado en la columna de la derecha
with col2:
    st.video('Adoptar.mp4')  # Asegúrate de que el archivo del video esté en la ubicación correcta

# Checkbox para activar la lógica del contador
with col1:
    run_webcam = st.checkbox('Iniciar el contador de emociones con la webcam')

FRAME_WINDOW_WEBCAM = col1.image([])

if run_webcam:
    camera = cv2.VideoCapture(0)
    contador_placeholder = st.empty()

    # Variables para el control del tiempo y predicciones
    start_time = time.time()
    predicciones_por_segundo = []

    while run_webcam:
        ret_webcam, frame_webcam = camera.read()
        if ret_webcam:
            frame_webcam = cv2.flip(frame_webcam, 1)
            landmarks_per_image, frame_webcam = extraer_landmarks_de_frame(frame_webcam, LANDMARKS_OF_INTEREST)
            
            if landmarks_per_image:
                distancias = extraer_distancias(landmarks_per_image)
                distancias_normalizadas = scaler.transform([distancias])
                prediccion = modelo.predict(distancias_normalizadas)
                emocion_predicha = label_dict[np.argmax(prediccion)]
                
                # Almacenar la predicción en la lista
                predicciones_por_segundo.append(emocion_predicha)
                
                # Mostrar la predicción en el frame
                cv2.putText(frame_webcam, f'Prediccion: {emocion_predicha}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            FRAME_WINDOW_WEBCAM.image(cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB))

            # Revisar si ha pasado un segundo
            current_time = time.time()
            if current_time - start_time >= 1:
                # Contar las emociones del segundo solo si hay predicciones
                if predicciones_por_segundo:
                    emocion_mas_frecuente = Counter(predicciones_por_segundo).most_common(1)[0][0]
                    
                    # Si la emoción más frecuente fue "Feliz", incrementar el contador
                    if emocion_mas_frecuente == 'Feliz':
                        st.session_state.contador_sonrisas += 1
                
                # Reiniciar el contador de tiempo y la lista de predicciones
                start_time = current_time
                predicciones_por_segundo = []

            # Mostrar el contador actualizado
            contador_placeholder.write(f'Número total de sonrisas detectadas: {st.session_state.contador_sonrisas}')

        time.sleep(0.1)
