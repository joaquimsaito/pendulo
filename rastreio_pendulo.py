import cv2
import numpy as np
import pandas as pd
import os

# ====================================================================
# 1. CONFIGURACOES GLOBAIS
# ====================================================================

VIDEO_PATH = 'simsim.mp4' 
OUTPUT_CSV = 'dados_pendulo.csv'

# Limites HSV (Hue, Saturation, Value) para rastrear o objeto escuro.
# Se o objeto for preto/escuro em fundo claro, o Valor (V) deve ser baixo (max 80)
LOWER_BOUND_HSV = np.array([0, 0, 0])      
UPPER_BOUND_HSV = np.array([180, 255, 80]) 
KERNEL = np.ones((5,5),np.uint8) # Kernel para limpeza de ruido (morfologia)

def extrair_posicoes_do_video(video_path, lower_hsv, upper_hsv, kernel):
    """ Rastreia o centro de massa da bolha em cada frame. """
    
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(video_path):
        print(f"ERRO: Arquivo {video_path} nao encontrado!")
        return []
    if not cap.isOpened():
        print("ERRO: Nao foi possivel abrir o video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback 
    
    data = []
    frame_num = 0
    
    # ====================================================================
    # 2. LOOP DE PROCESSAMENTO DE FRAMES
    # ====================================================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converte para HSV para rastreamento robusto por cor/brilho
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Limpeza morfologica: remove pequenos ruidos na mascara (melhora o rastreamento)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Encontra contornos (regioes de interesse)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Seleciona o maior contorno, assumindo que e a massa do pendulo
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                # Calcula as coordenadas do Centro de Massa (CM)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                time_s = frame_num / fps
                
                # Armazena os resultados
                data.append({'frame': frame_num, 'tempo': time_s, 'x': cX, 'y': cY})
                
        frame_num += 1

    cap.release()
    return data

# ====================================================================
# 3. EXECUCAO PRINCIPAL E SALVAMENTO
# ====================================================================
if __name__ == "__main__":
    print("Iniciando Rastreamento do Pendulo...")
    tracking_data = extrair_posicoes_do_video(VIDEO_PATH, LOWER_BOUND_HSV, UPPER_BOUND_HSV, KERNEL)
    
    if tracking_data:
        df_output = pd.DataFrame(tracking_data)
        df_output.to_csv(OUTPUT_CSV, index=False)
        print("=" * 50)
        print(f"SUCESSO: Dados salvos em '{OUTPUT_CSV}'")
        print(f"Total de pontos rastreados: {len(tracking_data)}")
        print("=" * 50)
    else:
        print("FALHA: Nenhuma posicao rastreada. Verifique o video e os limites HSV.")