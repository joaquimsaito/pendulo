import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os

# ====================================================================
# 1. CONSTANTES E FUNCOES MATEMATICAS
# ====================================================================

DATA_CSV = 'dados_pendulo.csv'
# Valores teoricos (L=0.59m do seu experimento)
L_METROS = 0.59
G_ACELERACAO = 9.81 

def oha(t, A, b, omega, phi):
    """ Funcao do Oscilador Harmonico Amortecido (OHA) """
    return A * np.exp(-b * t) * np.cos(omega * t + phi)

def realizar_analise_e_ajuste():
    """ Executa a analise do Item 6 e o ajuste do Item 7 """
    
    # COMENTARIO CHAVE: 2. CARREGAMENTO E PRE-PROCESSAMENTO DE DADOS
    if not os.path.exists(DATA_CSV):
        print(f"ERRO: Arquivo '{DATA_CSV}' nao encontrado. Execute rastreio_pendulo.py primeiro.")
        return

    df = pd.read_csv(DATA_CSV)
    t = df['tempo'].values
    # Centraliza X, que e a variavel de interesse para a oscilacao
    x = df['x'].values - df['x'].mean() 
    y = df['y'].values - df['y'].mean()

    # --- ITEM 6: ANALISE E VALIDACAO DO MOVIMENTO ---
    print("=" * 50)
    print("ITEM 6: ANALISE ESTATISTICA E VALIDACAO")
    print("=" * 50)

    var_x = np.std(x)
    var_y = np.std(y)
    amplitude_x = np.ptp(x) 
    
    # Imprime a relacao de variacao (sem acentos)
    print(f"Desvio padrao em X: {var_x:.2f} pixels")
    print(f"Desvio padrao em Y: {var_y:.2f} pixels")
    print(f"Amplitude X (pico-a-pico): {amplitude_x:.2f} pixels")
    print(f"Razao Y/X (variacao): {var_y/var_x:.4f}")
    
    if var_y/var_x > 0.1 and var_x < 20: 
        print("ATENCAO: Alta razao Y/X. O movimento rastreado em X pode ter sido muito pequeno.")
        print("         Continuando com a analise, assumindo pequenos angulos fisicamente.")
    else:
        print("Validacao: Variacao em Y e pequena, aproximacao de pequenos angulos e valida.")

    # COMENTARIO CHAVE: 3. ESTIMATIVAS INICIAIS (PARA ROBUSTEZ)
    try:
        fft_vals = np.fft.fft(x)
        fft_freq = np.fft.fftfreq(len(x), d=(t[1]-t[0]))
        idx = np.argmax(np.abs(fft_vals[1:len(fft_vals)//2])) + 1
        omega_guess = 2 * np.pi * np.abs(fft_freq[idx])
    except:
        omega_guess = 4.0 

    A_guess = np.max(np.abs(x))
    b_guess = 0.01 
    phi_guess = 0

    p0 = [A_guess, b_guess, omega_guess, phi_guess]
    
    # --- ITEM 7: AJUSTE DO OHA ---
    print("\n" + "=" * 50)
    print("ITEM 7: AJUSTE DO OSCILADOR HARMONICO AMORT.")
    print("=" * 50)

    try:
        bounds = ([0, 0, 0, -np.pi], [np.inf, np.inf, np.inf, np.pi])
        
        # COMENTARIO CHAVE: 4. AJUSTE DA CURVA
        popt, pcov = curve_fit(oha, t, x, p0=p0, bounds=bounds, maxfev=10000)
        A_fit, b_fit, omega_fit, phi_fit = popt
        
        perr = np.sqrt(np.diag(pcov)) 
        
        # COMENTARIO CHAVE: 5. RESULTADOS DO AJUSTE
        print("RESULTADOS DO AJUSTE:")
        print(f"A (amplitude inicial): {A_fit:.2f} +- {perr[0]:.2f} pixels")
        print(f"b (coef. amortecimento): {b_fit:.6f} +- {perr[1]:.6f} s^-1")
        print(f"w (freq. angular): {omega_fit:.4f} +- {perr[2]:.4f} rad/s")
        print(f"phi (fase inicial): {phi_fit:.4f} +- {perr[3]:.4f} rad")
        
        # Calculo de Q
        Q = omega_fit / (2 * b_fit)
        dQ = Q * np.sqrt((perr[2]/omega_fit)**2 + (perr[1]/b_fit)**2)
        print(f"\nFator de Qualidade Q: {Q:.2f} +- {dQ:.2f}")

        # R^2
        residuals = x - oha(t, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((x - np.mean(x))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R^2 (qualidade do ajuste): {r_squared:.6f}")
        
        # COMENTARIO CHAVE: 6. COMPARACAO COM TEORIA
        omega_teorico = np.sqrt(G_ACELERACAO / L_METROS)
        domega_teorico = omega_teorico * 0.5 * (0.001 / L_METROS)
        
        print("\n" + "=" * 50)
        print("COMPARACAO COM TEORIA (w_0 = sqrt(g/L)):")
        print("=" * 50)
        print(f"w teorico: {omega_teorico:.4f} +- {domega_teorico:.4f} rad/s")
        print(f"w experimental: {omega_fit:.4f} +- {perr[2]:.4f} rad/s")
        
        diff = abs(omega_fit - omega_teorico)
        sigma_total = np.sqrt(perr[2]**2 + domega_teorico**2)
        n_sigma = diff / sigma_total
        
        print(f"Compatibilidade (em sigmas): {n_sigma:.2f}")
        if n_sigma < 2:
            print("Resultado: Valores teorico e experimental sao compativeis (< 2 sigmas)")
        else:
            print("Resultado: Diferenca significativa entre teoria e experimento.")

        # ====================================================================
        # 7. GERACAO DE GRAFICOS (AJUSTE + RESIDUOS E PLOTAGEM SOLICITADA)
        # ====================================================================
        plt.figure(figsize=(14, 8))
        
        # COMENTARIO CHAVE: SUBPLOT 1: AJUSTE COMPLETO E ENVELOPE (ITEM 7 e SOLICITADO)
        plt.subplot(2, 1, 1)
        plt.plot(t, x, 'b.', markersize=2, alpha=0.5, label='Dados Experimentais')
        plt.plot(t, oha(t, *popt), 'r-', linewidth=2, label='Ajuste OHA') # Curva amortecida
        
        # Envelope exponencial (linhas tracejadas, conforme a imagem)
        envelope_pos = A_fit * np.exp(-b_fit * t)
        plt.plot(t, envelope_pos, 'r--', linewidth=1.5, label='Envelope')
        plt.plot(t, -envelope_pos, 'r--', linewidth=1.5) # Envelope negativo
        
        plt.xlabel('Tempo (s)', fontsize=12)
        plt.ylabel('Posicao X (pixels)', fontsize=12)
        plt.title('Ajuste do Oscilador Harmonico Amortecido', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # COMENTARIO CHAVE: SUBPLOT 2: RESIDUOS
        plt.subplot(2, 1, 2)
        plt.plot(t, residuals, 'b.', markersize=2, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
        plt.xlabel('Tempo (s)', fontsize=12)
        plt.ylabel('Residuos (pixels)', fontsize=12)
        plt.title('Residuos do Ajuste', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ajuste_oha_final.png', dpi=300)
        # plt.show()

    except Exception as e:
        print(f"ERRO NO AJUSTE: {e}")
        print("Verifique os parametros iniciais ou a qualidade dos dados.")

# ====================================================================
# 8. EXECUCAO PRINCIPAL
# ====================================================================
if __name__ == "__main__":
    realizar_analise_e_ajuste()