import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="GeoLaplace",
    page_icon="🌍",
    layout="wide"
)

# Precisión suficiente para la inversión numérica sin sobrecargar memoria/tiempo
mp.mp.dps = 24

# ============================================================
# FUNCIONES DE CÁLCULO
# ============================================================

def inverse_laplace_value(F, t):
    """Inversión numérica de Laplace con control básico de errores."""
    try:
        value = mp.invertlaplace(F, t, method="talbot")
        value = float(mp.re(value))
        return max(value, 0.0)
    except Exception:
        return 0.0


def time_for_U(U_array, target, t_days):
    if not np.any(U_array >= target):
        return None
    idx = np.argmax(U_array >= target)
    return float(t_days[idx])


def Uv_terzaghi_approx(Tv):
    Uv = np.zeros_like(Tv, dtype=float)

    for i, T in enumerate(Tv):
        if T < 0.287:
            Uv[i] = np.sqrt(4 * T / np.pi)
        else:
            Uv[i] = 1 - (8 / np.pi**2) * np.exp(-np.pi**2 * T / 4)

    return np.clip(Uv, 0, 1)


def calculate_terzaghi(H, cv, delta_sigma, mv, drainage_1D, t_max_days, n_t, n_z):
    """
    Calcula consolidación 1D con Transformada de Laplace.

    Optimización clave:
    - Se calcula u promedio para todos los tiempos.
    - La distribución u(z,t) se calcula solo para 5 tiempos representativos,
      porque eso es lo que realmente se grafica. Así evitamos crear una matriz
      grande n_t x n_z y muchas inversiones de Laplace innecesarias.
    """

    t_days = np.linspace(1, t_max_days, n_t)
    t_seconds = t_days * 24 * 3600
    z_vals = np.linspace(0, H, n_z)

    S_final = mv * delta_sigma * H

    def u_laplace_double(z, s):
        lam = mp.sqrt(s / cv)
        return (delta_sigma / s) * (
            1 - mp.cosh(lam * (z - H / 2)) / mp.cosh(lam * H / 2)
        )

    def u_laplace_single(z, s):
        lam = mp.sqrt(s / cv)
        return (delta_sigma / s) * (
            1 - mp.cosh(lam * (H - z)) / mp.cosh(lam * H)
        )

    def avg_u_laplace_double(s):
        lam = mp.sqrt(s / cv)
        return (delta_sigma / s) * (
            1 - (2 * mp.tanh(lam * H / 2)) / (lam * H)
        )

    def avg_u_laplace_single(s):
        lam = mp.sqrt(s / cv)
        return (delta_sigma / s) * (
            1 - mp.tanh(lam * H) / (lam * H)
        )

    if drainage_1D == "Drenagem dupla":
        u_laplace = u_laplace_double
        avg_u_laplace = avg_u_laplace_double
    else:
        u_laplace = u_laplace_single
        avg_u_laplace = avg_u_laplace_single

    u_avg = np.zeros(n_t, dtype=float)
    U_1D = np.zeros(n_t, dtype=float)

    for i, t in enumerate(t_seconds):
        F_avg = lambda s: avg_u_laplace(s)
        u_avg[i] = inverse_laplace_value(F_avg, t)
        U_1D[i] = np.clip(1 - u_avg[i] / delta_sigma, 0, 1)

    settlement = U_1D * S_final

    # Solo se calculan los perfiles que se van a mostrar en la gráfica.
    profile_indices = np.unique(np.array([
        0,
        int(n_t * 0.15),
        int(n_t * 0.30),
        int(n_t * 0.60),
        n_t - 1
    ], dtype=int))

    u_profiles = np.zeros((len(profile_indices), n_z), dtype=float)

    for p, idx in enumerate(profile_indices):
        t = t_seconds[idx]

        for j, z in enumerate(z_vals):
            if drainage_1D == "Drenagem dupla":
                if np.isclose(z, 0.0) or np.isclose(z, H):
                    u_profiles[p, j] = 0.0
                else:
                    F = lambda s, z=z: u_laplace(z, s)
                    u_profiles[p, j] = inverse_laplace_value(F, t)
            else:
                if np.isclose(z, 0.0):
                    u_profiles[p, j] = 0.0
                else:
                    F = lambda s, z=z: u_laplace(z, s)
                    u_profiles[p, j] = inverse_laplace_value(F, t)

    return {
        "t_days": t_days,
        "z_vals": z_vals,
        "S_final": S_final,
        "u_avg": u_avg,
        "U": U_1D,
        "settlement": settlement,
        "u_profiles": u_profiles,
        "profile_indices": profile_indices
    }


def calculate_geodrains(H, cv, ch, delta_sigma, mv, drainage_1D, spacing, pattern, a, b, t_max_days, n_t):
    t_days = np.linspace(1, t_max_days, n_t)
    t_seconds = t_days * 24 * 3600

    S_final = mv * delta_sigma * H

    if drainage_1D == "Drenagem dupla":
        hd = H / 2
    else:
        hd = H

    dw = 2 * (a + b) / np.pi

    if pattern == "Malha triangular":
        de = 1.05 * spacing
    else:
        de = 1.13 * spacing

    n = de / dw
    F_n = np.log(n) - 0.75

    # Evita divisiones problemáticas si el usuario pone una geometría no física.
    if F_n <= 0:
        F_n = np.nan

    Th = ch * t_seconds / de**2
    Uh = 1 - np.exp(-8 * Th / F_n)
    Uh = np.clip(Uh, 0, 1)

    Tv = cv * t_seconds / hd**2
    Uv = Uv_terzaghi_approx(Tv)

    U_geo = 1 - (1 - Uv) * (1 - Uh)
    U_geo = np.clip(U_geo, 0, 1)

    settlement_geo = U_geo * S_final
    u_geo_avg = delta_sigma * (1 - U_geo)

    return {
        "t_days": t_days,
        "S_final": S_final,
        "U": U_geo,
        "settlement": settlement_geo,
        "u_avg": u_geo_avg,
        "Uh": Uh,
        "Uv": Uv,
        "dw": dw,
        "de": de,
        "n": n,
        "F_n": F_n
    }

# ============================================================
# FUNCIONES DE GRÁFICAS
# ============================================================

def plot_consolidation(t_days, U_1D, U_geo=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_days, U_1D * 100, linewidth=3, label="Terzaghi 1D - Laplace")

    if U_geo is not None:
        ax.plot(t_days, U_geo * 100, linewidth=3, label="Geodrenos - Barron + Carrillo")

    ax.set_xlabel("Tempo [dias]")
    ax.set_ylabel("Grau de consolidação [%]")
    ax.set_title("Grau de consolidação")
    ax.grid(True)
    ax.legend()
    return fig


def plot_settlement(t_days, settlement_1D, S_final, settlement_geo=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_days, settlement_1D * 100, linewidth=3, label="Terzaghi 1D - Laplace")

    if settlement_geo is not None:
        ax.plot(t_days, settlement_geo * 100, linewidth=3, label="Geodrenos - Barron + Carrillo")

    ax.axhline(S_final * 100, linestyle="--", label="Recalque final")
    ax.set_xlabel("Tempo [dias]")
    ax.set_ylabel("Recalque [cm]")
    ax.set_title("Recalque ao longo do tempo")
    ax.grid(True)
    ax.legend()
    return fig


def plot_pore_pressure(t_days, u_1D_avg, u_geo_avg=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_days, u_1D_avg, linewidth=3, label="Terzaghi 1D")

    if u_geo_avg is not None:
        ax.plot(t_days, u_geo_avg, linewidth=3, label="Geodrenos")

    ax.set_xlabel("Tempo [dias]")
    ax.set_ylabel("Excesso médio de pressão neutra [kPa]")
    ax.set_title("Dissipação média da pressão neutra")
    ax.grid(True)
    ax.legend()
    return fig


def plot_pressure_profile(t_days, z_vals, u_profiles, profile_indices):
    fig, ax = plt.subplots(figsize=(8, 5))

    for p, idx in enumerate(profile_indices):
        ax.plot(
            u_profiles[p, :],
            z_vals,
            linewidth=2,
            label=f"t = {t_days[idx]:.0f} dias"
        )

    ax.invert_yaxis()
    ax.set_xlabel("Excesso de pressão neutra u [kPa]")
    ax.set_ylabel("Profundidade z [m]")
    ax.set_title("Distribuição de pressão neutra - Terzaghi 1D")
    ax.grid(True)
    ax.legend()
    return fig


def plot_characteristic_times(t50_1D, t90_1D, t95_1D, t50_geo=None, t90_geo=None, t95_geo=None):
    labels = ["t50", "t90", "t95"]
    terzaghi_times = [
        t50_1D if t50_1D is not None else np.nan,
        t90_1D if t90_1D is not None else np.nan,
        t95_1D if t95_1D is not None else np.nan
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.35

    if t50_geo is None:
        ax.bar(x, terzaghi_times, width, label="Terzaghi 1D")
    else:
        geo_times = [
            t50_geo if t50_geo is not None else np.nan,
            t90_geo if t90_geo is not None else np.nan,
            t95_geo if t95_geo is not None else np.nan
        ]
        ax.bar(x - width / 2, terzaghi_times, width, label="Terzaghi 1D")
        ax.bar(x + width / 2, geo_times, width, label="Geodrenos")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tempo [dias]")
    ax.set_title("Tempos característicos de consolidação")
    ax.grid(True, axis="y")
    ax.legend()
    return fig


def show_pyplot(fig):
    """Muestra y libera memoria de figuras Matplotlib."""
    st.pyplot(fig)
    plt.close(fig)
    gc.collect()

# ============================================================
# INTERFACE
# ============================================================

st.title("🌍 GeoLaplace")
st.caption("GeoLaplace v1.0 — Ferramenta de análise de consolidação")
st.subheader("Ferramenta interativa para análise de consolidação em solos moles")

st.markdown("""
**GeoLaplace** é uma ferramenta computacional para avaliar a consolidação
de camadas de argila saturada submetidas a uma carga distribuída.

A aplicação permite analisar dois cenários:

1. **Consolidação unidimensional de Terzaghi**, resolvida no domínio de Laplace.
2. **Consolidação com geodrenos**, considerando drenagem radial por Barron e combinação vertical-radial por Carrillo.
""")

image_path = "Casos de Analisis.png"
if os.path.exists(image_path):
    st.image(
        image_path,
        caption="Esquema conceitual dos casos de análise.",
        width="stretch"
    )
else:
    st.warning(f"Imagem não encontrada: {image_path}")

st.markdown("---")

st.header("Parâmetros utilizados")
st.markdown("""
| Símbolo | Significado | Unidade |
|---|---|---|
| H | Espessura da camada de argila | m |
| cv | Coeficiente de consolidação vertical | m²/s |
| ch | Coeficiente de consolidação horizontal | m²/s |
| Δσ | Sobrecarga ou incremento de tensão v ertical | kPa |
| mv | Coeficiente de compressibilidade volumétrica | 1/kPa |
| s | Espaçamento entre geodrenos | m |
| a | Largura do geodreno | m |
| b | Espessura do geodreno | m |
| t | Tempo de análise | dias |
""")

st.markdown("---")

st.header("Entrada de dados")

# Importante: fora do form para atualizar imediatamente os parâmetros dos geodrenos
analysis_mode = st.radio(
    "Tipo de análise",
    [
        "Somente consolidação 1D",
        "Comparação: consolidação 1D vs geodrenos"
    ],
    key="analysis_mode"
)

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        H = st.number_input("Espessura H [m]", value=6.0, min_value=0.1)
        cv = st.number_input("cv [m²/s]", value=1.2e-7, format="%.2e")
        drainage_1D = st.selectbox(
            "Condição de drenagem",
            ["Drenagem dupla", "Drenagem simples"]
        )

    with col2:
        delta_sigma = st.number_input("Sobrecarga Δσ [kPa]", value=80.0)
        mv = st.number_input("mv [1/kPa]", value=4.0e-4, format="%.2e")
        t_max_days = st.number_input("Tempo máximo [dias]", value=1000.0, min_value=1.0)

    with col3:
        # Limites reducidos para no superar memoria/CPU en Streamlit Community Cloud.
        n_t = st.slider("Pontos no tempo", 20, 80, 50)
        n_z = st.slider("Pontos em profundidade", 8, 25, 15)

    if analysis_mode == "Comparação: consolidação 1D vs geodrenos":
        st.subheader("Geometria dos geodrenos")
        if os.path.exists("malha.png"):
            st.image(
                "malha.png",
                caption="Padrões de instalação e parâmetros geométricos dos geodrenos.",
                width="stretch"
            )
        else:
            st.info("Figura 'malha.png' não encontrada. A análise pode ser executada normalmente.")

        st.subheader("Parâmetros dos geodrenos")
        col4, col5, col6 = st.columns(3)

        with col4:
            ch = st.number_input("ch [m²/s]", value=2.0e-7, format="%.2e")
            spacing = st.number_input("Espaçamento entre geodrenos s [m]", value=1.5, min_value=0.1)

        with col5:
            pattern = st.selectbox("Tipo de malha", ["Malha triangular", "Malha quadrada"])
            a = st.number_input("Largura do geodreno a [m]", value=0.10, min_value=0.001)

        with col6:
            b = st.number_input("Espessura do geodreno b [m]", value=0.005, min_value=0.0001)

    submitted = st.form_submit_button("Executar análise")

# ============================================================
# RESULTADOS
# ============================================================

if submitted:
    with st.spinner("Calculando consolidação 1D por Transformada Inversa de Laplace..."):
        terzaghi = calculate_terzaghi(
            H, cv, delta_sigma, mv, drainage_1D, t_max_days, n_t, n_z
        )

    geodrains = None
    if analysis_mode == "Comparação: consolidação 1D vs geodrenos":
        geodrains = calculate_geodrains(
            H, cv, ch, delta_sigma, mv, drainage_1D,
            spacing, pattern, a, b, t_max_days, n_t
        )

    t_days = terzaghi["t_days"]
    S_final = terzaghi["S_final"]

    t50_1D = time_for_U(terzaghi["U"], 0.50, t_days)
    t90_1D = time_for_U(terzaghi["U"], 0.90, t_days)
    t95_1D = time_for_U(terzaghi["U"], 0.95, t_days)

    if geodrains is not None:
        t50_geo = time_for_U(geodrains["U"], 0.50, t_days)
        t90_geo = time_for_U(geodrains["U"], 0.90, t_days)
        t95_geo = time_for_U(geodrains["U"], 0.95, t_days)
    else:
        t50_geo = t90_geo = t95_geo = None

    st.markdown("---")
    st.header("Resultados")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recalque final", f"{S_final * 100:.2f} cm")
    col2.metric("U final 1D", f"{terzaghi['U'][-1] * 100:.2f} %")
    col3.metric("u média final 1D", f"{terzaghi['u_avg'][-1]:.2f} kPa")

    if geodrains is not None:
        col4.metric("U final com geodrenos", f"{geodrains['U'][-1] * 100:.2f} %")
    else:
        col4.metric("Modo", "1D")

    tabs = st.tabs([
        "Resumo",
        "Grau de consolidação",
        "Recalque",
        "Pressão neutra",
        "Distribuição u(z,t)",
        "Interpretação técnica",
        "Sustento teórico"
    ])

    with tabs[0]:
        st.subheader("Resumo numérico")

        if geodrains is not None:
            data = {
                "Indicador": ["t50 [dias]", "t90 [dias]", "t95 [dias]", "Recalque final [cm]"],
                "Terzaghi 1D": [t50_1D, t90_1D, t95_1D, S_final * 100],
                "Geodrenos": [t50_geo, t90_geo, t95_geo, S_final * 100]
            }
        else:
            data = {
                "Indicador": ["t50 [dias]", "t90 [dias]", "t95 [dias]", "Recalque final [cm]"],
                "Terzaghi 1D": [t50_1D, t90_1D, t95_1D, S_final * 100]
            }

        st.dataframe(data, width="stretch")

        if geodrains is not None:
            st.subheader("Parâmetros geométricos dos geodrenos")
            st.write(f"Diâmetro equivalente do geodreno, dw = {geodrains['dw']:.4f} m")
            st.write(f"Diâmetro de influência, de = {geodrains['de']:.4f} m")
            st.write(f"Relação n = de/dw = {geodrains['n']:.2f}")
            st.write(f"F(n) = {geodrains['F_n']:.3f}")

    with tabs[1]:
        fig = plot_consolidation(
            t_days,
            terzaghi["U"],
            geodrains["U"] if geodrains is not None else None
        )
        show_pyplot(fig)

    with tabs[2]:
        fig = plot_settlement(
            t_days,
            terzaghi["settlement"],
            S_final,
            geodrains["settlement"] if geodrains is not None else None
        )
        show_pyplot(fig)

    with tabs[3]:
        fig = plot_pore_pressure(
            t_days,
            terzaghi["u_avg"],
            geodrains["u_avg"] if geodrains is not None else None
        )
        show_pyplot(fig)

    with tabs[4]:
        fig = plot_pressure_profile(
            t_days,
            terzaghi["z_vals"],
            terzaghi["u_profiles"],
            terzaghi["profile_indices"]
        )
        show_pyplot(fig)

        st.info(
            "Esta distribuição em profundidade é calculada para o caso 1D de Terzaghi "
            "a partir da solução no domínio de Laplace."
        )

    with tabs[5]:
        st.subheader("Interpretação técnica")

        if geodrains is not None:
            st.markdown(f"""
            Para os parâmetros inseridos, o recalque final primário estimado é de
            **{S_final * 100:.2f} cm** nos dois casos.

            Isso ocorre porque o recalque final depende principalmente de **mv**, **Δσ** e **H**,
            e não diretamente da presença dos geodrenos.

            A diferença fundamental está no tempo necessário para atingir esse recalque.
            No modelo 1D, a dissipação do excesso de pressão neutra ocorre por drenagem vertical.
            Com geodrenos, a água passa a escoar radialmente em direção aos drenos,
            encurtando o caminho de drenagem e acelerando a consolidação.

            Portanto, os geodrenos **não aumentam o recalque final primário**.
            Eles reduzem significativamente o tempo necessário para atingi-lo.
            """)
        else:
            st.markdown(f"""
            Para os parâmetros inseridos, o recalque final primário estimado é de
            **{S_final * 100:.2f} cm**.

            A solução considera a dissipação do excesso de pressão neutra ao longo da profundidade
            da camada de argila saturada. A Transformada de Laplace é aplicada em relação ao tempo,
            convertendo a equação diferencial parcial original em uma equação diferencial ordinária
            no domínio da profundidade.
            """)

    with tabs[6]:
        st.subheader("Sustento teórico")

        st.markdown("### 1. Consolidação unidimensional de Terzaghi")
        st.write("A equação governante é:")
        st.latex(r"""
        \frac{\partial u(z,t)}{\partial t}
        =
        c_v
        \frac{\partial^2 u(z,t)}{\partial z^2}
        """)

        st.write("Condição inicial:")
        st.latex(r"""u(z,0)=\Delta\sigma""")

        st.write("Para drenagem dupla:")
        st.latex(r"""u(0,t)=0""")
        st.latex(r"""u(H,t)=0""")

        st.write("Para drenagem simples:")
        st.latex(r"""u(0,t)=0""")
        st.latex(r"""\frac{\partial u(H,t)}{\partial z}=0""")

        st.write("Aplicando a Transformada de Laplace no tempo:")
        st.latex(r"""
        sU(z,s)-u(z,0)
        =
        c_v
        \frac{d^2U(z,s)}{dz^2}
        """)

        st.markdown("---")
        st.markdown("### 2. Consolidação radial com geodrenos")
        st.write("A equação idealizada para drenagem radial é:")
        st.latex(r"""
        \frac{\partial u(r,t)}{\partial t}
        =
        c_h
        \left(
        \frac{\partial^2 u}{\partial r^2}
        +
        \frac{1}{r}
        \frac{\partial u}{\partial r}
        \right)
        """)

        st.write("O fator de tempo horizontal é:")
        st.latex(r"""T_h=\frac{c_h t}{d_e^2}""")

        st.write("A solução média de Barron é:")
        st.latex(r"""U_h = 1 - \exp\left(-\frac{8T_h}{F(n)}\right)""")
        st.latex(r"""n=\frac{d_e}{d_w}""")
        st.latex(r"""F(n)=\ln(n)-0.75""")

        st.markdown("---")
        st.markdown("### 3. Consolidação combinada de Carrillo")
        st.latex(r"""U = 1-(1-U_v)(1-U_h)""")

        st.markdown("---")
        st.markdown("### 4. Recalque primário")
        st.latex(r"""S_f=m_v\Delta\sigma H""")
        st.latex(r"""S(t)=U(t)S_f""")

    # Limpieza explícita al terminar el cálculo.
    gc.collect()

else:
    st.info("Insira os dados e clique em **Executar análise**.")
