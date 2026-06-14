import os
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

mp.mp.dps = 30

# ============================================================
# FUNCIONES DE CÁLCULO
# ============================================================

def inverse_laplace_value(F, t):
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
    return t_days[idx]


def Uv_terzaghi_approx(Tv):
    Uv = np.zeros_like(Tv)

    for i, T in enumerate(Tv):
        if T < 0.287:
            Uv[i] = np.sqrt(4 * T / np.pi)
        else:
            Uv[i] = 1 - (8 / np.pi**2) * np.exp(-np.pi**2 * T / 4)

    return np.clip(Uv, 0, 1)


@st.cache_data(show_spinner=False)
def calculate_terzaghi(H, cv, delta_sigma, mv, drainage_1D, t_max_days, n_t, n_z):

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

    u_matrix = np.zeros((n_t, n_z))
    u_avg = np.zeros(n_t)
    U_1D = np.zeros(n_t)

    for i, t in enumerate(t_seconds):

        F_avg = lambda s: avg_u_laplace(s)
        u_avg[i] = inverse_laplace_value(F_avg, t)

        U_1D[i] = 1 - u_avg[i] / delta_sigma
        U_1D[i] = np.clip(U_1D[i], 0, 1)

        for j, z in enumerate(z_vals):

            if drainage_1D == "Drenagem dupla":
                if np.isclose(z, 0.0) or np.isclose(z, H):
                    u_matrix[i, j] = 0.0
                else:
                    F = lambda s, z=z: u_laplace(z, s)
                    u_matrix[i, j] = inverse_laplace_value(F, t)

            else:
                if np.isclose(z, 0.0):
                    u_matrix[i, j] = 0.0
                else:
                    F = lambda s, z=z: u_laplace(z, s)
                    u_matrix[i, j] = inverse_laplace_value(F, t)

    settlement = U_1D * S_final

    return {
        "t_days": t_days,
        "z_vals": z_vals,
        "S_final": S_final,
        "u_avg": u_avg,
        "U": U_1D,
        "settlement": settlement,
        "u_matrix": u_matrix
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


def plot_pressure_profile(t_days, z_vals, u_matrix):
    fig, ax = plt.subplots(figsize=(8, 5))

    n_t = len(t_days)
    times_to_plot = [
        0,
        int(n_t * 0.15),
        int(n_t * 0.30),
        int(n_t * 0.60),
        n_t - 1
    ]

    for idx in times_to_plot:
        ax.plot(
            u_matrix[idx, :],
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


# ============================================================
# INTERFACE
# ============================================================

st.title("🌍 GeoLaplace")
st.subheader("Ferramenta interativa para análise de consolidação em solos moles")

st.markdown("""
**GeoLaplace** é uma ferramenta computacional educacional para avaliar a consolidação
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
        use_container_width=True
    )
else:
    st.error(f"Imagem não encontrada: {image_path}")
    st.write("Arquivos disponíveis na pasta:")
    st.write(os.listdir("."))

st.markdown("---")

st.header("Parâmetros utilizados")

st.markdown("""
| Símbolo | Significado | Unidade |
|---|---|---|
| H | Espessura da camada de argila | m |
| cv | Coeficiente de consolidação vertical | m²/s |
| ch | Coeficiente de consolidação horizontal | m²/s |
| Δσ | Sobrecarga ou incremento de tensão vertical | kPa |
| mv | Coeficiente de compressibilidade volumétrica | 1/kPa |
| s | Espaçamento entre geodrenos | m |
| a | Largura do geodreno | m |
| b | Espessura do geodreno | m |
| t | Tempo de análise | dias |
""")

st.markdown("---")

st.header("Entrada de dados")

with st.form("input_form"):

    analysis_mode = st.radio(
        "Tipo de análise",
        [
            "Somente consolidação 1D",
            "Comparação: consolidação 1D vs geodrenos"
        ]
    )

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
        n_t = st.slider("Pontos no tempo", 30, 150, 80)
        n_z = st.slider("Pontos em profundidade", 10, 40, 20)

    if analysis_mode == "Comparação: consolidação 1D vs geodrenos":

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

        st.dataframe(data, use_container_width=True)

        if geodrains is not None:
            st.subheader("Parâmetros geométricos dos geodrenos")
            st.write(f"Diâmetro equivalente do geodreno, dw = {geodrains['dw']:.4f} m")
            st.write(f"Diâmetro de influência, de = {geodrains['de']:.4f} m")
            st.write(f"Relação n = de/dw = {geodrains['n']:.2f}")
            st.write(f"F(n) = {geodrains['F_n']:.3f}")

    with tabs[1]:

        if geodrains is not None:
            fig = plot_consolidation(t_days, terzaghi["U"], geodrains["U"])
        else:
            fig = plot_consolidation(t_days, terzaghi["U"])

        st.pyplot(fig)

    with tabs[2]:

        if geodrains is not None:
            fig = plot_settlement(
                t_days,
                terzaghi["settlement"],
                S_final,
                geodrains["settlement"]
            )
        else:
            fig = plot_settlement(t_days, terzaghi["settlement"], S_final)

        st.pyplot(fig)

    with tabs[3]:

        if geodrains is not None:
            fig = plot_pore_pressure(t_days, terzaghi["u_avg"], geodrains["u_avg"])
        else:
            fig = plot_pore_pressure(t_days, terzaghi["u_avg"])

        st.pyplot(fig)

    with tabs[4]:

        fig = plot_pressure_profile(
            t_days,
            terzaghi["z_vals"],
            terzaghi["u_matrix"]
        )

        st.pyplot(fig)

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

        st.latex(r"""
        u(z,0)=\Delta\sigma
        """)

        st.write("Para drenagem dupla:")

        st.latex(r"""
        u(0,t)=0
        """)

        st.latex(r"""
        u(H,t)=0
        """)

        st.write("Para drenagem simples:")

        st.latex(r"""
        u(0,t)=0
        """)

        st.latex(r"""
        \frac{\partial u(H,t)}{\partial z}=0
        """)

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

        st.latex(r"""
        T_h=\frac{c_h t}{d_e^2}
        """)

        st.write("A solução média de Barron é:")

        st.latex(r"""
        U_h = 1 - \exp\left(-\frac{8T_h}{F(n)}\right)
        """)

        st.latex(r"""
        n=\frac{d_e}{d_w}
        """)

        st.latex(r"""
        F(n)=\ln(n)-0.75
        """)

        st.markdown("---")

        st.markdown("### 3. Consolidação combinada de Carrillo")

        st.latex(r"""
        U = 1-(1-U_v)(1-U_h)
        """)

        st.markdown("---")

        st.markdown("### 4. Recalque primário")

        st.latex(r"""
        S_f=m_v\Delta\sigma H
        """)

        st.latex(r"""
        S(t)=U(t)S_f
        """)

else:
    st.info("Insira os dados e clique em **Executar análise**.")