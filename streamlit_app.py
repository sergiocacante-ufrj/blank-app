import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

st.set_page_config(
    page_title="GeoLaplace",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GeoLaplace")
st.subheader("Consolidação Unidimensional de Terzaghi usando Transformada de Laplace")

st.markdown("""
Este aplicativo calcula a dissipação do excesso de pressão neutra, o grau de consolidação
e o recalque ao longo do tempo para uma camada de argila saturada submetida a um carregamento instantâneo.
""")

st.markdown("---")

with st.sidebar:
    st.header("Parâmetros de entrada")

    H = st.number_input("Espessura da camada H [m]", value=6.0, min_value=0.1)
    cv = st.number_input("Coeficiente de consolidação cv [m²/s]", value=1.2e-7, format="%.2e")
    delta_sigma = st.number_input("Incremento de tensão Δσ [kPa]", value=80.0)
    mv = st.number_input("Coeficiente de compressibilidade mv [1/kPa]", value=4.0e-4, format="%.2e")

    drainage = st.selectbox(
        "Tipo de drenagem",
        ["Drenagem dupla", "Drenagem simples"]
    )

    t_max_days = st.number_input("Tempo máximo de análise [dias]", value=2000.0, min_value=1.0)
    n_t = st.slider("Número de pontos no tempo", 10, 80, 40)
    n_z = st.slider("Número de pontos em profundidade", 10, 60, 30)

    calcular = st.button("Calcular")

mp.mp.dps = 30

if calcular:

    t_days = np.linspace(1, t_max_days, n_t)
    t_seconds = t_days * 24 * 3600
    z_vals = np.linspace(0, H, n_z)

    if drainage == "Drenagem dupla":
        drainage_type = "double"
    else:
        drainage_type = "single"

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

    if drainage_type == "double":
        u_laplace = u_laplace_double
        avg_u_laplace = avg_u_laplace_double
    else:
        u_laplace = u_laplace_single
        avg_u_laplace = avg_u_laplace_single

    def inverse_laplace_value(F, t):
        try:
            value = mp.invertlaplace(F, t, method="talbot")
            value = float(mp.re(value))
            return max(value, 0.0)
        except Exception:
            return 0.0

    u_matrix = np.zeros((n_t, n_z))

    with st.spinner("Calculando solução por Transformada Inversa de Laplace..."):
        for i, t in enumerate(t_seconds):
            for j, z in enumerate(z_vals):

                if drainage_type == "double":
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

        avg_u = np.zeros(n_t)
        U = np.zeros(n_t)

        for i, t in enumerate(t_seconds):
            F_avg = lambda s: avg_u_laplace(s)
            avg_u[i] = inverse_laplace_value(F_avg, t)
            U[i] = 1 - avg_u[i] / delta_sigma
            U[i] = min(max(U[i], 0.0), 1.0)

    S_final = mv * delta_sigma * H
    settlement = U * S_final

    st.success("Cálculo concluído com sucesso.")

    st.markdown("---")
    st.header("Resultados principais")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Recalque final", f"{S_final * 100:.2f} cm")
    col2.metric("Grau de consolidação final", f"{U[-1] * 100:.2f} %")
    col3.metric("Recalque no tempo final", f"{settlement[-1] * 100:.2f} cm")
    col4.metric("Pressão neutra média final", f"{avg_u[-1]:.2f} kPa")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Modelo físico",
        "Grau de consolidação",
        "Recalque",
        "Pressão neutra"
    ])

    with tab1:
        st.header("Modelo físico e condições do problema")

        st.markdown(r"""
        Considera-se uma camada de argila saturada de espessura \(H\), submetida a um incremento
        instantâneo de tensão vertical \(\Delta \sigma\).

        A equação governante é:

        \[
        \frac{\partial u(z,t)}{\partial t}
        =
        c_v \frac{\partial^2 u(z,t)}{\partial z^2}
        \]

        onde:

        - \(u(z,t)\) é o excesso de pressão neutra;
        - \(c_v\) é o coeficiente de consolidação;
        - \(z\) é a profundidade;
        - \(t\) é o tempo.

        A condição inicial é:

        \[
        u(z,0) = \Delta \sigma
        \]

        Para drenagem dupla:

        \[
        u(0,t)=0
        \]

        \[
        u(H,t)=0
        \]

        Para drenagem simples:

        \[
        u(0,t)=0
        \]

        \[
        \frac{\partial u(H,t)}{\partial z}=0
        \]
        """)

    with tab2:
        st.header("Grau de consolidação")

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(t_days, U * 100, linewidth=2)
        ax1.set_xlabel("Tempo [dias]")
        ax1.set_ylabel("Grau de consolidação U [%]")
        ax1.set_title("Evolução do grau de consolidação")
        ax1.grid(True)

        st.pyplot(fig1)

    with tab3:
        st.header("Recalque ao longo do tempo")

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(t_days, settlement * 100, linewidth=2)
        ax2.set_xlabel("Tempo [dias]")
        ax2.set_ylabel("Recalque [cm]")
        ax2.set_title("Recalque por consolidação")
        ax2.grid(True)

        st.pyplot(fig2)

    with tab4:
        st.header("Dissipação do excesso de pressão neutra")

        fig3, ax3 = plt.subplots(figsize=(8, 5))

        times_to_plot = [
            0,
            int(n_t * 0.25),
            int(n_t * 0.50),
            int(n_t * 0.75),
            n_t - 1
        ]

        for idx in times_to_plot:
            ax3.plot(
                u_matrix[idx, :],
                z_vals,
                linewidth=2,
                label=f"t = {t_days[idx]:.0f} dias"
            )

        ax3.invert_yaxis()
        ax3.set_xlabel("Excesso de pressão neutra u [kPa]")
        ax3.set_ylabel("Profundidade z [m]")
        ax3.set_title("Distribuição de pressão neutra na camada")
        ax3.legend()
        ax3.grid(True)

        st.pyplot(fig3)

    st.markdown("---")

    st.header("Interpretação técnica")

    st.markdown(f"""
    Para os parâmetros inseridos, a camada de argila apresenta um recalque final estimado de
    **{S_final * 100:.2f} cm**. No tempo máximo analisado de **{t_max_days:.0f} dias**,
    o grau de consolidação calculado é de **{U[-1] * 100:.2f} %**.

    A solução foi obtida aplicando-se a Transformada de Laplace à equação diferencial parcial
    da consolidação unidimensional. No domínio de Laplace, o problema é transformado em uma
    equação diferencial ordinária em relação à profundidade. Posteriormente, a solução no domínio
    do tempo é recuperada por inversão numérica da Transformada de Laplace.
    """)

else:
    st.info("Insira os parâmetros na barra lateral e clique em Calcular.")