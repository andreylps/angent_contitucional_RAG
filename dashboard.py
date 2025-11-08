import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Define o caminho para o arquivo de log
# Define o caminho da raiz do projeto de forma robusta
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "logs" / "interactions.jsonl"


def load_log_data() -> pd.DataFrame:
    """Carrega os dados do arquivo de log JSONL para um DataFrame do Pandas."""
    if not LOG_FILE.exists():
        return pd.DataFrame()

    records = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Ignora linhas malformadas
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Converte o timestamp para um formato legível e ajusta para o fuso horário local
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(None)
    return df


# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Monitoramento - RAG Jurídico",
    page_icon="⚖️",
    layout="wide",
)

# --- Título e Atualização ---
st.title("⚖️ Dashboard de Monitoramento - RAG Jurídico")
st.markdown("Visualização das interações recentes com o sistema.")

if st.button("Recarregar Dados"):
    st.rerun()

# --- Carregar Dados ---
log_df = load_log_data()

if log_df.empty:
    st.warning(
        "Nenhum dado de interação encontrado. Execute o sistema para gerar logs."
    )
else:
    # --- Métricas Principais ---
    st.header("Métricas Gerais")
    total_interactions = len(log_df)
    avg_duration = log_df["duration_seconds"].mean()
    success_rate = (
        (log_df["status"] == "success").sum() / total_interactions * 100
        if total_interactions > 0
        else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Interações", f"{total_interactions}")
    col2.metric("Duração Média (s)", f"{avg_duration:.2f}")
    col3.metric("Taxa de Sucesso (%)", f"{success_rate:.2f}")

    # --- Gráficos ---
    st.header("Visualizações Gráficas")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Uso de Agentes")
        agent_usage = log_df["primary_agent"].value_counts().reset_index()
        agent_usage.columns = ["Agente", "Contagem"]
        fig_agent = px.pie(
            agent_usage,
            names="Agente",
            values="Contagem",
            title="Distribuição de Agentes Acionados",
            hole=0.3,
        )
        st.plotly_chart(fig_agent, use_container_width=True)

    with col_chart2:
        st.subheader("Distribuição de Status")
        status_dist = log_df["status"].value_counts().reset_index()
        status_dist.columns = ["Status", "Contagem"]
        fig_status = px.pie(
            status_dist,
            names="Status",
            values="Contagem",
            title="Distribuição de Status das Respostas",
            hole=0.3,
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # --- Tabela de Interações ---
    st.header("Últimas Interações")
    st.dataframe(
        log_df[
            [
                "timestamp_local",
                "original_query",
                "final_answer",
                "primary_agent",
                "duration_seconds",
                "status",
            ]
        ].sort_values(by="timestamp_local", ascending=False),
        use_container_width=True,
    )
