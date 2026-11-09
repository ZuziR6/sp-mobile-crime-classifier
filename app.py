
import json
import joblib
import streamlit as st
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Caminho dos arquivos do modelo
PASTA = Path(__file__).parent / "modelo_app"

st.set_page_config(
    page_title="SP Mobile Crime Classifier",
    page_icon="🔎",
    layout="centered"
)

@st.cache_resource
def carregar_componentes():
    modelo = tf.keras.models.load_model(
        PASTA / "modelo_mlp.keras"
    )

    preprocessor = joblib.load(
        PASTA / "preprocessor.joblib"
    )

    with open(
        PASTA / "configuracao.json",
        "r",
        encoding="utf-8"
    ) as arquivo:
        config = json.load(arquivo)

    return modelo, preprocessor, config


modelo, preprocessor, config = carregar_componentes()

encoder = preprocessor.named_transformers_["categoricas"]

colunas = config["features"]

categorias = {
    coluna: [str(valor) for valor in valores]
    for coluna, valores in zip(colunas, encoder.categories_)
}

# Interface
st.title("SP Mobile Crime Classifier")

st.subheader(
    "Classificação de ocorrências de furto e roubo de celulares"
)

st.write(
    "Modelo de Machine Learning desenvolvido com "
    "Multilayer Perceptron (MLP), utilizando dados públicos "
    "da Secretaria da Segurança Pública de São Paulo."
)

st.info(
    "Informe as características de uma ocorrência para "
    "obter a classificação estimada pelo modelo."
)

st.divider()

st.header("Dados da ocorrência")

with st.form("formulario_ocorrencia"):

    bairro = st.selectbox(
        "Bairro",
        options=categorias["BAIRRO"]
    )

    meses = sorted(
        categorias["MES"],
        key=int
    )

    nomes_meses = {
        "1": "Janeiro", "2": "Fevereiro",
        "3": "Março", "4": "Abril",
        "5": "Maio", "6": "Junho",
        "7": "Julho", "8": "Agosto",
        "9": "Setembro", "10": "Outubro",
        "11": "Novembro", "12": "Dezembro"
    }

    mes = st.selectbox(
        "Mês",
        options=meses,
        format_func=lambda x: nomes_meses.get(x, x)
    )

    dias = sorted(
        categorias["DIA_SEMANA"],
        key=int
    )

    nomes_dias = {
        "0": "Segunda-feira",
        "1": "Terça-feira",
        "2": "Quarta-feira",
        "3": "Quinta-feira",
        "4": "Sexta-feira",
        "5": "Sábado",
        "6": "Domingo"
    }

    dia_semana = st.selectbox(
        "Dia da semana",
        options=dias,
        format_func=lambda x: nomes_dias.get(x, x)
    )

    periodo = st.selectbox(
        "Período do dia",
        options=categorias["DESCR_PERIODO"]
    )

    tipo_local = st.selectbox(
        "Tipo de local",
        options=categorias["DESCR_TIPOLOCAL"]
    )

    subtipo_local = st.selectbox(
        "Subtipo do local",
        options=categorias["DESCR_SUBTIPOLOCAL"]
    )

    enviar = st.form_submit_button(
        "Classificar ocorrência",
        use_container_width=True
    )

# Processamento e classificação
if enviar:

    dados = pd.DataFrame([{
        "BAIRRO": bairro,
        "MES": mes,
        "DIA_SEMANA": dia_semana,
        "DESCR_PERIODO": periodo,
        "DESCR_TIPOLOCAL": tipo_local,
        "DESCR_SUBTIPOLOCAL": subtipo_local
    }], columns=colunas)

    entrada = preprocessor.transform(dados)

    if entrada.shape[1] != config[
        "quantidade_features_processadas"
    ]:
        st.error("Dimensão dos atributos incompatível.")
        st.stop()

    prob_roubo = float(
        modelo.predict(entrada, verbose=0)[0][0]
    )

    prob_furto = 1 - prob_roubo

    limiar = config["limiar_classificacao"]

    classe = int(prob_roubo >= limiar)

    resultado = config["classes"][str(classe)]

    st.divider()
    st.header("Resultado da classificação")

    if resultado == "Roubo":
        st.error("Classificação estimada: ROUBO")
    else:
        st.success("Classificação estimada: FURTO")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Furto", f"{prob_furto * 100:.2f}%")

    with col2:
        st.metric("Roubo", f"{prob_roubo * 100:.2f}%")

    st.caption(f"Limiar utilizado: {limiar:.2f}")

    st.progress(prob_roubo)

    st.warning(
        "As probabilidades são estimativas do modelo e não "
        "representam certeza ou previsão de crimes futuros. "
        "Esta ferramenta possui finalidade acadêmica e experimental."
    )

st.divider()

st.caption(
    "Projeto acadêmico | Fonte: SSP-SP | Base de celulares subtraídos 2025"
)

