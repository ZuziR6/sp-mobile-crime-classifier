# Segurança Pública Inteligente em São Paulo — Classificação de ocorrências com MLP

Global Solution da disciplina **Aprendizado Profundo com Redes Neurais Convolucionais**.

Uma rede neural **Multilayer Perceptron (MLP)** que classifica uma ocorrência registrada na cidade de São Paulo como **roubo** ou **furto**, a partir de atributos como bairro, mês, dia da semana, período do dia e tipo de local. O modelo escolhido está integrado a uma aplicação web em Streamlit.

> ### O que o modelo faz — e o que não faz
> A MLP analisa as características de uma ocorrência **já registrada (ou hipotética)** e estima se ela é mais compatível com *roubo* ou com *furto*.
> Ela **não prevê** se, onde ou quando um crime vai acontecer. Uma saída como "roubo = 72%" significa que, **dado** que uma ocorrência tem aquelas características, o modelo atribui 72% de probabilidade à classe roubo. **Não** significa que há 72% de chance de ocorrer um roubo naquele local e horário.

## Aplicação web

**https://spmobilecrimeclassifier.streamlit.app/**

O formulário reflete exatamente os atributos usados no treinamento: bairro, mês, dia da semana, período do dia, tipo de local e subtipo de local. Ao clicar em *Classificar ocorrência*, a aplicação mostra a classe estimada e as probabilidades de furto e de roubo.

## Equipe

Gabriel Guimarães de Oliveira - Rm567835
Pedro Paulo Ferreira Agnelo D'angelo - Rm567564
Mark Lima Leal - Rm566760

## Fonte dos dados e procedimento de obtenção

- **Fonte pública:** Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP), Portal de Estatísticas — <https://www.ssp.sp.gov.br/estatistica/consultas>
- **Base:** *Celulares Subtraídos*, ano **2025** (arquivo `CelularesSubtraidos_2025.xlsx`, aba `CELULAR_2025`, 383.635 registros × 55 colunas).
- **Como obter:** acesse o link acima → **Bases de Dados** → **Celulares Subtraídos** → **2025** → *Download*.

Os dados **não são versionados neste repositório** (a planilha tem ≈ 103 MB). A própria SSP-SP informa que a base é extraída direto dos sistemas de registro, sem padronização, e que não constitui estatística oficial.

### Recorte e rótulo

| Etapa | Regra | Resultado |
|---|---|---|
| Local do fato | `CIDADE == "S.PAULO"` (cidade dos fatos, não o local de registro) e data da ocorrência em 2025 | 213.411 linhas |
| Natureza | `RUBRICA` = *Furto (art. 155)* ou *Roubo (art. 157)* | 181.707 linhas |
| Ambiguidade | Remove boletins com as duas naturezas | −44 boletins |
| Consistência | Remove boletins cujas linhas divergem em bairro, tipo ou subtipo de local | −820 boletins |
| Consolidação | Uma linha por boletim (`NOME_DELEGACIA` + `ANO_BO` + `NUM_BO`) | **158.859 boletins** |

**Alvo:** `0` = Furto (63,31%) · `1` = Roubo (36,69%). A coluna `RUBRICA` não entra nas entradas do modelo (evita *data leakage*).

### Atributos de entrada

`BAIRRO`, `MES`, `DIA_SEMANA`, `DESCR_PERIODO`, `DESCR_TIPOLOCAL`, `DESCR_SUBTIPOLOCAL` — todos categóricos, codificados com *one-hot* (293 entradas; categorias com menos de 50 ocorrências no treino são agrupadas). O codificador é ajustado apenas com o conjunto de treino.

## Modelo

- **Divisão:** 70% treino (111.201) · 15% validação (23.829) · 15% teste (23.829), estratificada, `random_state=42`.
- **Estrutura:** `Entrada(293) → [Dense(n, ReLU, L2 1e-4) → Dropout(0,20)] × k → Dense(1, sigmoid)`.
- **Treinamento:** perda *binary crossentropy*, otimizador Adam (taxa inicial 1e-3), batch 256, até 50 épocas com *early stopping* e `ReduceLROnPlateau`.

### Arquiteturas comparadas (validação)

| Modelo | Camadas ocultas | Acurácia | Precision (roubo) | Recall (roubo) | F1 (roubo) | F1 macro |
|---|---|---|---|---|---|---|
| **MLP_1_Compacta** (escolhido) | 64 → 32 | 0,8302 | 0,7602 | 0,7851 | 0,7724 | 0,8185 |
| MLP_2_Intermediaria | 128 → 64 → 32 | 0,8296 | 0,7640 | 0,7750 | 0,7695 | 0,8172 |
| MLP_3_Profunda | 256 → 128 → 64 | 0,8302 | 0,7647 | 0,7761 | 0,7703 | 0,8178 |

### Resultado final (teste, limiar 0,50)

| Acurácia | Precision (roubo) | Recall (roubo) | F1 (roubo) | F1 macro |
|---|---|---|---|---|
| 0,8319 | 0,7587 | 0,7947 | 0,7763 | 0,8208 |

A arquitetura foi escolhida pela validação (menor `val_loss`, maior F1 macro e menor capacidade); o conjunto de teste foi usado uma única vez. As três arquiteturas ficaram praticamente empatadas, e o critério decisivo foi a parcimônia. Detalhes, curvas de treinamento, matrizes de confusão e análise de overfitting/underfitting estão no notebook.

## Estrutura do repositório

```
.
├── app.py                                        # aplicação Streamlit
├── modelo_app/
│   ├── modelo_mlp.keras                          # MLP treinada
│   ├── preprocessor.joblib                       # one-hot ajustado no treino
│   └── configuracao.json                         # atributos, classes e limiar (0,50)
├── Global_Solution_Seguranca_Publica_MLP.ipynb   # notebook completo
├── requirements.txt
└── README.md
```

## Como executar a aplicação localmente

```bash
git clone https://github.com/ZuziR6/sp-mobile-crime-classifier.git
cd sp-mobile-crime-classifier

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

A aplicação abre em `http://localhost:8501`. O modelo e o pré-processador já estão em `modelo_app/`; não é necessário treinar para usá-la.

## Como reproduzir o treinamento

1. Baixe a planilha da SSP-SP (instruções acima).
2. Abra `Global_Solution_Seguranca_Publica_MLP.ipynb` no Google Colab e execute **Runtime → Run all**; na célula de upload, selecione `CelularesSubtraidos_2025.xlsx`. (Fora do Colab, coloque o arquivo na mesma pasta do notebook.)
3. O notebook regenera `modelo_mlp.keras`, `preprocessor.joblib`, `configuracao.json` e `app.py` em `/content/modelo_app`. Copie-os para este repositório.

> O `.keras` e o `.joblib` devem ser carregados com as mesmas versões de TensorFlow e scikit-learn usadas no treino (veja `requirements.txt`).

## Limitações

- Poucos atributos, todos categóricos e de contexto: o desempenho se estabiliza em ≈ 83% de acurácia, independente da arquitetura.
- Dados administrativos sem tratamento: há variações de grafia de bairro (por exemplo, `BRAS`/`BRÁS`) que o pré-processamento não normaliza. Em 48,6% dos boletins o período do dia não foi informado.
- A base só contém fatos que viraram boletim de ocorrência; a classificação em roubo ou furto depende de como o fato foi registrado.
- Escopo restrito: celulares, capital paulista e ano de 2025.
- Uso acadêmico e experimental: o modelo **não** estima risco de crime, não deve orientar policiamento e não substitui análise humana.

## Uso de IA generativa

Assistentes de IA generativa foram usados como apoio em programação, depuração, documentação e análise dos resultados. A equipe revisou e compreende o código, o processo de treinamento, o funcionamento das camadas da MLP e os critérios de escolha da arquitetura.

## Fonte

Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP) — Portal de Estatísticas: <https://www.ssp.sp.gov.br/estatistica/consultas>
