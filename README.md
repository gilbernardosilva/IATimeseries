# Análise de Séries Temporais e Previsão da Ocupação de Salas de Aula: Um Estudo com Dados de Sensores

## Introdução

**Motivação:**

A qualidade do ar interior (QAI) é um fator crucial para o bem-estar e a produtividade em ambientes fechados como salas de aula. A concentração de dióxido de carbono (CO2) é um indicador chave da QAI, pois níveis elevados de CO2 estão associados a sintomas como dores de cabeça, fadiga e dificuldades de concentração. Além disso, a monitorização da ocupação das salas de aula pode auxiliar na gestão eficiente dos espaços e na otimização do uso de energia.

**Objetivos:**

Este projeto tem como objetivos:

1. **Analisar e visualizar:** Explorar os dados de sensores (CO2, temperatura, humidade) coletados em salas de aula da UFP e/ou da escola secundária de Alpendurada, buscando padrões e correlações.
2. **Desenvolver modelos de previsão:** Construir e avaliar modelos de machine learning para prever a ocupação das salas de aula com base nos dados dos sensores, com foco principal no CO2.
3. **Avaliar o desempenho:** Comparar o desempenho dos diferentes modelos de previsão em termos de métricas de erro, como o erro quadrático médio (MSE), erro absoluto médio (MAE) e raiz quadrada do erro quadrático médio (RMSE).

## Descrição do Problema

O problema central é estimar a ocupação de salas de aula (número de pessoas presentes) utilizando dados de sensores de CO2, temperatura e humidade. A hipótese é que a concentração de CO2 aumenta com o número de pessoas na sala, e que essa relação pode ser modelada para prever a ocupação.

## Estado da Arte

A previsão da ocupação de espaços fechados tem sido objeto de estudo em diversas áreas, como gestão de edifícios, eficiência energética e saúde pública. Diversas abordagens têm sido utilizadas, incluindo:

* **Modelos estatísticos:** Modelos lineares, ARIMA, etc.
* **Machine learning:** Regressão linear, árvores de decisão, redes neurais (incluindo LSTM), etc.
* **Deep learning:** Redes neurais mais complexas, como Transformers.
* **Sensores:** Uso de diferentes tipos de sensores, como CO2, temperatura, humidade, movimento, Wi-Fi, etc.

## Descrição do Trabalho Realizado

1. **Coleta e Pré-processamento de Dados:**
   - Obtenção de datasets do projeto AirMon contendo dados de sensores de CO2, temperatura e humidade em salas de aula.
   - Limpeza dos dados: remoção de valores ausentes, outliers e duplicatas.
   - Transformação dos dados: extração de atributos de data e hora para análise temporal.

2. **Análise Exploratória de Dados:**
   - Visualização da distribuição dos dados de cada sensor.
   - Análise da variação temporal dos dados, buscando identificar padrões diários, semanais ou sazonais.
   - Cálculo de estatísticas descritivas para cada sensor.

3. **Divisão dos Dados:**
   - Separação dos dados em conjuntos de treinamento e teste para avaliação dos modelos.

4. **Modelagem e Previsão:**
   - **Prophet:** Modelo de previsão de séries temporais do Facebook, que incorpora automaticamente componentes de tendência, sazonalidade e feriados.
   - **LSTM (Long Short-Term Memory):** Tipo de rede neural recorrente eficaz na modelagem de dados sequenciais e temporais.
   - **Random Forest:** Algoritmo de aprendizado de máquina baseado em árvores de decisão que pode ser usado para regressão.

5. **Avaliação do Desempenho:**
   - Cálculo de métricas de erro (MSE, RMSE, MAE) para cada modelo e alvo (CO2, Partículas).
   - Comparação dos resultados em uma tabela para identificar o modelo com melhor desempenho.
   - Visualização das previsões em relação aos valores reais para análise qualitativa.

## Análise de Resultados





## Error Metrics Comparison:

| Modelo                   | Alvo                |      MSE |     RMSE |      MAE |   STD_DEV |     MEAN |
|--------------------------|---------------------|----------:|---------:|---------:|----------:|---------:|
| Prophet                 | Particules 1        | 70.477306 |  8.39508 |  6.54201 |  7.40080 |  7.92116 |
| LSTM                    | Particules 1        |  2.26234 |  1.50411 |  0.96214 |  7.40090 |  7.92544 |
| Random Forest            | Particules 1        |  3.53628 |  1.88050 |  1.08570 |  7.40090 |  7.92544 |
| Prophet                 | Particules 2.5      | 174.09058 | 13.19434 | 10.42809 | 11.67977 | 13.23929 |
| LSTM                    | Particules 2.5      |  6.79463 |  2.60665 |  1.71237 | 11.67985 | 13.24614 |
| Random Forest            | Particules 2.5      | 11.72906 |  3.42477 |  1.97846 | 11.67985 | 13.24614 |
| Prophet                 | Particules 10       | 248.63454 | 15.76815 | 12.05699 | 14.46059 | 16.10358 |
| LSTM                    | Particules 10       | 14.83924 |  3.85217 |  2.51360 | 14.46099 | 16.11172 |
| Random Forest            | Particules 10       | 31.13180 |  5.57959 |  3.04187 | 14.46099 | 16.11172 |
| Prophet                 | CO2                 | 971473.48 | 985.63354 | 521.13377 | 973.70317 | 1065.41659|
| LSTM                    | CO2                 |  6902.1908 | 83.07943 | 40.09228 | 973.93849 | 1065.66544|
| Random Forest            | CO2                 | 39964.1458 | 199.91034 | 75.75867 | 973.93849 | 1065.66544|



## Conclusões e Perspectivas de Desenvolvimento

*Resumir as principais conclusões do trabalho, destacando o modelo que apresentou o melhor desempenho na previsão da ocupação das salas de aula.*

*Discutir as limitações do estudo e propor possíveis melhorias, como:*

* **Incorporar mais dados:** Utilizar datasets maiores e mais diversos para aumentar a robustez dos modelos.
* **Testar outros modelos:** Explorar outros algoritmos de machine learning e deep learning, como redes neurais convolucionais (CNN) ou modelos híbridos.
* **Incluir mais sensores:** Adicionar dados de outros sensores, como temperatura, humidade e movimento, para melhorar a precisão das previsões.
* **Considerar o contexto:** Incorporar informações contextuais, como horário das aulas, tipo de sala e eventos especiais, para refinar os modelos.
