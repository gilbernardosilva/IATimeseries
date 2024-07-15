# Análise de Séries Temporais e Previsão da Ocupação de Salas de Aula: Um Estudo com Dados de Sensores

## Introdução

**Motivação:**

A qualidade do ar interior (QAI) é um fator crucial para o bem-estar e a produtividade em ambientes fechados como salas de aula. A concentração de dióxido de carbono (CO2) é um indicador chave da QAI, pois níveis elevados de CO2 estão associados a sintomas como dores de cabeça, fadiga e dificuldades de concentração. Além disso, a monitorização da ocupação das salas de aula pode auxiliar na gestão eficiente dos espaços e na otimização do uso de energia.

**Objetivos:**

Este projeto tem como objetivos:

1. **Analisar e visualizar:** Explorar os dados de sensores (CO2, temperatura, umidade) coletados em salas de aula da UFP e/ou da escola secundária de Alpendurada, buscando padrões e correlações.
2. **Desenvolver modelos de previsão:** Construir e avaliar modelos de machine learning para prever a ocupação das salas de aula com base nos dados dos sensores, com foco principal no CO2.
3. **Avaliar o desempenho:** Comparar o desempenho dos diferentes modelos de previsão em termos de métricas de erro, como o erro quadrático médio (MSE), erro absoluto médio (MAE) e raiz quadrada do erro quadrático médio (RMSE).

## Descrição do Problema

O problema central é estimar a ocupação de salas de aula (número de pessoas presentes) utilizando dados de sensores de CO2, temperatura e umidade. A hipótese é que a concentração de CO2 aumenta com o número de pessoas na sala, e que essa relação pode ser modelada para prever a ocupação.

## Estado da Arte

A previsão da ocupação de espaços fechados tem sido objeto de estudo em diversas áreas, como gestão de edifícios, eficiência energética e saúde pública. Diversas abordagens têm sido utilizadas, incluindo:

* **Modelos estatísticos:** Modelos lineares, ARIMA, etc.
* **Machine learning:** Regressão linear, árvores de decisão, redes neurais (incluindo LSTM), etc.
* **Deep learning:** Redes neurais mais complexas, como Transformers.
* **Sensores:** Uso de diferentes tipos de sensores, como CO2, temperatura, umidade, movimento, Wi-Fi, etc.

## Descrição do Trabalho Realizado

1. **Coleta e Pré-processamento de Dados:**
   - Obtenção de datasets do projeto AirMon contendo dados de sensores de CO2, temperatura e umidade em salas de aula.
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






## Conclusões e Perspectivas de Desenvolvimento

*Resumir as principais conclusões do trabalho, destacando o modelo que apresentou o melhor desempenho na previsão da ocupação das salas de aula.*

*Discutir as limitações do estudo e propor possíveis melhorias, como:*

* **Incorporar mais dados:** Utilizar datasets maiores e mais diversos para aumentar a robustez dos modelos.
* **Testar outros modelos:** Explorar outros algoritmos de machine learning e deep learning, como redes neurais convolucionais (CNN) ou modelos híbridos.
* **Incluir mais sensores:** Adicionar dados de outros sensores, como temperatura, umidade e movimento, para melhorar a precisão das previsões.
* **Considerar o contexto:** Incorporar informações contextuais, como horário das aulas, tipo de sala e eventos especiais, para refinar os modelos.
