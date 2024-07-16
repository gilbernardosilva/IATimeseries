# Análise Avançada de Séries Temporais e Previsão da Ocupação de Salas de Aula com Base em Dados de Sensores de Qualidade do Ar

## Introdução

**Motivação:**

A qualidade do ar interior (QAI) é um fator crucial para o bem-estar e a produtividade em ambientes fechados como salas de aula. A concentração de dióxido de carbono (CO2) é um indicador chave da QAI, pois níveis elevados de CO2 estão associados a sintomas como dores de cabeça, fadiga e dificuldades de concentração. Além disso, a monitorização da ocupação das salas de aula pode auxiliar na gestão eficiente dos espaços e na otimização do uso de energia.

**Objetivos:**

Este projeto tem como objetivos:

1. **Analisar e visualizar:** Explorar os dados de sensores (CO2, temperatura, humidade, etc) coletados em salas de aula da UFP e/ou da escola secundária de Alpendurada, buscando padrões e correlações.
2. **Desenvolver modelos de previsão:** Construir e avaliar modelos de machine learning para prever a ocupação das salas de aula com base nos dados dos sensores, com foco principal no CO2 e nas Particulas 1, 2.5 e 10.
3. **Avaliar o desempenho:** Comparar o desempenho dos diferentes modelos de previsão em termos de métricas de erro, como o erro quadrático médio (MSE), erro absoluto médio (MAE) e raiz quadrada do erro quadrático médio (RMSE).

## Descrição do Problema

O problema central é estimar a ocupação de salas de aula (número de pessoas presentes) utilizando dados de sensores. A hipótese é que a concentração de CO2 aumenta com o número de pessoas na sala, e que essa relação pode ser modelada para prever a ocupação.

## Estado da Arte

A previsão da ocupação de espaços fechados tem sido objeto de estudo em diversas áreas, como gestão de edifícios, eficiência energética e saúde pública. Diversas abordagens têm sido utilizadas, incluindo:

* **Modelos estatísticos:** Modelos lineares, ARIMA, etc.
* **Machine learning:** Regressão linear, árvores de decisão, redes neurais (incluindo LSTM), etc.
* **Deep learning:** Redes neurais mais complexas, como Transformers.
* **Sensores:** Uso de diferentes tipos de sensores, como CO2, temperatura, humidade, janelas, etc.

## Descrição do Trabalho Realizado

1. **Coleta e Pré-processamento de Dados:**
   - Obtenção de datasets do projeto AirMon contendo dados de sensores das salas de aula.
   - Limpeza dos dados: remoção de valores ausente, nulos e duplicados.
   - Transformação dos dados: extração de atributos de data e hora para análise temporal e associação a sazonalidades.

2. **Análise Exploratória de Dados:**
   - Visualização da distribuição dos dados de cada sensor.
   - Análise da variação temporal dos dados, buscando identificar padrões diários, semanais ou sazonais.
   - Cálculo de estatísticas descritivas para cada sensor.

3. **Divisão dos Dados:**
   - Separação dos dados em conjuntos de treino e teste para avaliação dos modelos.

4. **Modelagem e Previsão:**
   - **Prophet:** Modelo de previsão de séries temporais do Facebook, que incorpora automaticamente componentes de tendência e sazonalidade.
   - **LSTM (Long Short-Term Memory):** Tipo de rede neural recorrente eficaz na modelagem de dados sequenciais e temporais.
   - **Random Forest:** Algoritmo de machine learning baseado em árvores de decisão que pode ser usado para regressão.

5. **Avaliação do Desempenho:**
   - Cálculo de métricas de erro (MSE, RMSE, MAE, STD_DEV, MEAN) para cada modelo e alvo (CO2, Partículas).
   - Comparação dos resultados numa tabela para identificar o modelo com melhor desempenho.
   - Visualização das previsões em relação aos valores reais para análise qualitativa.


### Análise de Resultados

Inicialmente, realizamos um levantamento das variáveis em análise para assegurar a adequação dos gráficos de previsão e atualidade. Isso foi feito através de matrizes de correlação com heatmaps, gráficos de séries temporais e plot boxes.
Através dos mesmos conseguimos pois determinar que regressores usar nos nossos modelos, para determinar as variáveis que mais vão impactar o modelo positivamente.

#### Regressores

Os regressores são variáveis independentes utilizadas em modelos estatísticos e preditivos para explicar e prever variações em uma variável dependente. Eles ajudam a aumentar a precisão dos modelos ao considerar múltiplos fatores que podem influenciar o resultado desejado, controlando fatores de confusão e identificando relações causais entre variáveis. A inclusão de regressores relevantes permite selecionar as variáveis mais importantes para o modelo, melhorando a interpretabilidade e a eficácia das análises realizadas.

No nosso caso, para o nosso *use case* utilizamos a seguinte lista de regressores para o CO2 e as partículas que foram escolhidos através dos mecanismos expostos a seguir.

| ![Regressores](<imgs/Regressors.png>)| 
|:--:| 
| *Lista de Regressores de Partículas e CO2* |
  

#### Matrizes de Correlação

Matrizes de correlação com heatmaps são essenciais para explorar relações entre variáveis em conjuntos de dados complexos, oferecendo tanto uma representação numérica detalhada quanto uma visualização gráfica intuitiva das correlações. Eles são utilizados principalmente na análise exploratória de dados para entender padrões e informar a escolha de variáveis em análises estatísticas e modelos preditivos.
No nosso caso utilizamos a mesma para determinar que regressores usar nos modelos.

| ![Matriz de Correlação](<imgs/Correlation Matrix.png>)| 
|:--:| 
| *Matriz de Correlação do dataset* |
  
#### Box Plots

Os box plots são gráficos que resumem a distribuição dos dados, destacando a mediana, quartis e outliers. São úteis para comparar distribuições, identificar outliers e visualizar assimetrias nos dados.

- **Resumo Estatístico**: Apresentam visualmente a distribuição dos dados, destacando a mediana, quartis e valores atípicos.
- **Comparação**: Permitem comparar várias distribuições lado a lado.
- **Identificação de Outliers**: Facilitam a detecção de valores atípicos.
- **Detecção de Assimetria**: Ajudam a visualizar se os dados apresentam uma distribuição simétrica ou enviesada.

#### Utilidade dos Gráficos de Séries Temporais

Os gráficos de séries temporais são essenciais para entender a evolução dos dados ao longo do tempo, sendo utilizados em diversas aplicações analíticas e estratégicas.

Eles permitem a  Identificação de Tendências, Análise de Sazonalidade, Detecção de Ciclos, Avaliação de Mudanças e Eventos, Previsão e Planeamento, Monitorazação de Desempenho, entre outros.

### Exemplo Prático

Os gráficos de séries temporais utilizados ilustram as concentrações de diferentes partículas e níveis de CO2 de maio de 2023 a fevereiro de 2024. Cada subgráfico representa uma medição distinta da variável ao longo do tempo.

| ![Gráficos de série temporal](<imgs/Actual data.png>)| 
|:--:| 
| *Gráfico de Series de Partículas e CO2.* |
  

Plotboxes das variavéis em estudo.

| ![Distribuição de CO2](<imgs/Distribuition of CO2.png>) | ![Distribuição de Partículas 1](<imgs/Distribuition of Particules 1.png>) | ![Distribuição de Partículas 2.5](<imgs/Distribuition of Particules 2.5.png>) | ![Distribuição de Partículas 10](<imgs/Distribuition of Particules 10.png>) |
|:-----------------------------------------------------:|:-------------------------------------------------------:|:---------------------------------------------------------:|:-------------------------------------------------------:|
| Distribuição de concentração de CO2  | Distribuição de partículas 1           | Distribuição de partículas 2.5              | Distribuição de partículas 10            |


### Comparação de Resultados
A tabela abaixo apresenta os resultados da comparação dos modelos Prophet, LSTM e Random Forest para a previsão da concentração de partículas (1, 2.5 e 10) e CO2, utilizando as métricas MSE, RMSE, MAE, STD_DEV e MEAN:

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

**Análise:**

A análise das métricas de erro revela um desempenho superior consistente do modelo LSTM em comparação com Prophet e Random Forest para todos os alvos (partículas e CO2). O LSTM apresenta os menores valores de MSE, RMSE e MAE, indicando uma maior precisão nas previsões e menor magnitude dos erros.

O modelo Prophet demonstra dificuldades em prever a concentração de CO2, exibindo os maiores erros entre os modelos. As previsões de partículas também apresentam erros consideráveis, sugerindo que o modelo não captura adequadamente a complexidade das relações temporais e a influência dos regressores nesse conjunto de dados.

O modelo Random Forest apresenta um desempenho intermediário, superando o Prophet em todos os casos, mas sendo menos preciso que o LSTM. Isso pode indicar que a estrutura baseada em árvores do Random Forest consegue capturar algumas das não linearidades nos dados, mas não tão efetivamente quanto o LSTM, que é projetado para lidar com sequências temporais.

**Escolha do Melhor Modelo:**

Com base nos resultados, o modelo LSTM destaca-se como a melhor opção para prever a ocupação de salas de aula, considerando os dados e alvos analisados. A sua capacidade de aprender padrões complexos em séries temporais e incorporar informações de regressores resulta em previsões mais precisas e com menor erro em relação aos outros modelos.

**Gráficos:**


**Partículas 1 (PM1):**

| LSTM                                                                                   | Prophet                                                                                  | Random Forest                                                                              |
| :-----------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| ![Previsões de partículas 1 - LSTM](<imgs/LSTM Particules 1.png>) | ![Previsões de partículas 1 - Prophet](<imgs/Prophet Particules 1.png>) | ![Previsões de partículas 1 - Random Forest](<imgs/Random Forest Particules 1.png>) |
| *Previsões de partículas 1 - LSTM*                                                     | *Previsões de partículas 1 - Prophet*                                                    | *Previsões de partículas 1 - Random Forest*                                                |

**Partículas 2.5 (PM2.5):**

| LSTM                                                                                   | Prophet                                                                                  | Random Forest                                                                              |
| :-----------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| ![Previsões de partículas 2.5 - LSTM](<imgs/LSTM Particules 2.5.png>) | ![Previsões de partículas 2.5 - Prophet](<imgs/Prophet Particules 2.5.png>) | ![Previsões de partículas 2.5 - Random Forest](<imgs/Random Forest Particules 2.5.png>) |
| *Previsões de partículas 2.5 - LSTM*                                                     | *Previsões de partículas 2.5 - Prophet*                                                    | *Previsões de partículas 2.5 - Random Forest*                                                |


**Partículas 10 (PM10):**

| LSTM                                                                                   | Prophet                                                                                  | Random Forest                                                                              |
| :-----------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| ![Previsões de partículas 10 - LSTM](<imgs/LSTM Particules 10.png>) | ![Previsões de partículas 10 - Prophet](<imgs/Prophet Particules 10.png>) | ![Previsões de partículas 10 - Random Forest](<imgs/Random Forest Particules 10.png>) |
| *Previsões de partículas 10 - LSTM*                                                     | *Previsões de partículas 10 - Prophet*                                                    | *Previsões de partículas 10 - Random Forest*                                                |

**Dióxido de Carbono (CO2):**

| LSTM                                                                                   | Prophet                                                                                  | Random Forest                                                                              |
| :-----------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| ![Previsões de partículas CO2 - LSTM](<imgs/LSTM CO2.png>)   | ![Previsões de partículas CO2 - Prophet](<imgs/Prophet CO2.png>)   | ![Previsões de partículas CO2 - Random Forest](<imgs/Random Forest CO2.png>)   |
| *Previsões de CO2 - LSTM*                                                            | *Previsões de CO2 - Prophet*                                                           | *Previsões de CO2 - Random Forest*                                                           |


**Considerações:**

É importante ressaltar que a superioridade do LSTM não garante um desempenho ideal em todas as situações. A escolha do modelo ideal pode variar dependendo das características específicas dos dados, dos requisitos do problema e dos recursos computacionais disponíveis. Adicionalmente, a análise exploratória dos dados e a otimização dos hiperparâmetros dos modelos podem influenciar significativamente os resultados.


## Conclusões e Perspectivas de Desenvolvimento

Este estudo explorou a viabilidade de prever a ocupação de salas de aula utilizando dados de sensores de CO2, partículas e outros fatores ambientais. A análise comparativa dos modelos Prophet, LSTM e Random Forest revelou a superioridade do LSTM em termos de precisão, com os menores erros de previsão (MSE, RMSE e MAE) para todos os alvos.

O LSTM demonstrou uma capacidade notável de capturar padrões complexos e não-lineares nos dados das séries temporais, superando as limitações dos modelos Prophet e Random Forest. Essa vantagem é particularmente evidente na previsão da concentração de CO2, onde o Prophet apresentou um desempenho significativamente inferior.

No entanto, é importante reconhecer as limitações deste estudo. A análise foi realizada em um conjunto de dados específico do projeto AirMon, que pode não ser representativo de todas as condições e cenários de ocupação de salas de aula. Adicionalmente, a utilização de apenas alguns tipos de sensores pode restringir a capacidade dos modelos de capturar informações relevantes para a previsão da ocupação.

**Perspectivas de Desenvolvimento:**

Para aprimorar a precisão e a robustez das previsões, futuras pesquisas podem explorar as seguintes direções:

1. **Incorporação de Dados Mais Abrangentes:** A utilização de datasets maiores e mais diversos, abrangendo diferentes tipos de salas de aula, horários e condições ambientais, pode fortalecer a capacidade de generalização dos modelos.

2. **Exploração de Modelos Híbridos e de transformados:** A combinação de diferentes modelos, como LSTM e Random Forest, em um modelo híbrido é uma estratégia avançada para aproveitar os pontos fortes de cada abordagem e melhorar o desempenho geral de um sistema de previsão ou modelagem. Além disso, a utilização de modelos transformados pode enriquecer ainda mais essa abordagem.

3. **Inclusão de Dados de Outros Sensores:** A integração de dados de outros sensores pode fornecer informações adicionais relevantes para a previsão da ocupação, especialmente em situações em que a concentração de CO2 não é o único fator determinante.

4. **Consideração do Contexto:** A incorporação de informações contextuais, como horário das aulas, tipo de sala (laboratório, auditório, etc.) e eventos especiais (palestras, exames), pode refinar os modelos e torná-los mais sensíveis às nuances da ocupação das salas de aula.

Seria interessante no futuro dispor de algum tempo para testar os modelos mais aprofundadamente, com datasets maiores, regressores diferentes e hiperparametrização diferente. Mesmo assim, no nosso caso e com os nossos testes, adquirimos modelos bastante satisfatórios.

Em suma, este estudo demonstra o potencial do uso de modelos de machine learning, para prever a ocupação de salas de aula com base em dados de sensores. As perspectivas de desenvolvimento apontam para um futuro promissor na otimização da gestão de espaços e na melhoria da qualidade do ar interior em ambientes educacionais.

Trabalho elaborado por:
Gil Silva 41805
Nuno Moutinho 47181