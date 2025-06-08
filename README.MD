# Classificação de E-mails Phishing Rede Neural

Este projeto tem como objetivo a classificação de e-mails como phishing ou legítimos, utilizando técnicas de Processamento de Linguagem Natural (PLN) e aprendizado de máquina.

## Visão Geral

O problema de detecção de e-mails de phishing foi abordado com uma base de dados que se mostrou relativamente simples. Durante a exploração, foi observado que o modelo desempenhou bem com diferentes configurações da rede neural.

Por isso, mesmo com uma arquitetura simples de rede neural e uma vetorização básica por Word2Vec, o modelo já atinge resultados satisfatórios.

## Pipeline de Execução

1. **Pré-processamento**:
   - Limpeza de URLs, pontuação e caracteres especiais
   - Remoção de acentos
   - Remoção de stopwords

2. **Vetorização com Word2Vec**:
   - Treinamento do Word2Vec nos próprios e-mails
   - Geração de vetores médios por e-mail a partir das palavras vetorizadas

3. **Rede Neural**:
   - Rede com 2 camadas escondidas
   - Ativação ReLU e Sigmoid
   - Otimizador: Adam
   - Função de perda: Binary Crossentropy

## Resultados

- O modelo demonstrou bom desempenho com apenas uma arquitetura simples.
- A base se mostrou relativamente fácil, permitindo alta acurácia.
- Conseguiu desempenhar de forma razoável com alguns dados reais de e-mails pessoais e dados sintéticos gerados pelo ChatGPT.

## Como Executar

1. Instale as dependências: pip install -r requirements.txt

2. Execute os notebooks na pasta `notebooks/` para visualizar o processamento de dados e a criação do modelo passo a passo.

3. Rode o app.py: streamlit run app/app.py

## Sobre a construção

- Os modelos do w2v e da rede neural (gerados pelos notebooks) são salvos na pasta "models" para serem utilizados na aplicação do streamlit.
- Na aplicação do streamlit, a entrada do usuário é processada e tratada pelo "data_processing.py", utilizando o w2v salvo em "modelos". Depois disso, o resultado é processado pelo modelo de rede neural salvo em pkl em "modelos".
