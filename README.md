# Analise de Sobrevivencia de Empresas Brasileiras - Quanto Tempo Dura um CNPJ

Este projeto investiga dois problemas complementares a partir dos Dados Abertos do CNPJ: quanto tempo empresas brasileiras permanecem ativas e quais sinais ajudam a antecipar baixa no horizonte de 12 meses. O trabalho foi estruturado como um case completo de Data Science, com auditoria da base, desenho temporal explicito, analise de sobrevivencia e modelagem preditiva calibrada.

## O que o projeto responde

- qual e a duracao observada das empresas brasileiras
- como a sobrevivencia varia por coorte, porte, UF e atividade economica
- quais sinais cadastrais conseguem antecipar risco de baixa sem recorrer a leakage temporal

## Base e desenho analitico

O estudo usa o snapshot `2026-03` dos Dados Abertos do CNPJ como fonte principal e organiza o trabalho em oito fases, separando integridade, consolidacao, auditoria, analise exploratoria, sobrevivencia, modelagem, storytelling e artefatos finais. A unidade analitica foi ancorada em `cnpj_basico`, aproximando a empresa pela matriz, e o radar de baixa em 12 meses foi avaliado com split temporal absoluto: treino em 2022, validacao em 2023 e teste em 2024.

## Principais resultados

- 67.175.202 empresas analisadas e 31.342.895 eventos de baixa observados
- mediana Kaplan-Meier de 14,17 anos
- sobrevivencia estimada de 85,4% em 1 ano, 75,7% em 3 anos, 68,6% em 5 anos e 57,2% em 10 anos
- modelo final `logistic_full`, calibrado por isotonic, com ROC-AUC de 0,6975, PR-AUC de 0,1327 e lift de 2,53x no top 10%
- sinais mais fortes do radar concentrados em idade da empresa, enquadramento em Simples/MEI e coorte de abertura

## Achados centrais

O principal ganho do projeto veio menos da sofisticacao do algoritmo e mais do desenho metodologico. A camada de sobrevivencia mostra que coortes de abertura sao a leitura mais robusta da duracao empresarial, enquanto recortes por UF, porte, CNAE e natureza juridica devem ser lidos como estrutura observada do cadastro, nao como baseline historico perfeito. Na camada preditiva, o radar foi tratado como score operacional e nao como explicacao causal, com enfase em calibracao, estabilidade e controle de leakage.

## Resultado final

O projeto fecha como um estudo aplicado de alto volume com trilha auditavel: base consolidada, analise de sobrevivencia, score de risco 12M, figuras finais, artigo e materiais de comunicacao. Mais do que produzir uma metrica, ele mostra como transformar dado administrativo massivo em inferencia temporal defensavel e comunicacao tecnica clara.
