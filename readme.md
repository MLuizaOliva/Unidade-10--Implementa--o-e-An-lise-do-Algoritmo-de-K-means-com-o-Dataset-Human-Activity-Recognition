# Unidade 10 - Implementacao e Analise do Algoritmo de K means com o Dataset Human Activity Recognition

## Objetivo
Este projeto implementa um sistema de reconhecimento de atividades humanas utilizando dados de sensores de smartphones através do algoritmo K-means. O sistema analisa dados de acelerômetro e giroscópio para identificar e agrupar seis diferentes atividades físicas: caminhada, subida de escadas, descida de escadas, sentar, ficar em pé e deitar.

## Pré-requisitos
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Instalação
1. Clone este repositório
```bash
git clone https://github.com/MLuizaOliva/Unidade_10__Implementacao_e_Analise_do_Algoritmo_de_K_means_com_o_Dataset_Human_Activity_Recognition.git
cd Unidade_10__Implementacao_e_Analise_do_Algoritmo_de_K_means_com_o_Dataset_Human_Activity_Recognition
```


2. Instale as dependências
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Download do Dataset

- Faça o download do [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- Extraia o arquivo ZIP
- Mova a pasta "UCI HAR Dataset" para o diretório raiz do projeto

## Estrutura do Projeto

```
Unidade_10__Implementacao_e_Analise_do_Algoritmo_de_K_means_com_o_Dataset_Human_Activity_Recognition/
│
├── main.py                    
├── docs/
│   ├── Relatório.pdf          
├── README.md                  
└── UCI HAR Dataset/           # Dataset (deve ser baixado separadamente)
    ├── train/
    │   ├── X_train.txt
    │   └── y_train.txt
    └── features.txt
```

## Como executar

```
python main.py
```


## Principais Resultados
- Redução efetiva da dimensionalidade dos dados através de PCA, mantendo aproximadamente 85% da variância com apenas 3 componentes
- Identificação de 5 clusters principais através do método do cotovelo
- Alta precisão na identificação de atividades estáticas (sentar, em pé, deitar)
- Distinção moderada entre atividades dinâmicas (caminhada, subir/descer escadas)

## Considerações
- O algoritmo K-means mostrou-se eficaz na separação entre atividades estáticas e dinâmicas
- Atividades dinâmicas similares apresentam alguma sobreposição nos clusters
- A visualização 3D dos clusters permite melhor compreensão da separação entre atividades
- Potencial para melhorias através da incorporação de características temporais dos dados


