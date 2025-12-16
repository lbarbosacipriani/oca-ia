### Como executar

- Permitir execucao do arquivo setup.sh:
        
        chmod +777 setup.sh
    
- Executar arquivo setup.sh:

        ./setup.sh


    ### setup.sh:
        
        - Cria o ambiente virtual .incor_env

        - Instala pacotes arquivo requirements.txt

        - Executa script main.py. 

    
   ### main.py:

    Variaveis:
    
    - folds = 10
    - epochs = 30
    - BATCH_SIZE = 5
    - flg_salvar_modelos = True

            - Realiza a leitura do csv com a estrutura "path", "label". Transformando em tensor.
                O arquivo csv deve ser referenciado na variável FILE_PATH. 

            - Processa a imagem para 3x256x256 com remocao de background 

            - Cria e executa loop de treinamento e validacao para modelo. 

            - Salva modelos caso flg true. 

            - Salva metricas em path output/metricas (cria automaticamente).

