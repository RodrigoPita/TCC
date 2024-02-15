# TCC - Motando um Reconhecedor de Acordes

_Aluno: Rodrigo Pita_ <br>
_Orientador: Hugo Tremonte_

O objetivo deste projeto é programar um reconhecedor de acordes complexos, focado em músicas como Jazz, Bossa Nova e MPB, usando como ponto de partida os módulos de reconhecimento de acorde do livro _Fundamentals of Music Processing_ do Meinard Müller. 

## Testes Inicias
A primeira etapa do processo será apenas dedicada para entender os funcionamentos dos módulos do livro e descobrir sobre as limitações de cada algoritmo (Template-Based ou HMM-Based). Para isso serão feitos testes usando músicas com características diferentes, seja por tipo de instrumento, ruído, complexidade de acordes ou forma como os acordes se apresentam. 

### Primeira bateria de testes
As músicas usadas para se familiarizar com o código foram:
1. Wonderwall - Oasis
2. Someone You Loved - Lewis Capaldi

Delas foram usados os segundos iniciais, apenas instrumental, para ajudar o reconhecedor. Apenas com essas duas músicas já foi possível notar uma diferença de comportamento, onde a primeira, além de ser mais ruidosa, é tocada primariamente no violão e apresenta notas contantes independentes do acorde tocado, enquanto a segunda é exclusivamente tocada num piano, com um som captado de forma mais limpa. 

### Segunda bateria de testes
Mantidas as duas músicas da primeira bateria. O foco dos testes nesse momento foi de não só comparar as músicas, mas também cada versão de cromagrama gerado (STFT, CQT, IIR) juntamente com os procedimentos de reconhecimento de acordes (Template, HMM). Os resultados mostraram que para a primeira música (Wonderwall), a melhor combinação de parâmetros foi a do cromagrama baseado em IIR com o procedimento HMM, já para a segunda (Someone You Loved), a melhor combinação se deu pelo cromagrama baseado em CQT com o procedimento HMM. 

Se baseando apenas nestes testes, pode-se afirmar que o procedimento HMM se destaca em relação ao Template, como era de se esperar, devido a sua complexidade maior. Quanto aos cromagramas, pode-se concluir que eles dependem a perfomance de cada um dependerá das características particulares da música, seja em relação ao ruído, instrumento, forma de tocar, entre outras possibilidades.

### Terceira bateria de testes
Agora a música analisada foi:
- Carol Pita - Um Cafuné E Um Violão

Essa música apresenta uma progressão de acordes não usual, além de ter uma mudança de comportamento da primeira para a segunda parte, por esse motivo, foi dividida em duas faixas para fins de comparação:
1. Parte 1: Apresenta um estilo mais _staccato_
2. Parte 2: Altera para um harpejo, tendendo ao _legato_

Os testes mostram novamente uma performance superior do cromagrama baseado em CQT e que as formas de tocar praticamente não afetaram o resultado do reconhecimento. 

### Quarta bateria de testes
Para esta leva de testes, foram analisadas sete novas músicas:
1. Franz Ferdinand - Take Me Out
2. Stevie Ray Vaughan - Pride And Joy
3. Elton John - Your Song
4. Billie Eilish - everything i wanted
5. Paul McCartney - Maybe I'm Amazed
6. Chico Buarque - Teresinha
7. The Animals - House Of The Rising Sun

Essas músicas servem para ajudar no julgamento dos limites do algoritmo sem que as matrizes de transição e probabilidade de emissão sejam alteradas. 

### Quinta bateria de testes
Mais uma leva de sete músicas analisada:
1. Daniel Caesar & H.E.R. - Best Part
2. Israel IZ Kamakawiwo ole - Somewhere over the Rainbow
3. Shawn Mendes - Stitches
4. Charlie Puth - Attention
5. Bruno Mars - When I Was Your Man
6. Maroon 5 - Sunday Morning
7. Coldplay - The Scientist

### Sexta bateria de testes
Última leva inicial de testes, com o intuito de extrair imagens de espectrogramas, cromagras e matrizes de transição para enriquecer o projeto.

## Testes MPB

### Mudança de matriz de transição
A primeira grande alteração para os testes em músicas da MPB é a alteração da Matriz de Transição. A matriz inicial era uma matriz uniforme e foi alterada para uma matriz tirada gerada a partir do Projeto MPB, resultado de uma análise de 550 músicas. 

Os testes dessa etapa visam comparar os resultados da matriz uniforme com a matriz alterada e ver o comportamento do reconhecimento com as seguintes músicas da MPB (não usadas na geração da matriz de transição):

1. Caetano Veloso - Comeu
2. Caetano Veloso - Deixa Sangrar
3. Chico Buarque - Ate Segunda-Feira
4. Chico Buarque - Juca
5. Ivan Lins - Atras Poeira
6. Ivan Lins - Comecar De Novo
7. Tom Jobim - As Praias Desertas
8. Tom Jobim - Aula de Matematica
 
 
 
 
 
 