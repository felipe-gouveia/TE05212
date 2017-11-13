%% Redes Neurais Artificiais - Tarefa em Sala

%  Descrição
%  ------------
%
%  Este arquivo contém o código para solução do comando da tarefa02
%  utilizando Foward Propagation não vetorizada.
%  AUTOR: Felipe Rodrigues Veludo Gouveia
%  Data: 03-11-2017
%
%  Comando da Questão:
%  --------------------
%  - O código matlab ELM1 usa laços e deve ser completado (usando laços) de 
%  modo a produzir resultados equivalentes (mesmas variáveis) aos produzidos
%  pelo código ELM2. Este último utiliza representação matricial.
%  - Você pode chamar a rotina que faz pseudo-inversão, assim como o código 
%  ELM2 chama.
%  A atividade pode ser realizada em duplas (de dois).
%  Incluir no arquivo (script matlab) os nomes dos autores.

 % CONJUNTO DE TREINAMENTO

X = [0.35 -1.4;
     3.1 1.2;
     -2.5 2.4;
     1.7 0.8;
     -0.4 0.6;
     1.3 2.6];
 
 % NÚMERO DE FEATURES DA CAMADA DE ENTRADA (sem bias)
 M = size(X,2);
 
 % NÚMERO DE ENTRADAS DO TREINAMENTO
 N = size(X,1);
 
 % DESIGN MATRIX
 X = [X ones(N,1)];
 
 % CONJUNTO DE SAÍDAS
 D = [6.2 1.4;
      -3.5 1.8;
      5.1 2.3;
      2.6 1.3;
      0.5 -2.1;
      0.2 4.1];
 
 % #NEURÔNIOS SAÍDA
 C = size(D, 2);
 
 % #NEURÔNIOS OCULTA
 P = 3;
 
 % PESOS DA CAMADA DE ENTRADA
 W1 = [0.53 0.86 -0.43;
       1.83 0.31  0.34;
       -2.25 -1.3 3.57];
 W1 = W1';
   %randn(P,M+1);
 
 %PREALOCAÇÃO DE VARIÁVEIS PARA MELHOR DESEMPENHO
 S1 = zeros(N,P);
 H = zeros(N,P);
 S2 = zeros(N,C);
 Y = zeros(N,C);
 
% S1 e H
for k = 1:N % 1 até N, onde N é o número de neurônios na camada de entrada (número de linhas da matriz de entrada)
    
    for j = 1:P % 1 até P, onde P é o número de neurônios na camada oculta 
      s = 0; % inicializando a soma ponderada
      
      for i = 1:M+1 % 1 a M, onde M é o número de entradas externas
         s = s + X(k,i)*W1(i,j);
      end
      %s = s + W1(j, M+1); % adicionar o bias, que esta na posicao m+1
      
      S1(k, j) = s;
      H(k, j) = tanh(s);
          
    end   
end

% APÓS TERMOS O VALOR DA HIPÓTESE DA SAÍDA DA CAMADA OCULTA,
% CALCULAMOS OS PESOS

Ha = [H ones(N,1)];
%W2t = pinv(Ha)*D; % pinv(Ha)=inv(Ha'Ha)*Ha'
%W2 = W2t';
 W2 = [ -23.71 -17.85 -23.40 17.32;
        -16.79  -13.99 -19.96 13.02];
 W2 = W2';
 
% S2 e Y
for k = 1:N % 1 até N, onde N é o número de neurônios na camada de entrada (número de linhas da matriz de entrada)
    
    for r = 1:C % 1 até P, onde P é o número de neurônios na camada de saída 
       s2 = 0; % inicializando a soma ponderada 2

       for q = 1:size(H,2) + 1 % 1 a size(H,2), onde size(H,2) é o número de features da camada oculta (sem bias)
           s2 = s2 + Ha(k,q) * W2(q, r);
       end
       
       %s2 = s2 + W2(r, P+1); % acrescenta peso bias
       S2(k, r) = s2;
       Y(k, r) = s2; %funçãoo de ativação tipo identidade
       
    end
end

% SAÍDA
Y

% CALCULAMOS O ERRO MÉDIO QUADRÁTICO
E = D - Y
eqm = 1/(N*C)*E(:)'*E(:)