%% Redes Neurais Artificiais - Tarefa 02

%  Descri��o
%  ------------
%
%  Este arquivo cont�m o c�digo para solu��o do comando da tarefa02
%  utilizando Foward Propagation n�o vetorizada.
%  AUTOR: Felipe Rodrigues Veludo Gouveia
%  Data: 03-11-2017
%
%  Comando da Quest�o:
%  --------------------
%  - O c�digo matlab ELM1 usa la�os e deve ser completado (usando la�os) de 
%  modo a produzir resultados equivalentes (mesmas vari�veis) aos produzidos
%  pelo c�digo ELM2. Este �ltimo utiliza representa��o matricial.
%  - Voc� pode chamar a rotina que faz pseudo-invers�o, assim como o c�digo 
%  ELM2 chama.
%  A atividade pode ser realizada em duplas (de dois).
%  Incluir no arquivo (script matlab) os nomes dos autores.

 % CONJUNTO DE TREINAMENTO

X = [0 0;
     0 1;
     1 0;
     1 1];
 
 % N�MERO DE FEATURES DA CAMADA DE ENTRADA (sem bias)
 M = size(X,2);
 
 % N�MERO DE ENTRADAS DO TREINAMENTO
 N = size(X,1);
 
 % DESIGN MATRIX
 X = [X ones(N,1)];
 
 % CONJUNTO DE SA�DAS
 D = [0
      1
      1
      1];
 
 % #NEUR�NIOS SA�DA
 C = size(D, 2);
 
 % #NEUR�NIOS OCULTA
 P = 5;
 
 % PESOS DA CAMADA DE ENTRADA
 W1 = randn(P,M+1);
 
 %PREALOCA��O DE VARI�VEIS PARA MELHOR DESEMPENHO
 S1 = zeros(N,P);
 H = zeros(N,P);
 S2 = zeros(N,C);
 Y = zeros(N,C);
 
% S1 e H
for k = 1:N % 1 at� N, onde N � o n�mero de neur�nios na camada de entrada (n�mero de linhas da matriz de entrada)
    
    for j = 1:P % 1 at� P, onde P � o n�mero de neur�nios na camada oculta 
      s = 0; % inicializando a soma ponderada
      
      for i = 1:M % 1 a M, onde M � o n�mero de entradas externas
         s = s + W1(j,i)*X(k,i);
      end
      s = s + W1(j, M+1); % adicionar o bias, que esta na posicao m+1
      
      S1(k, j) = s;
      H(k, j) = tanh(s);
          
    end   
end

% AP�S TERMOS O VALOR DA HIP�TESE DA SA�DA DA CAMADA OCULTA,
% CALCULAMOS OS PESOS

Ha = [H ones(N,1)];
W2t = pinv(Ha)*D; % pinv(Ha)=inv(Ha'Ha)*Ha'
W2 = W2t';

% S2 e Y
for k = 1:N % 1 at� N, onde N � o n�mero de neur�nios na camada de entrada (n�mero de linhas da matriz de entrada)
    
    for r = 1:C % 1 at� P, onde P � o n�mero de neur�nios na camada de sa�da 
       s2 = 0; % inicializando a soma ponderada 2

       for q = 1:size(H,2) % 1 a size(H,2), onde size(H,2) � o n�mero de features da camada oculta (sem bias)
           s2 = s2 + W2(r,q)*Ha(k,q);
       end
       
       s2 = s2 + W2(r, P+1); % acrescenta peso bias
       S2(k, r) = s2;
       Y(k, r) = s2; %fun��oo de ativa��o tipo identidade
       
    end
end

% SA�DA
Y

% CALCULAMOS O ERRO M�DIO QUADR�TICO
E = D - Y
eqm = 1/(N*C)*E(:)'*E(:)