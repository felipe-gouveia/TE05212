%% Redes Neurais Artificiais - Tarefa 01 utilizando Eq. Normal

%  Descri��o
%  ------------
%
%  Este arquivo cont�m o c�digo para solu��o do comando da tarefa01
%  utilizando o m�todo de Regress�o Linear com Eq. Normal.
%  Autor: Felipe Rodrigues Veludo Gouveia
%  Data: 29-10-2017
%
%  Comando da Quest�o:
%  --------------------
%     Fazer o ajuste dos coeficientes do modelo y = a*x^2 + b*x + c, de
%     modo a minimizar o erro quadr�tico m�dio.
%     Material a ser entregue:
%      - Gr�fico contendo: Sa�da x Design Matrix, Sa�da do Modelo de
%      Regress�o Linear de 1a Ordem x Design Matrix, Sa�da do Modelo de
%      Regress�o Linear de 2a Ordem x Design Matrix.
%      - C�digo utilizado para ajuste dos coeficientes;
%      - Breve coment�rio sobre os resultados obtidos em uma p�gina A4
%      (vers�o eletr�nica em PDF). Neste documento inclua os valores dos
%      coeficientes obtidos nos dois casos.
%

% Conjunto de Treinamento dado na quest�o
training = [0
     1/4
     1/2
     3/4
     1
     5/4
     3/2];

% Conjunto de Sa�das dado na quest�o
y = [0
     1/16
     1/4
     9/16
     1
     25/16
     9/4]
 
% Definimos 'm' como o n�mero de sa�das
m = length(y);

% Definimos nossas Design Matrix:
% 1) X1 = Theta0 + Theta1 * treinamento
% 2) X2 = Theta0 + Theta1 * treinamento + Theta2 * treinamento^2
% Onde, theta0 � o BIAS, que � representado como uma coluna de 1 do tamanho
% do conjunto de sa�das (tamb�m � o mesmo tamanho do conjunto de entrada)
X1 = [ones(m,1) training]
X2 = [ones(m,1) training.^2 training ]


% Determinamos os valores das matrizes theta
% para as design matrix de primeira e de segunda ordem
A = X'*X;
theta_1aordem = pinv(X1'*X1)*X1'*y;
theta_2aordem = pinv(X2'*X2)*X2'*y

  
% Determinamos as hypothesis de 1a e 2a ordem
% A hip�tese determina os coeficientes da nossa rede para que com novas
% entradas sejamos capazes de prever sa�das que sejam consistentes com os
% valores do conjunto de treinamento x sa�da.
hypothesis_1aordem = X1 * theta1;
hypothesis_2aordem = X2 * theta2;

% Plotamos o gr�fico
% 1o plot = Design Matrix (bias e training) x Hip�tese de 1a Ordem
% 2o plot = Design Matrix (bias e training) x Hip�tese de 2a Ordem
% 3o plot = Design Matrix (bias e training) x Conjunto de Sa�da
figure(1)
plot(X,X1*theta1,'g',X,X2*theta2,'b.',X,y,'r-o')
xlabel('x')
ylabel('resposta')
title('valor desejado: o e resposta modelo: .')

% E) Calcule o erro quadr�tico
  
%sqrError = (h - y).^2;
%J = 1/(2*m) * sum(sqrError)
%vete = h - y;
%eqm = 1/nl*(vete'*vete);
%rmse = sqrt(eqm)
  