%% Redes Neurais Artificiais - Tarefa 01 utilizando Eq. Normal

%  Descrição
%  ------------
%
%  Este arquivo contém o código para solução do comando da tarefa01
%  utilizando o método de Regressão Linear com Eq. Normal.
%  Autor: Felipe Rodrigues Veludo Gouveia
%  Data: 29-10-2017
%
%  Comando da Questão:
%  --------------------
%     Fazer o ajuste dos coeficientes do modelo y = a*x^2 + b*x + c, de
%     modo a minimizar o erro quadrático médio.
%     Material a ser entregue:
%      - Gráfico contendo: Saída x Design Matrix, Saída do Modelo de
%      Regressão Linear de 1a Ordem x Design Matrix, Saída do Modelo de
%      Regressão Linear de 2a Ordem x Design Matrix.
%      - Código utilizado para ajuste dos coeficientes;
%      - Breve comentário sobre os resultados obtidos em uma página A4
%      (versão eletrônica em PDF). Neste documento inclua os valores dos
%      coeficientes obtidos nos dois casos.
%

% Conjunto de Treinamento dado na questão
training = [0
     1/4
     1/2
     3/4
     1
     5/4
     3/2];

% Conjunto de Saídas dado na questão
y = [0
     1/16
     1/4
     9/16
     1
     25/16
     9/4]
 
% Definimos 'm' como o número de saídas
m = length(y);

% Definimos nossas Design Matrix:
% 1) X1 = Theta0 + Theta1 * treinamento
% 2) X2 = Theta0 + Theta1 * treinamento + Theta2 * treinamento^2
% Onde, theta0 é o BIAS, que é representado como uma coluna de 1 do tamanho
% do conjunto de saídas (também é o mesmo tamanho do conjunto de entrada)
X1 = [ones(m,1) training]
X2 = [ones(m,1) training.^2 training ]


% Determinamos os valores das matrizes theta
% para as design matrix de primeira e de segunda ordem
A = X'*X;
theta_1aordem = pinv(X1'*X1)*X1'*y;
theta_2aordem = pinv(X2'*X2)*X2'*y

  
% Determinamos as hypothesis de 1a e 2a ordem
% A hipótese determina os coeficientes da nossa rede para que com novas
% entradas sejamos capazes de prever saídas que sejam consistentes com os
% valores do conjunto de treinamento x saída.
hypothesis_1aordem = X1 * theta1;
hypothesis_2aordem = X2 * theta2;

% Plotamos o gráfico
% 1o plot = Design Matrix (bias e training) x Hipótese de 1a Ordem
% 2o plot = Design Matrix (bias e training) x Hipótese de 2a Ordem
% 3o plot = Design Matrix (bias e training) x Conjunto de Saída
figure(1)
plot(X,X1*theta1,'g',X,X2*theta2,'b.',X,y,'r-o')
xlabel('x')
ylabel('resposta')
title('valor desejado: o e resposta modelo: .')

% E) Calcule o erro quadrático
  
%sqrError = (h - y).^2;
%J = 1/(2*m) * sum(sqrError)
%vete = h - y;
%eqm = 1/nl*(vete'*vete);
%rmse = sqrt(eqm)
  