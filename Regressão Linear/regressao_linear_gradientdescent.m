%% Redes Neurais Artificiais - Tarefa 01 utilizando Gradient Descent

%  Descrição
%  ------------
%
%  Este arquivo contém o código para solução do comando da tarefa01
%  utilizando o método de Regressão Linear com Gradient Descent.
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

X = [0
     1/4
     1/2
     3/4
     1
     5/4
     3/2];

y = [0
     1/16
     1/4
     9/16
     1
     25/16
     9/4]

m = length(y); % Número de exemplos do conjunto de treinamento    
X1 = [ones(m, 1), X]; % % Adicionando bias à entrada de primeiro grau
X2 = [ones(m, 1), X.^2, X]; %  Adicionando bias à entrada de segundo grau
n1 = size(X1, 2); % inicializando os pesos de acordo com o número de features
n2 = size(X2, 2); % inicializando os pesos de acordo com o número de features
theta1 = zeros(n1, 1); % inicializando os coeficientes
theta2 = zeros(n2, 1); % inicializando os coeficientes
 
% Iterações e Valor de Alpha
iterations = 1500;
alpha = 0.01;
 
% Calcula o custo inicial para X1 e X2
J1 = computeCost(X1, y, theta1); %function described below
J2 = computeCost(X2, y, theta2); %function described below

theta1 = gradientDescent(X1, y, theta1, alpha, iterations); % theta contains value of theta0 and theta1
theta2 = gradientDescent(X2, y, theta2, alpha, iterations); % theta contains value of theta0 and theta1

% Calcula o custo inicial para X1 e X2
J1 = computeCost(X1, y, theta1); %function described below
J2 = computeCost(X2, y, theta2); %function described below

%predict1 = [1, 3.5] *theta; %predict for new data

% D) Construa um gráfico obtendo h versus training e y versus training
figure(1)
plot(X,X1*theta1,'g',X,X2*theta2,'b-X',X,y,'r--o')
%hold on
%plot(X,X1*theta1,'b-',X,y,'r-')
%hold on
%plot(X,X2*theta2,'bX',X,y,'rX')
%hold on
%plot(X,X2*theta2,'b-',X,y,'r-')
%hold off
xlabel('x')
ylabel('resposta')
title('valor desejado: o e resposta modelo: .')