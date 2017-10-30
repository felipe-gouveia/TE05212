%% Redes Neurais Artificiais - Tarefa 01 utilizando Gradient Descent

%  Descri��o
%  ------------
%
%  Este arquivo cont�m o c�digo para solu��o do comando da tarefa01
%  utilizando o m�todo de Regress�o Linear com Gradient Descent.
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

m = length(y); % N�mero de exemplos do conjunto de treinamento    
X1 = [ones(m, 1), X]; % % Adicionando bias � entrada de primeiro grau
X2 = [ones(m, 1), X.^2, X]; %  Adicionando bias � entrada de segundo grau
n1 = size(X1, 2); % inicializando os pesos de acordo com o n�mero de features
n2 = size(X2, 2); % inicializando os pesos de acordo com o n�mero de features
theta1 = zeros(n1, 1); % inicializando os coeficientes
theta2 = zeros(n2, 1); % inicializando os coeficientes
 
% Itera��es e Valor de Alpha
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

% D) Construa um gr�fico obtendo h versus training e y versus training
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