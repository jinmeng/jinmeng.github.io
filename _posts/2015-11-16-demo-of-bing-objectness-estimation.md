---
layout: post
title: A Demo of BING(objectness estimation)
---

{% include JB/setup %}

## Abstract

本文的代码是我读论文《BING: Binarized Normed Gradients for Objectness Estimation at 300fps》的时候，用matlab实现的一个简单的脚本，用来演示BING的基本思想。
个人的理解，本文的主要贡献是提出一种高效的方法来计算两个向量的内积。
--- 

## Code
论文中的用到了几个参数
* d为64，这里设为4
* Nw=2, Ng = 4

```
d = 4; 

Nw = 8; \%Nw <= 2^d
Ng = 8; \%Ng<=8, for the reason that each pixel is 8 bits in [0,255]

\%set w and b for explanation 
\%generate a random double vector 
w0 = rand(d,1);
ind = randi(100, d,1);
w = w0 .* (-1) .^ ind; 
\%w = [ 0.2316,   -0.4889,   -0.6241,  0.6791]';
disp('w = ');
disp(w');

\%compute the binarized approximation of w
[beta, a, dim] = binariedVector(w, Nw);

\%print all beta and vecA 
for ii = 1 : dim
	fprintf('beta(\%d) = \%f, \na{\%d} = ', ii, beta(ii), ii);
	\%disp(beta(ii));
	disp(a{ii}');
end

\%generate a random gradient feature g
mm = randi([0, 255],  sqrt(d));
[fx, fy] = gradient(mm);
grad = min( abs(fx)+abs(fy), 255);
g = grad(:);
\%g = [128, 255, 78,  214]';
disp('g = ');
disp(g');

\%compute binarized representation g_l = sum_{k=1}^Ng (2^{8-k} * b_kl
gl =  dec2bin(g, Ng); 
disp(gl);
gl = gl - '0';

\%define an anonymous function for the inner product of vx and vy 
\%vx is like [1, -1, -1, 1],  vy is like [1, 0, 1, 0]
\%set vxPlus = (vx == 1), then vx = vxPlus - ~vxPlus
\%and dot(vx, vy) = 2*dot(vxPlus, vy) - sum(vy) 
dotBinVec = @(vx, vy) 2 * dot(double(vx == 1), vy)  - sum(vy);

\%compute the score of g, i.e., the inner product of binarized w and g 
sl = 0;
for  jj = 1 : dim
	tmp = 0;
	for kk = 1: Ng
		tmp = tmp + 2^(8-kk) * dotBinVec(a{jj}, gl(:,kk));
	end
    sl = sl + beta(jj) * tmp; 
end
dotProd = dot(w, g);
disp(sl);
disp(dotProd);

```

需要调用的函数

```
\% Algorithm 1 in BING: Binaried Normed Gradients for Objectness Estimation at 300 fps
\% input a vector w and a given parameter Nw,  output the BinarizedVector of w

function [beta, vecA, dim] = binariedVector(w, Nw)
\%Input: 
\%       w, a float vector of dimension d
\%       Nw, the number of basis used for approximating w
\%Output: 
\%       {beta_j}   a set of coefficient vector of dimension d
\%       {a_j}, a set of basis vector of dimension d,  in the form of [1, -1, ...]
\% Formula:          w is approximately equal to Sum_j{ beta_j * a_j }


    \%1. generate the cartesian product of [1, -1] of dimension d
    d = length(w);  \%say w = [2, -4, 6], then d = 3
    \%dim = min(Nw, 2^d);  \%the maximal size of vecA
	dim = min(Nw, 2^d);
    \%2 compute binaried vector
    epsilon = w;
    
    for jj = 1 : dim
        vecA{jj} = sign(epsilon);
        beta(jj) = sum(vecA{jj} .* epsilon) / sum(vecA{jj}.^2);
        epsilon = epsilon - beta(jj)*vecA{jj};
		if max(abs(epsilon)) < 1e-10
			dim = jj;
            break;
        end
    end

end

```


下面的代码用来验证子函数的正确性。

```
\% verify the correctness of BinarizedVector
 sigma = zeros(d, 1); 
 for ii=1:dim
     sigma = sigma+ beta(ii)*a{ii}; 
 end
 \%for ii = 1:d
     \%fprintf('w(\%d) = \%f,   sum(\%d) = \%f\n', ii, w(ii), ii,  sigma(ii));
 \%    fprintf('w(\%d)-sum(\%d) = \%f\n', ii, ii,  w(ii) - sigma(ii));
 \%end
 fprintf('w - sum = \n');
 disp(w - sigma);

 verify the correctness of innerProd: innerProd(vx, vy) ?= sum(vx .* vy)
 dd = 8; 
 for ii = 1: 10
 	aa = (-1).^(randi([-1, 1], 1, dd));
 	bb = randi([0, 1], 1, dd);
 	disp(aa);
 	disp(bb);
 	t1 = dotBinVec(aa, bb);
 	t2 = dot(aa, bb);
 	fprintf('dotBinVec vs. dot: %d  %d\n', t1, t2);
 end

```






