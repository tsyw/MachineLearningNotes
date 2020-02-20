# 总结

## Math

1.  MLE
    $$
    \theta_{MLE}=\mathop{argmax}\limits _{\theta}\log p(X|\theta)\mathop{=}\limits _{iid}\mathop{argmax}\limits _{\theta}\sum\limits _{i=1}^{N}\log p(x_{i}|\theta)
    $$
    

2.  MAP
    $$
    \theta_{MAP}=\mathop{argmax}\limits _{\theta}p(\theta|X)=\mathop{argmax}\limits _{\theta}p(X|\theta)\cdot p(\theta)
    $$

3.  Gaussian Distribution
    $$
    \begin{align}&p(x|\mu,\Sigma)=\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}\\
    &\Delta=(x-\mu)^{T}\Sigma^{-1}(x-\mu)=\sum\limits _{i=1}^{p}(x-\mu)^{T}u_{i}\frac{1}{\lambda_{i}}u_{i}^{T}(x-\mu)=\sum\limits _{i=1}^{p}\frac{y_{i}^{2}}{\lambda_{i}}
    \end{align}
    $$

4.  已知 $x\sim\mathcal{N}(\mu,\Sigma), y\sim Ax+b$，有：
    $$
    \begin{align}y\sim\mathcal{N}(A\mu+b, A\Sigma A^T)
    \end{align}
    $$

5.  记 $x=(x_1, x_2,\cdots,x_p)^T=(x_{a,m\times 1}, x_{b,n\times1})^T,\mu=(\mu_{a,m\times1}, \mu_{b,n\times1}),\Sigma=\begin{pmatrix}\Sigma_{aa}&\Sigma_{ab}\\\Sigma_{ba}&\Sigma_{bb}\end{pmatrix}$，已知 $x\sim\mathcal{N}(\mu,\Sigma)$，则：
    $$
    \begin{align}&x_a\sim\mathcal{N}(\mu_a,\Sigma_{aa})\\
    &x_b|x_a\sim\mathcal{N}(\mu_{b|a},\Sigma_{b|a})\\
    &\mu_{b|a}=\Sigma_{ba}\Sigma_{aa}^{-1}(x_a-\mu_a)+\mu_b\\
    &\Sigma_{b|a}=\Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}
    \end{align}
    $$

## Linear Regression

### Model

1.  Dataset: 
    $$
    \mathcal{D}=\{(x_1, y_1),(x_2, y_2),\cdots,(x_N, y_N)\}
    $$

2.  Notation:
    $$
    X=(x_1,x_2,\cdots,x_N)^T,Y=(y_1,y_2,\cdots,y_N)^T
    $$

3.  Model:
    $$
    f(w)=w^Tx
    $$

### Loss Function

1.  最小二乘误差/高斯噪声的MLE
    $$
    L(w)=\sum\limits_{i=1}^N||w^Tx_i-y_i||^2_2
    $$
    

### 闭式解

$$
\begin{align}\hat{w}=(X^TX)^{-1}X^TY=X^+Y\\
X=U\Sigma V^T\\
X^+=V\Sigma^{-1}U^T
\end{align}
$$

### 正则化

$$
\begin{align}
L1-Gaussian \ priori&:\mathop{argmin}\limits_wL(w)+\lambda||w||_1,\lambda\gt0\\
L2-Laplasian\ priori-Sparsity&:\mathop{argmin}\limits_wL(w)+\lambda||w||^2_2,\lambda \gt 0
\end{align}
$$

## Linear Classification

### Hard

#### PCA

1.  Idea: 在线性模型上加入激活函数

2.  Loss Function:

$$
L(w)=\sum\limits_{x_i\in\mathcal{D}_{wrong}}-y_iw^Tx_i
$$

3.  Parameters:

$$
w^{t+1}\leftarrow w^{t}+\lambda y_ix_i
$$

#### Fisher

1.  Idea: 投影，类内小，类间大。

2.  Loss Function:
    $$
    \begin{align}&J(w)=\frac{w^TS_bw}{w^TS_ww}\\
    &S_b=(\overline{x_{c1}}-\overline{x_{c2}})(\overline{x_{c1}}-\overline{x_{c2}})^T\\
    &S_w=S_1+S_2
    \end{align}
    $$

3.  闭式解，投影方向:
    $$
    S_w^{-1}(\overline{x_{c1}}-\overline{x_{c2}})
    $$
    

### Soft

#### 判别模型

##### Logistic Regression

1.  Idea，激活函数:
    $$
    \begin{align}p(C_1|x)&=\frac{1}{1+\exp(-a)}\\
    a&=w^Tx
    \end{align}
    $$

2.  Loss Function(交叉熵):
    $$
    \hat{w}=\mathop{argmax}_wJ(w)=\mathop{argmax}_w\sum\limits_{i=1}^N(y_i\log p_1+(1-y_i)\log p_0)
    $$

3.  解法，SGD
    $$
    J'(w)=\sum\limits_{i=1}^N(y_i-p_1)x_i
    $$

#### 生成模型

##### GDA

1.  Model

    1.  $y\sim Bernoulli(\phi)$
    2.  $x|y=1\sim\mathcal{N}(\mu_1,\Sigma)$
    3.  $x|y=0\sim\mathcal{N}(\mu_0,\Sigma)$

2.  MAP
    $$
    \begin{align}
    &\mathop{argmax}_{\phi,\mu_0,\mu_1,\Sigma}\log p(X|Y)p(Y)\nonumber\\
    &=\mathop{argmax}_{\phi,\mu_0,\mu_1,\Sigma}\sum\limits_{i=1}^N((1-y_i)\log\mathcal{N}(\mu_0,\Sigma)+y_i\log \mathcal{N}(\mu_1,\Sigma)+y_i\log\phi+(1-y_i)\log(1-\phi))
    \end{align}
    $$

3.  解
    $$
    \begin{align}\phi&=\frac{N_1}{N}\\
    \mu_1&=\frac{\sum\limits_{i=1}^Ny_ix_i}{N_1}\\
    \mu_0&=\frac{\sum\limits_{i=1}^N(1-y_i)x_i}{N_0}\\
    \Sigma&=\frac{N_1S_1+N_2S_2}{N}
    \end{align}
    $$

##### Naive Bayesian

1.  Model, 对单个数据点的各个维度作出限制
    $$
    x_i\perp x_j|y,\forall\  i\ne j
    $$

    1.  $x_i$ 为连续变量：$p(x_i|y)=\mathcal{N}(\mu_i,\sigma_i^2)$
    2.  $x_i$ 为离散变量：类别分布（Categorical）：$p(x_i=i|y)=\theta_i,\sum\limits_{i=1}^K\theta_i=1$
    3.  $p(y)=\phi^y(1-\phi)^{1-y}$

2.  解：和GDA相同

## Dimension Reduction

中心化：
$$
\begin{align}S
&=\frac{1}{N}X^T(E_N-\frac{1}{N}\mathbb{I}_{N1}\mathbb{I}_{1N})(E_N-\frac{1}{N}\mathbb{I}_{N1}\mathbb{I}_{1N})^TX\nonumber\\
&=\frac{1}{N}X^TH^2X=\frac{1}{N}X^THX
\end{align}
$$

### PCA

1.  Idea: 坐标变换，寻找线性无关的新基矢，取信息损失最小的前几个维度

2.  Loss Function:
    $$
    \begin{align}J
    &=\sum\limits_{j=1}^qu_j^TSu_j\ ,\ s.t.\ u_j^Tu_j=1
    \end{align}
    $$

3.  解：

    1.  特征分解法
        $$
        S=U\Lambda U^T
        $$

    2.  SVD for X/S
        $$
        \begin{align}HX=U\Sigma V^T\\
        S=\frac{1}{N}V\Sigma^T\Sigma V^T
        \\new\ co=HX\cdot V\end{align}
        $$

    3.  SVD for T
        $$
        \begin{align}T=HXX^TH=U\Sigma\Sigma^TU^T\\
        new\ co=U\Sigma
        \end{align}
        $$

### p-PCA

1.  Model:
    $$
    \begin{align}
    z&\sim\mathcal{N}(\mathbb{O}_{q1},\mathbb{I}_{qq})\\
    x&=Wz+\mu+\varepsilon\\
    \varepsilon&\sim\mathcal{N}(0,\sigma^2\mathbb{I}_{pp})
    \end{align}
    $$

2.  Learning: E-M

3.  Inference:
    $$
    p(z|x)=\mathcal{N}(W^T(WW^T+\sigma^2\mathbb{I})^{-1}(x-\mu),\mathbb{I}-W^T(WW^T+\sigma^2\mathbb{I})^{-1}W)
    $$

## SVM

1.  强对偶关系：凸优化+（松弛）Slater 条件->强对偶。
2.  参数求解：KKT条件
    1.  可行域
    2.  互补松弛+梯度为0

### Hard-margin

1.  Idea: 最大化间隔

2.  Model:
    $$
    \mathop{argmin}_{w,b}\frac{1}{2}w^Tw\ s.t.\ y_i(w^Tx_i+b)\ge1,i=1,2,\cdots,N
    $$

3.  对偶问题
    $$
    \max_{\lambda}-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j+\sum\limits_{i=1}^N\lambda_i,\ s.t.\ \lambda_i\ge0
    $$

4.  模型参数
    $$
    \hat{w}=\sum\limits_{i=1}^N\lambda_iy_ix_i\\
    \hat{b}=y_k-w^Tx_k=y_k-\sum\limits_{i=1}^N\lambda_iy_ix_i^Tx_k,\exist k,1-y_k(w^Tx_k+b)=0
    $$

### Soft-margin

1.  Idea:允许少量错误

2.  Model:
    $$
    error=\sum\limits_{i=1}^N\max\{0,1-y_i(w^Tx_i+b)\}\\
    \mathop{argmin}_{w,b}\frac{1}{2}w^Tw+C\sum\limits_{i=1}^N\xi_i\ s.t.\ y_i(w^Tx_i+b)\ge1-\xi_i,\xi_i\ge0,i=1,2,\cdots,N
    $$

### Kernel

对称的正定函数都可以作为正定核。

## Exp Family

1.  表达式
    $$
    p(x|\eta)=h(x)\exp(\eta^T\phi(x)-A(\eta))=\frac{1}{\exp(A(\eta))}h(x)\exp(\eta^T\phi(x))
    $$

2.  对数配分函数
    $$
    \begin{align} 
    A'(\eta)=\mathbb{E}_{p(x|\eta)}[\phi(x)]\\
    A''(\eta)=Var_{p(x|\eta)}[\phi(x)]
    \end{align}
    $$

3.  指数族分布满足最大熵定理

## PGM

### Representation

1.  有向图
    $$
    p(x_1,x_2,\cdots,x_p)=\prod\limits_{i=1}^pp(x_i|x_{parent(i)})
    $$
    D-separation
    $$
    p(x_i|x_{-i})=\frac{p(x)}{\int p(x)dx_{i}}=\frac{\prod\limits_{j=1}^pp(x_j|x_{parents(j)})}{\int\prod\limits_{j=1}^pp(x_j|x_{parents(j)})dx_i}=\frac{p(x_i|x_{parents(i)})p(x_{child(i)}|x_i)}{\int p(x_i|x_{parents(i)})p(x_{child(i)}|x_i)dx_i}
    $$
    

2.  无向图
    $$
    \begin{align}p(x)=\frac{1}{Z}\prod\limits_{i=1}^{K}\phi(x_{ci})\\
    Z=\sum\limits_{x\in\mathcal{X}}\prod\limits_{i=1}^{K}\phi(x_{ci})\\
    \phi(x_{ci})=\exp(-E(x_{ci}))
    \end{align}
    $$

3.  有向转无向

    1.  将每个节点的父节点两两相连
    2.  将有向边替换为无向边

### Learning

参数学习-EM

1.  目的：解决具有隐变量的混合模型的参数估计（极大似然估计）

2.  参数：
    $$
    \theta_{MLE}=\mathop{argmax}\limits_\theta\log p(x|\theta)
    $$
    

3.  迭代求解：
    $$
    \theta^{t+1}=\mathop{argmax}\limits_{\theta}\int_z\log [p(x,z|\theta)]p(z|x,\theta^t)dz=\mathbb{E}_{z|x,\theta^t}[\log p(x,z|\theta)]
    $$

4.  原理
    $$
    \log p(x|\theta^t)\le\log p(x|\theta^{t+1})
    $$

5.  广义EM

    1.  E step：
        $$
        \hat{q}^{t+1}(z)=\mathop{argmax}_q\int_zq^t(z)\log\frac{p(x,z|\theta)}{q^t(z)}dz,fixed\ \theta
        $$

    2.  M step：
        $$
        \hat{\theta}=\mathop{argmax}_\theta \int_zq^{t+1}(z)\log\frac{p(x,z|\theta)}{q^{t+1}(z)}dz,fixed\ \hat{q}
        $$


### Inference

1.  精确推断

    1.  VE

    2.  BP
        $$
        m_{j\to i}(i)=\sum\limits_j\phi_j(j)\phi_{ij}(ij)\prod\limits_{k\in Neighbour(j)-i}m_{k\to j}(j)
        $$

    3.  MP
        $$
        m_{j\to i}=\max\limits_{j}\phi_j\phi_{ij}\prod\limits_{k\in Neighbour(j)-i}m_{k\to j}
        $$

2.  近似推断

    1.  确定性近似，VI

        1.  变分表达式
            $$
            \hat{q}(Z)=\mathop{argmax}_{q(Z)}L(q)
            $$

        2.  平均场近似下的 VI-坐标上升
            $$
            \mathbb{E}_{\prod\limits_{i\ne j}q_i(Z_i)}[\log p(X,Z)]=\log \hat{p}(X,Z_j)\\
            q_j(Z_j)=\hat{p}(X,Z_j)
            $$

        3.  SGVI-变成优化问题，重参数法
            $$
            \mathop{argmax}_{q(Z)}L(q)=\mathop{argmax}_{\phi}L(\phi)\\
            \nabla_\phi L(\phi)=\mathbb{E}_{q_\phi}[(\nabla_\phi\log q_\phi)(\log p_\theta(x^i,z)-\log q_\phi(z))]\\
            =\mathbb{E}_{p(\varepsilon)}[\nabla_z[\log p_\theta(x^i,z)-\log q_\phi(z)]\nabla_\phi g_\phi(\varepsilon,x^i)]\\
            z=g_\phi(\varepsilon,x^i),\varepsilon\sim p(\varepsilon)
            $$

    2.  随机性近似

        1.  蒙特卡洛方法采样

            1.  CDF 采样

            2.  拒绝采样， $q(z)$，使得 $\forall z_i,Mq(z_i)\ge p(z_i)$，拒绝因子：$\alpha=\frac{p(z^i)}{Mq(z^i)}\le1$

            3.  重要性采样
                $$
                \mathbb{E}_{p(z)}[f(z)]=\int p(z)f(z)dz=\int \frac{p(z)}{q(z)}f(z)q(z)dz\simeq\frac{1}{N}\sum\limits_{i=1}^Nf(z_i)\frac{p(z_i)}{q(z_i)}
                $$

            4.  重要性重采样：重要性采样+重采样

        2.  MCMC：构建马尔可夫链概率序列，使其收敛到平稳分布 $p(z)$。

            1.  转移矩阵（提议分布）
                $$
                p(z)\cdot Q_{z\to z^*}\alpha(z,z^*)=p(z^*)\cdot Q_{z^*\to z}\alpha(z^*,z)\\
                \alpha(z,z^*)=\min\{1,\frac{p(z^*)Q_{z^*\to z}}{p(z)Q_{z\to z^*}}\}
                $$

            2.  算法（MH）：

                1.  通过在0，1之间均匀分布取点 $u$
                2.  生成 $z^*\sim Q(z^*|z^{i-1})$
                3.  计算 $\alpha$ 值
                4.  如果 $\alpha\ge u$，则 $z^i=z^*$，否则 $z^{i}=z^{i-1}$

        3.  Gibbs 采样：给定初始值 $z_1^0,z_2^0,\cdots$在 $t+1$ 时刻，采样 $z_i^{t+1}\sim p(z_i|z_{-i})$，从第一个维度一个个采样。

## GMM

1.  Model
    $$
    p(x)=\sum\limits_{k=1}^Kp_k\mathcal{N}(x|\mu_k,\Sigma_k)
    $$

2.  求解-EM
    $$
    \begin{align}Q(\theta,\theta^t)&=\sum\limits_z[\log\prod\limits_{i=1}^Np(x_i,z_i|\theta)]\prod \limits_{i=1}^Np(z_i|x_i,\theta^t)\nonumber\\
    &=\sum\limits_z[\sum\limits_{i=1}^N\log p(x_i,z_i|\theta)]\prod \limits_{i=1}^Np(z_i|x_i,\theta^t)\nonumber\\
    &=\sum\limits_{i=1}^N\sum\limits_{z_i}\log p(x_i,z_i|\theta)p(z_i|x_i,\theta^t)\nonumber\\
    &=\sum\limits_{i=1}^N\sum\limits_{z_i}\log p_{z_i}\mathcal{N(x_i|\mu_{z_i},\Sigma_{z_i})}\frac{p_{z_i}^t\mathcal{N}(x_i|\mu_{z_i}^t,\Sigma_{z_i}^t)}{\sum\limits_kp_k^t\mathcal{N}(x_i|\mu_k^t,\Sigma_k^t)}
    \end{align}
    $$

    $$
    p_k^{t+1}=\frac{1}{N}\sum\limits_{i=1}^Np(z_i=k|x_i,\theta^t)
    $$

## 序列模型-HMM，LDS，Particle

1.  假设：

    1.  齐次 Markov 假设（未来只依赖于当前）：
        $$
        p(i_{t+1}|i_t,i_{t-1},\cdots,i_1,o_t,o_{t-1},\cdots,o_1)=p(i_{t+1}|i_t)
        $$

    2.  观测独立假设：
        $$
        p(o_t|i_t,i_{t-1},\cdots,i_1,o_{t-1},\cdots,o_1)=p(o_t|i_t)
        $$

2.  参数
    $$
    \lambda=(\pi,A,B)
    $$
    

### 离散线性隐变量-HMM

1.  Evaluation：$p(O|\lambda)$，Forward-Backward 算法
    $$
    p(O|\lambda)=\sum\limits_{i=1}^Np(O,i_T=q_i|\lambda)=\sum\limits_{i=1}^N\alpha_T(i)=\sum\limits_{i=1}^Nb_i(o_1)\pi_i\beta_1(i)\\
    \alpha_{t+1}(j)=\sum\limits_{i=1}^Nb_{j}(o_t)a_{ij}\alpha_t(i)\\
    \beta_t(i)=\sum\limits_{j=1}^Nb_j(o_{t+1})a_{ij}\beta_{t+1}(j)
    $$

2.  Learning：$\lambda=\mathop{argmax}\limits_{\lambda}p(O|\lambda)$，EM 算法（Baum-Welch）
    $$
    \lambda^{t+1}=\mathop{argmax}_\lambda\sum\limits_I\log p(O,I|\lambda)p(O,I|\lambda^t)\\=\sum\limits_I[\log \pi_{i_1}+\sum\limits_{t=2}^T\log a_{i_{t-1},i_t}+\sum\limits_{t=1}^T\log b_{i_t}(o_t)]p(O,I|\lambda^t)
    $$

3.  Decoding：$I=\mathop{argmax}\limits_{I}p(I|O,\lambda)$，Viterbi 算法-动态规划
    $$
    \delta_{t}(j)=\max\limits_{i_1,\cdots,i_{t-1}}p(o_1,\cdots,o_t,i_1,\cdots,i_{t-1},i_t=q_i)\\\delta_{t+1}(j)=\max\limits_{1\le i\le N}\delta_t(i)a_{ij}b_j(o_{t+1})\\\psi_{t+1}(j)=\mathop{argmax}\limits_{1\le i\le N}\delta_t(i)a_{ij}
    $$

### 连续线性隐变量-LDS

1.  Model
    $$
    \begin{align}
    p(z_t|z_{t-1})&\sim\mathcal{N}(A\cdot z_{t-1}+B,Q)\\
    p(x_t|z_t)&\sim\mathcal{N}(C\cdot z_t+D,R)\\
    z_1&\sim\mathcal{N}(\mu_1,\Sigma_1)
    \end{align}
    $$

2.  滤波
    $$
    p(z_t|x_{1:t})=p(x_{1:t},z_t)/p(x_{1:t})\propto p(x_{1:t},z_t)\\=p(x_t|z_t)p(z_t|x_{1:t-1})p(x_{1:t-1})\propto p(x_t|z_t)p(z_t|x_{1:t-1})
    $$

3.  递推求解-线性高斯模型

    1.  Prediction
        $$
        p(z_t|x_{1:t-1})=\int_{z_{t-1}}p(z_t|z_{t-1})p(z_{t-1}|x_{1:t-1})dz_{t-1}=\int_{z_{t-1}}\mathcal{N}(Az_{t-1}+B,Q)\mathcal{N}(\mu_{t-1},\Sigma_{t-1})dz_{t-1}
        $$

    2.  Update:
        $$
        p(z_t|x_{1:t})\propto p(x_t|z_t)p(z_t|x_{1:t-1}
        $$

### 连续非线性隐变量-粒子滤波

通过采样(SIR)解决：
$$
\mathbb{E}[f(z)]=\int_zf(z)p(z)dz=\int_zf(z)\frac{p(z)}{q(z)}q(z)dz=\sum\limits_{i=1}^Nf(z_i)\frac{p(z_i)}{q(z_i)}
$$

1.  采样
    $$
    w_t^i\propto\frac{p(x_t|z_t)p(z_t|z_{t-1})}{q(z_t|z_{1:t-1},x_{1:t})}w_{t-1}^i\\
    q(z_t|z_{1:t-1},x_{1:t})=p(z_t|z_{t-1})
    $$

2.  重采样

## CRF

1.  PDF
    $$
    p(Y=y|X=x)=\frac{1}{Z(x,\theta)}\exp[\theta^TH(y_t,y_{t-1},x)]
    $$

2.  边缘概率
    $$
    p(y_t=i|x)=\sum\limits_{y_{1:t-1}}\sum\limits_{y_{t+1:T}}\frac{1}{Z}\prod\limits_{t'=1}^T\phi_{t'}(y_{t'-1},y_{t'},x)\\
    p(y_t=i|x)=\frac{1}{Z}\Delta_l\Delta_r\\
    \Delta_l=\sum\limits_{y_{1:t-1}}\phi_{1}(y_0,y_1,x)\phi_2(y_1,y_2,x)\cdots\phi_{t-1}(y_{t-2},y_{t-1},x)\phi_t(y_{t-1},y_t=i,x)\\
    \Delta_r=\sum\limits_{y_{t+1:T}}\phi_{t+1}(y_t=i,y_{t+1},x)\phi_{t+2}(y_{t+1},y_{t+2},x)\cdots\phi_T(y_{T-1},y_T,x)
    $$

    $$
    \alpha_t(i)=\Delta_l=\sum\limits_{j\in S}\phi_t(y_{t-1}=j,y_t=i,x)\alpha_{t-1}(j)\\
    \Delta_r=\beta_t(i)=\sum\limits_{j\in S}\phi_{t+1}(y_t=i,y_{t+1}=j,x)\beta_{t+1}(j)
    $$

3.  学习
    $$
    \nabla_\lambda L=\sum\limits_{i=1}^N\sum\limits_{t=1}^T[f(y_{t-1},y_t,x^i)-\sum\limits_{y_{t-1}}\sum\limits_{y_t}p(y_{t-1},y_t|x^i)f(y_{t-1},y_t,x^i)]
    $$
    

