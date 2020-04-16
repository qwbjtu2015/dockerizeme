import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score

## 1.データの読み込み
def load_data(filename):
    data = loadmat(filename)
    return np.array(data['X']), np.array(data['Xval']), np.ravel(np.array(data['yval']))
X, Xval, yval = load_data("ex8data1.mat")
# データのプロット
def plot_data():
    plt.plot(X[:,0], X[:, 1], "bx")
    plt.xlabel("Latencey (ms)")
    plt.ylabel("Throughput (mb/s)")
plot_data()
plt.show()

## 2.データの統計量を推定
def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)
    return mu, sigma2
mu, sigma2 = estimate_gaussian(X)
# 多変量正規分布の確率密度関数を計算
# 分散のベクトルを共分散行列に変形（対角要素＝分散とする）
cov_matrix = np.diag(sigma2)
# （自分で定義してもいいが、scipy.stats.multivariate_normalを使うと楽）
p = multivariate_normal.pdf(X, mean=mu, cov=cov_matrix)
# 可視化
def visualize_fit(X, mu, sigma2):
    plot_data()
    X1, X2 = np.meshgrid(np.arange(0, 35, 0.5), np.arange(0, 35, 0.5))
    Z = multivariate_normal.pdf(np.c_[np.ravel(X1), np.ravel(X2)], mean=mu, cov=np.diag(sigma2))
    Z = Z.reshape(X1.shape)
    if not np.isinf(np.sum(p)):
        plt.contour(X1, X2, Z, levels=10**np.arange(-20, 0, 3, dtype="float"))
visualize_fit(X, mu, sigma2)
plt.show()

## 3.外れ値を探す
# 交差検証データに対するpvalを計算
pval = multivariate_normal.pdf(Xval, mean=mu, cov=cov_matrix)
# しきい値を選択
def select_threshold(yval, pval):
    best_epsilon, best_f1 = 0, 0
    steps = np.linspace(np.min(pval), np.max(pval), 1000)
    for epsilon in steps:
        pred_positive = pval < epsilon
        f1 = f1_score(yval, pred_positive)
        if f1 > best_f1:
            best_f1, best_epsilon = f1, epsilon
    return best_epsilon, best_f1
# εはハイパーパラメーターなので交差検証データに対してフィットさせる
epsilon, f1 = select_threshold(yval, pval)
print("Best epsilon found using cross-validation:", epsilon)
print("Best F1 on Cross Validation Set:", f1)
print("   (you should see a value epsilon of about 8.99e-05)")
print("   (you should see a Best F1 value of  0.875000)\n")
# 外れ値を探す
outliers = p < epsilon
# 外れ値のプロット
visualize_fit(X, mu, sigma2)
plt.plot(X[outliers, 0], X[outliers, 1], "ro", markerfacecolor="none", linewidth=2, markersize=10)
plt.show()

## 4.多次元の外れ値
# データの読み込み
X, Xval, yval = load_data("ex8data2.mat")
# 統計量の推定
mu, sigma2 = estimate_gaussian(X)
# 訓練データ
p = multivariate_normal.pdf(X, mean=mu, cov=np.diag(sigma2))
# 交差検証データ
pval = multivariate_normal.pdf(Xval, mean=mu, cov=np.diag(sigma2))
# しきい値の選択
epsilon, f1 = select_threshold(yval, pval)
print("Best epsilon found using cross-validation: ", epsilon)
print("Best F1 on Cross Validation Set:  ", f1)
print("   (you should see a value epsilon of about 1.38e-18)")
print("   (you should see a Best F1 value of 0.615385)")
print("# Outliers found:", np.sum(p < epsilon))