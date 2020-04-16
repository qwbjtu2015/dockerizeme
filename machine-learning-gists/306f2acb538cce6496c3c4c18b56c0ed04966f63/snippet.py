import numpy as np
from sklearn.cluster import KMeans


class Cluster:

    def __init__(self, ids: [str], features: [[float]], centroid: []):
        self.ids = np.array(ids)
        self.features = np.array(features)
        self.centroid = centroid

        assert self.ids.ndim == 1 and self.features.ndim == 2 and self.centroid.ndim == 1   # 次元
        assert self.features[0].size == self.centroid.size    # 各データの要素数は同じ
        assert self.ids.size == self.features.shape[0]        # idと特徴量は同じサイズ

    def log_likelihood(self) -> float:
        num_of_data = self.features.shape[0]
        num_of_attributes = self.features.shape[1]

        if num_of_data <= 1:
            return 0

        variance = self.get_distortion() / (num_of_data - 1.0)
        p1 = (num_of_data / 2.0) * np.log(np.pi * 2.0)
        p2 = float("inf") if variance == 0 else num_of_data * num_of_attributes / 2 * np.log(variance)
        p3 = (num_of_data - 1.0) / 2.0
        p4 = num_of_data * np.log(num_of_data)

        return -p1 - p2 - p3 + p4

    def get_distortion(self) -> float:
        return np.linalg.norm(self.features - self.centroid, axis=1).sum()


class Xmeans:

    def __init__(self, min_clusters: int=2, max_clusters: int=10, max_iter: int=300, n_init: int=10):
        assert min_clusters < max_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.labels_ = None

    def split_centroid(self, cluster: Cluster) -> list:
        vector = np.random.uniform(0.0, 1.0, cluster.features.shape[1])
        variance = cluster.get_distortion() / cluster.features.shape[0]
        length = variance ** 0.5
        vector *= length / np.linalg.norm(vector)

        return np.array([cluster.centroid + vector, cluster.centroid - vector])

    def split_cluster(self, cluster: Cluster) -> (Cluster, Cluster):
        """
        k-meansでクラスターを2分割する
        """
        if cluster.features.shape[0] <= 2:
            return None

        centroid_list = self.split_centroid(cluster)

        model = KMeans(max_iter=self.max_iter, n_init=self.n_init, init=centroid_list).fit(cluster.features)
        labels = model.labels_

        # 2分割できないようなデータだった(同じデータが複数あるときとか)
        if len(set(labels)) != 2:
            return None

        assert labels[(labels != 0) & (labels != 1)].size == 0  # ラベルは0か1だけ

        cluster_list = []
        for label in set(labels):
            l1, f1 = cluster.ids[labels == label], cluster.features[labels == label]
            cluster_list.append(Cluster(l1, f1, model.cluster_centers_[label]))

        return cluster_list

    def kmeans(self, clusters: [Cluster]) ->list:
        all_ids, all_features, all_centroids = [], [], []
        for cluster in clusters:
            all_ids += list(cluster.ids)
            all_features += list(cluster.features)
            all_centroids += list([cluster.centroid])

        all_ids, all_features, all_centroids = map(np.array, [all_ids, all_features, all_centroids])
        model = KMeans(max_iter=self.max_iter, n_init=self.n_init, init=all_centroids).fit(all_features)
        labels = model.labels_

        clusters = []
        for label in set(labels):
            l, f = all_ids[labels == label], all_features[labels == label]
            clusters.append(Cluster(l, f, model.cluster_centers_[label]))

        return clusters

    def bic(self, clusters: [Cluster]) -> float:
        num_of_clusters = len(clusters)
        num_of_dimensions = clusters[0].features.shape[1]
        num_of_parameters = (num_of_clusters - 1) + num_of_clusters * num_of_dimensions + num_of_clusters

        num_of_data = 0
        log_likelihood = 0
        for cluster in clusters:
            log_likelihood += cluster.log_likelihood()
            num_of_data += cluster.features.shape[0]

        return log_likelihood - (num_of_data * np.log(num_of_data)) - (num_of_parameters / 2.0 * np.log(num_of_data))

    def xmeans(self, clusters: [Cluster]) -> list:

        if len(clusters) >= self.max_iter:
            return clusters

        # Improve-Params
        clusters = self.kmeans(clusters)

        # Improve-Structure
        bic_clusters = {}
        for parent in clusters:
            children = self.split_cluster(parent)

            # 2分割できなかった
            if not children:
                bic_clusters[float('inf')] = [parent]
                continue

            old_bic = self.bic([parent])
            new_bic = self.bic(children)

            if old_bic < new_bic:
                bic_clusters[new_bic] = children
            else:
                bic_clusters[float('inf')] = [parent]

        # BICの高い順に追加
        num = self.max_clusters - len(clusters)
        new_cluster_list = []
        for k, v in sorted(bic_clusters.items())[:num]:
            new_cluster_list += v

        # クラスタが増えてない
        if len(new_cluster_list) == len(clusters):
            return clusters

        new_cluster_list = self.kmeans(new_cluster_list)

        # 全体のBICを比較
        old_bic = self.bic(clusters)
        new_bic = self.bic(new_cluster_list)

        if old_bic < new_bic:
            return new_cluster_list
        else:
            return clusters

    def fit(self, features: [[float]]) -> 'Xmeans':
        ids = np.array([i for i in range(len(features))])
        features = np.array(features)

        # self.min_clustersに分割
        model = KMeans(n_clusters=self.min_clusters, max_iter=self.max_iter, n_init=self.n_init).fit(features)
        cluster_list = []
        for label in set(model.labels_):
            l1, f1 = ids[model.labels_ == label], features[model.labels_ == label]
            cluster_list.append(Cluster(l1, f1, model.cluster_centers_[label]))

        # self.max_clustersになるか，それ以上分割できなくなるまで分割していく
        while True:
            new_cluster_list = self.xmeans(cluster_list)
            if len(new_cluster_list) == len(cluster_list):
                break
            cluster_list = new_cluster_list

        no = 0
        self.labels_ = [0] * len(ids)
        for cluster in cluster_list:
            for id_ in cluster.ids:
                self.labels_[id_] = no
            no += 1

        return self


def main():
    features = []

    # https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
    with open("iris.data") as f:
        for line in f:
            features.append(list(map(float, line.split(",")[:-1])))

    model = Xmeans(min_clusters=1, max_clusters=100).fit(features)
    print(len(set(model.labels_)))

    
if __name__ == '__main__':
    main()
