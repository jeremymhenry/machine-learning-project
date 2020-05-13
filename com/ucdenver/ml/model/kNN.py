from sklearn.neighbors import KNeighborsClassifier

from com.ucdenver.ml.model import Model


class kNN(Model.Model):

    def define_model(self):
        self.model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
