from sklearn.ensemble import RandomForestClassifier

from com.ucdenver.ml.model import Model


class RandomForest(Model.Model):

    def define_model(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=0)
