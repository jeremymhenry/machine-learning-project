import abc

import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report


class Model(abc.ABC):
    model = None

    def __init__(self, name, OUTPERFORMANCE=10):
        self.define_model()
        self.OUTPERFORMANCE = OUTPERFORMANCE
        self.name = name

    @abc.abstractmethod
    def define_model(self):
        pass

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_stocks(self, X_test, z=None):
        y_pred = self.model.predict(X_test)

        if sum(y_pred) == 0:
            print("No stocks predicted!")
        elif z is not None:
            invest_list = z[y_pred].tolist()
            print(
                f"{len(invest_list)} stocks predicted to outperform the S&P500 by more than {self.OUTPERFORMANCE}%:"
            )
            print(" ".join(invest_list))
        return y_pred

    def saveConfusionMatrix(self, y_test, y_pred):
        cm = confusion_matrix(y_target=y_test, y_predicted=y_pred, binary=False)
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        ax.set_title('RandomForest Confusion Matrix')
        plt.savefig('images/' + self.name + '_Confusion_Matrix.png')
        plt.show()
        plt.close()

    def getModelStats(self, y_test, z_test, y_pred):
        c_report = classification_report(y_test, y_pred)
        print(c_report)

        # Because y_pred is an array of 1s and 0s, the number of positive predictions
        # is equal to the sum of the array
        num_positive_predictions = sum(y_pred)
        if num_positive_predictions < 0:
            print("No stocks predicted!")

        # Recall that z_test stores the change in stock price in column 0, and the
        # change in S&P500 price in column 1.
        # Whenever a stock is predicted to outperform (y_pred = 1), we 'buy' that stock
        # and simultaneously `buy` the index for comparison.
        stock_returns = 1 + z_test[y_pred, 0] / 100
        market_returns = 1 + z_test[y_pred, 1] / 100

        # Calculate the average growth for each stock we predicted 'buy'
        # and the corresponding index growth
        avg_predicted_stock_growth = sum(stock_returns) / num_positive_predictions
        index_growth = sum(market_returns) / num_positive_predictions
        percentage_stock_returns = 100 * (avg_predicted_stock_growth - 1)
        percentage_market_returns = 100 * (index_growth - 1)
        total_outperformance = percentage_stock_returns - percentage_market_returns

        print("\n Stock prediction performance report \n", "=" * 40)
        print(f"Total Trades:", num_positive_predictions)
        print(f"Average return for stock predictions: {percentage_stock_returns: .1f} %")
        print(
            f"Average market return in the same period: {percentage_market_returns: .1f}% "
        )
        print(
            f"Compared to the index, our strategy earns {total_outperformance: .1f} percentage points more"
        )
