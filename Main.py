from com.ucdenver.ml.dataprocess.current_data import CurrentDataset
from com.ucdenver.ml.dataprocess.historical_data import HistoricalDataset
from com.ucdenver.ml.model.RandomForest import RandomForest
from com.ucdenver.ml.model.kNN import kNN
from com.ucdenver.ml.utils.divide_data import getTrainAndTest, getCurrentData

# #Download Historical Dataset
# hdata = HistoricalDataset()
#
# hdata.build_stock_dataset()
# hdata.build_sp500_dataset()
#
# #Build features from Historical Dataset
# sp500_df, stock_df = hdata.preprocess_price_data()
# hdata.parse_keystats(sp500_df, stock_df)
#

### Uncomment below section if prediction of latest data is required
# #Download current Dataset
# cdata = CurrentDataset()
# cdata.check_yahoo()
# cdata.forward()

##Load dataset
X_train, X_test, y_train, y_test, z_train, z_test = getTrainAndTest()
X_current, z_current = getCurrentData()

print("\n*************      RandomForest Model      *************\n")
rf = RandomForest(name="Random_Forest")
rf.train_model(X_train, y_train)
y_pred = rf.predict_stocks(X_test)
rf.getModelStats(y_test, z_test, y_pred)
rf.saveConfusionMatrix(y_test, y_pred)
y_pred = rf.predict_stocks(X_current, z_current)
print("\n*************      kNN Model      *************\n")
knn = kNN(name="kNN")
knn.train_model(X_train, y_train)
y_pred = knn.predict_stocks(X_test)
knn.getModelStats(y_test, z_test, y_pred)
knn.saveConfusionMatrix(y_test, y_pred)
y_pred = knn.predict_stocks(X_current, z_current)
