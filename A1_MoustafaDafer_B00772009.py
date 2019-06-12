####################
#   ML for BD A1   #
#  Moustafa Dafer  #
#    B00 772009    #
####################

###Initialization & Imports###
import pandas as pandas
import numpy as numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ttest_ind


###read dataset###
dataset = pandas.read_csv('dataset_original.csv')
#print dataset.columns
#dataset = dataset.values



###Binarize Target & Replace it in Dataset###
dataset = dataset.assign(targetElk=[1 if dataset.iloc[i]["class"] == "ELK" else 0 for i in range(len(dataset.values))])
dataset = dataset.assign(targetCattle=[1 if dataset.iloc[i]["class"] == "CATTLE" else 0 for i in range(len(dataset.values))])
dataset = dataset.assign(targetDeer=[1 if dataset.iloc[i]["class"] == "DEER" else 0 for i in range(len(dataset.values))])



###split dataset###
train = dataset.sample(frac=0.7, random_state=1)
test = dataset.loc[~dataset.index.isin(train.index)]
print('')
print("train: "+str(train.shape))
print("test: "+str(test.shape))
print('')
print('')


### Correlations and Filter columns###
#correlations = dataset.corr()["class"]
#print correlations

columns = dataset.columns.tolist()
columns = [c for c in columns if c not in ["class","targetElk","targetCattle","targetDeer"]]



###Train classifiers###
LRmodelElk = LogisticRegression()
LRmodelElk.fit(train[columns], train["targetElk"])
LRmodelCattle = LogisticRegression()
LRmodelCattle.fit(train[columns], train["targetCattle"])
LRmodelDeer = LogisticRegression()
LRmodelDeer.fit(train[columns], train["targetDeer"])

DTmodelElk = tree.DecisionTreeClassifier()
DTmodelElk.fit(train[columns], train["targetElk"])
DTmodelCattle = tree.DecisionTreeClassifier()
DTmodelCattle.fit(train[columns], train["targetCattle"])
DTmodelDeer = tree.DecisionTreeClassifier()
DTmodelDeer.fit(train[columns], train["targetDeer"])

RFRmodelElk = RandomForestClassifier()
RFRmodelElk.fit(train[columns], train["targetElk"])
RFRmodelCattle = RandomForestClassifier()
RFRmodelCattle.fit(train[columns], train["targetCattle"])
RFRmodelDeer = RandomForestClassifier()
RFRmodelDeer.fit(train[columns], train["targetDeer"])

GNBmodelElk = GaussianNB()
GNBmodelElk.fit(train[columns], train["targetElk"])
GNBmodelCattle = GaussianNB()
GNBmodelCattle.fit(train[columns], train["targetCattle"])
GNBmodelDeer = GaussianNB()
GNBmodelDeer.fit(train[columns], train["targetDeer"])



###Prediction and errors### 
#DTpredictions = DTmodel.predict(test[columns])
#DTscore = accuracy_score(DTpredictions, test[target])
#print DTscore

LRpredictionsElk = LRmodelElk.predict(test[columns])
LRpredictionsCattle = LRmodelCattle.predict(test[columns])
LRpredictionsDeer = LRmodelDeer.predict(test[columns])

DTpredictionsElk = DTmodelElk.predict(test[columns])
DTpredictionsCattle = DTmodelCattle.predict(test[columns])
DTpredictionsDeer = DTmodelDeer.predict(test[columns])

RFRpredictionsElk = RFRmodelElk.predict(test[columns])
RFRpredictionsCattle = RFRmodelCattle.predict(test[columns])
RFRpredictionsDeer = RFRmodelDeer.predict(test[columns])

GNBpredictionsElk = GNBmodelElk.predict(test[columns])
GNBpredictionsCattle = GNBmodelCattle.predict(test[columns])
GNBpredictionsDeer = GNBmodelDeer.predict(test[columns])


RFRscoreElk = accuracy_score(RFRpredictionsElk, test["targetElk"])
RFRscoreCattle = accuracy_score(RFRpredictionsCattle, test["targetCattle"])
RFRscoreDeer = accuracy_score(RFRpredictionsDeer, test["targetDeer"])

RFRcmatrixElk = confusion_matrix(RFRpredictionsElk, test["targetElk"])
RFRcmatrixCattle = confusion_matrix(RFRpredictionsCattle, test["targetCattle"])
RFRcmatrixDeer = confusion_matrix(RFRpredictionsDeer, test["targetDeer"])

LRscoreElk = accuracy_score(LRpredictionsElk, test["targetElk"])
LRscoreCattle = accuracy_score(LRpredictionsCattle, test["targetCattle"])
LRscoreDeer = accuracy_score(LRpredictionsDeer, test["targetDeer"])

LRcmatrixElk = confusion_matrix(LRpredictionsElk, test["targetElk"])
LRcmatrixCattle = confusion_matrix(LRpredictionsCattle, test["targetCattle"])
LRcmatrixDeer = confusion_matrix(LRpredictionsDeer, test["targetDeer"])

DTscoreElk = accuracy_score(DTpredictionsElk, test["targetElk"])
DTscoreCattle = accuracy_score(DTpredictionsCattle, test["targetCattle"])
DTscoreDeer = accuracy_score(DTpredictionsDeer, test["targetDeer"])

DTcmatrixElk = confusion_matrix(DTpredictionsElk, test["targetElk"])
DTcmatrixCattle = confusion_matrix(DTpredictionsCattle, test["targetCattle"])
DTcmatrixDeer = confusion_matrix(DTpredictionsDeer, test["targetDeer"])

GNBscoreElk = accuracy_score(GNBpredictionsElk, test["targetElk"])
GNBscoreCattle = accuracy_score(GNBpredictionsCattle, test["targetCattle"])
GNBscoreDeer = accuracy_score(GNBpredictionsDeer, test["targetDeer"])

GNBcmatrixElk = confusion_matrix(GNBpredictionsElk, test["targetElk"])
GNBcmatrixCattle = confusion_matrix(GNBpredictionsCattle, test["targetCattle"])
GNBcmatrixDeer = confusion_matrix(GNBpredictionsDeer, test["targetDeer"])

###Prediction and errors of Training set###
RFRTrainpredictionsElk = RFRmodelElk.predict(train[columns])
RFRTrainpredictionsCattle = RFRmodelCattle.predict(train[columns])
RFRTrainpredictionsDeer = RFRmodelDeer.predict(train[columns])

LRTrainpredictionsElk = LRmodelElk.predict(train[columns])
LRTrainpredictionsCattle = LRmodelCattle.predict(train[columns])
LRTrainpredictionsDeer = LRmodelDeer.predict(train[columns])

DTTrainpredictionsElk = DTmodelElk.predict(train[columns])
DTTrainpredictionsCattle = DTmodelCattle.predict(train[columns])
DTTrainpredictionsDeer = DTmodelDeer.predict(train[columns])

GNBTrainpredictionsElk = GNBmodelElk.predict(train[columns])
GNBTrainpredictionsCattle = GNBmodelCattle.predict(train[columns])
GNBTrainpredictionsDeer = GNBmodelDeer.predict(train[columns])


RFRTrainscoreElk = accuracy_score(RFRTrainpredictionsElk, train["targetElk"])
RFRTrainscoreCattle = accuracy_score(RFRTrainpredictionsCattle, train["targetCattle"])
RFRTrainscoreDeer = accuracy_score(RFRTrainpredictionsDeer, train["targetDeer"])

LRTrainscoreElk = accuracy_score(LRTrainpredictionsElk, train["targetElk"])
LRTrainscoreCattle = accuracy_score(LRTrainpredictionsCattle, train["targetCattle"])
LRTrainscoreDeer = accuracy_score(LRTrainpredictionsDeer, train["targetDeer"])

DTTrainscoreElk = accuracy_score(DTTrainpredictionsElk, train["targetElk"])
DTTrainscoreCattle = accuracy_score(DTTrainpredictionsCattle, train["targetCattle"])
DTTrainscoreDeer = accuracy_score(DTTrainpredictionsDeer, train["targetDeer"])

GNBTrainscoreElk = accuracy_score(GNBTrainpredictionsElk, train["targetElk"])
GNBTrainscoreCattle = accuracy_score(GNBTrainpredictionsCattle, train["targetCattle"])
GNBTrainscoreDeer = accuracy_score(GNBTrainpredictionsDeer, train["targetDeer"])

RFRTraincmatrixElk = confusion_matrix(RFRTrainpredictionsElk, train["targetElk"])
RFRTraincmatrixCattle = confusion_matrix(RFRTrainpredictionsCattle, train["targetCattle"])
RFRTraincmatrixDeer = confusion_matrix(RFRTrainpredictionsDeer, train["targetDeer"])

LRTraincmatrixElk = confusion_matrix(LRTrainpredictionsElk, train["targetElk"])
LRTraincmatrixCattle = confusion_matrix(LRTrainpredictionsCattle, train["targetCattle"])
LRTraincmatrixDeer = confusion_matrix(LRTrainpredictionsDeer, train["targetDeer"])

DTTraincmatrixElk = confusion_matrix(DTTrainpredictionsElk, train["targetElk"])
DTTraincmatrixCattle = confusion_matrix(DTTrainpredictionsCattle, train["targetCattle"])
DTTraincmatrixDeer = confusion_matrix(DTTrainpredictionsDeer, train["targetDeer"])

GNBTraincmatrixElk = confusion_matrix(GNBTrainpredictionsElk, train["targetElk"])
GNBTraincmatrixCattle = confusion_matrix(GNBTrainpredictionsCattle, train["targetCattle"])
GNBTraincmatrixDeer = confusion_matrix(GNBTrainpredictionsDeer, train["targetDeer"])


###Reset Classifiers before cross validation###
LRmodelElk = LogisticRegression()
n_estimators=10
LRmodelCattle = LogisticRegression()
LRmodelDeer = LogisticRegression()

DTmodelElk = tree.DecisionTreeClassifier()
DTmodelCattle = tree.DecisionTreeClassifier()
DTmodelDeer = tree.DecisionTreeClassifier()

RFRmodelElk = RandomForestClassifier()
RFRmodelCattle = RandomForestClassifier()
RFRmodelDeer = RandomForestClassifier()

GNBmodelElk = GaussianNB()
GNBmodelCattle = GaussianNB()
GNBmodelDeer = GaussianNB()



###Cross Validation Scores###
LRcvsScoreElk = cross_val_score(LRmodelElk, train[columns], train["targetElk"], cv=10)
LRcvsScoreCattle = cross_val_score(LRmodelCattle, train[columns], train["targetCattle"], cv=10)
LRcvsScoreDeer = cross_val_score(LRmodelDeer, train[columns], train["targetDeer"], cv=10)

DTcvsScoreElk = cross_val_score(DTmodelElk, train[columns], train["targetElk"], cv=10)
DTcvsScoreCattle = cross_val_score(DTmodelCattle, train[columns], train["targetCattle"], cv=10)
DTcvsScoreDeer = cross_val_score(DTmodelDeer, train[columns], train["targetDeer"], cv=10)

RFRcvsScoreElk = cross_val_score(RFRmodelElk, train[columns], train["targetElk"], cv=10)
RFRcvsScoreCattle = cross_val_score(RFRmodelCattle, train[columns], train["targetCattle"], cv=10)
RFRcvsScoreDeer = cross_val_score(RFRmodelDeer, train[columns], train["targetDeer"], cv=10)

GNBcvsScoreElk = cross_val_score(GNBmodelElk, train[columns], train["targetElk"], cv=10)
GNBcvsScoreCattle = cross_val_score(GNBmodelCattle, train[columns], train["targetCattle"], cv=10)
GNBcvsScoreDeer = cross_val_score(GNBmodelDeer, train[columns], train["targetDeer"], cv=10)



###Mean and Std###
RFRcvsElkMean = numpy.mean(RFRcvsScoreElk)
RFRcvsElkStd = numpy.std(RFRcvsScoreElk)
RFRcvsCattleMean = numpy.mean(RFRcvsScoreCattle)
RFRcvsCattleStd = numpy.std(RFRcvsScoreCattle)
RFRcvsDeerMean = numpy.mean(RFRcvsScoreDeer)
RFRcvsDeerStd = numpy.std(RFRcvsScoreDeer)

LRcvsElkMean = numpy.mean(LRcvsScoreElk)
LRcvsElkStd = numpy.std(LRcvsScoreElk)
LRcvsCattleMean = numpy.mean(LRcvsScoreCattle)
LRcvsCattleStd = numpy.std(LRcvsScoreCattle)
LRcvsDeerMean = numpy.mean(LRcvsScoreDeer)
LRcvsDeerStd = numpy.std(LRcvsScoreDeer)

DTcvsElkMean = numpy.mean(DTcvsScoreElk)
DTcvsElkStd = numpy.std(DTcvsScoreElk)
DTcvsCattleMean = numpy.mean(DTcvsScoreCattle)
DTcvsCattleStd = numpy.std(DTcvsScoreCattle)
DTcvsDeerMean = numpy.mean(DTcvsScoreDeer)
DTcvsDeerStd = numpy.std(DTcvsScoreDeer)

GNBcvsElkMean = numpy.mean(GNBcvsScoreElk)
GNBcvsElkStd = numpy.std(GNBcvsScoreElk)
GNBcvsCattleMean = numpy.mean(GNBcvsScoreCattle)
GNBcvsCattleStd = numpy.std(GNBcvsScoreCattle)
GNBcvsDeerMean = numpy.mean(GNBcvsScoreDeer)
GNBcvsDeerStd = numpy.std(GNBcvsScoreDeer)

RFRAvg = numpy.average([RFRscoreElk,RFRscoreCattle,RFRscoreDeer])
DTAvg = numpy.average([DTscoreElk,DTscoreCattle,DTscoreDeer])
LRAvg = numpy.average([LRscoreElk,LRscoreCattle,LRscoreDeer])
GNBAvg = numpy.average([GNBscoreElk,GNBscoreCattle,GNBscoreDeer])

RFRcmatrixMean = numpy.mean([RFRcmatrixElk, RFRcmatrixCattle, RFRcmatrixDeer], axis=0)
RFRTraincmatrixMean = numpy.mean([RFRTraincmatrixElk, RFRTraincmatrixCattle, RFRTraincmatrixDeer], axis=0)
DTcmatrixMean = numpy.mean([DTcmatrixElk, DTcmatrixCattle, DTcmatrixDeer], axis=0)
LRcmatrixMean = numpy.mean([LRcmatrixElk, LRcmatrixCattle, LRcmatrixDeer], axis=0)
GNBcmatrixMean = numpy.mean([GNBcmatrixElk, GNBcmatrixCattle, GNBcmatrixDeer], axis=0)



###Concat classifier types of cross validation scores###
concatRFR = numpy.concatenate([RFRcvsScoreElk, RFRcvsScoreCattle, RFRcvsScoreDeer]);
concatDT = numpy.concatenate([DTcvsScoreElk, DTcvsScoreCattle, DTcvsScoreDeer]);
concatLR = numpy.concatenate([LRcvsScoreElk, LRcvsScoreCattle, LRcvsScoreDeer]);
concatGNB = numpy.concatenate([GNBcvsScoreElk, GNBcvsScoreCattle, GNBcvsScoreDeer]);



###T-test between most accurate classifier (RFR) and others###
statistic, RFRvsDTtTest = ttest_ind(concatRFR, concatDT)
statistic, RFRvsLRtTest = ttest_ind(concatRFR, concatLR)
statistic, RFRvsGNBtTest =  ttest_ind(concatRFR, concatGNB)



###New random forests with variable n-estimators###
RFR10modelElk = RandomForestClassifier(n_estimators=10)
RFR10modelCattle = RandomForestClassifier(n_estimators=10)
RFR10modelDeer = RandomForestClassifier(n_estimators=10)

RFR20modelElk = RandomForestClassifier(n_estimators=20)
RFR20modelCattle = RandomForestClassifier(n_estimators=20)
RFR20modelDeer = RandomForestClassifier(n_estimators=20)

RFR50modelElk = RandomForestClassifier(n_estimators=50)
RFR50modelCattle = RandomForestClassifier(n_estimators=50)
RFR50modelDeer = RandomForestClassifier(n_estimators=50)

RFR100modelElk = RandomForestClassifier(n_estimators=100)
RFR100modelCattle = RandomForestClassifier(n_estimators=100)
RFR100modelDeer = RandomForestClassifier(n_estimators=100)

###Cross Validation Scores###
RFR10cvsScoreElk = cross_val_score(RFR10modelElk, train[columns], train["targetElk"], cv=10)
RFR10cvsScoreCattle = cross_val_score(RFR10modelCattle, train[columns], train["targetCattle"], cv=10)
RFR10cvsScoreDeer = cross_val_score(RFR10modelDeer, train[columns], train["targetDeer"], cv=10)

RFR20cvsScoreElk = cross_val_score(RFR20modelElk, train[columns], train["targetElk"], cv=10)
RFR20cvsScoreCattle = cross_val_score(RFR20modelCattle, train[columns], train["targetCattle"], cv=10)
RFR20cvsScoreDeer = cross_val_score(RFR20modelDeer, train[columns], train["targetDeer"], cv=10)

RFR50cvsScoreElk = cross_val_score(RFR50modelElk, train[columns], train["targetElk"], cv=10)
RFR50cvsScoreCattle = cross_val_score(RFR50modelCattle, train[columns], train["targetCattle"], cv=10)
RFR50cvsScoreDeer = cross_val_score(RFR50modelDeer, train[columns], train["targetDeer"], cv=10)

RFR100cvsScoreElk = cross_val_score(RFR100modelElk, train[columns], train["targetElk"], cv=10)
RFR100cvsScoreCattle = cross_val_score(RFR100modelCattle, train[columns], train["targetCattle"], cv=10)
RFR100cvsScoreDeer = cross_val_score(RFR100modelDeer, train[columns], train["targetDeer"], cv=10)



###Mean and Avg###
RFR10cvsElkMean = numpy.mean(RFR10cvsScoreElk)
RFR10cvsCattleMean = numpy.mean(RFR10cvsScoreCattle)
RFR10cvsDeerMean = numpy.mean(RFR10cvsScoreDeer)
RFR10Avg = numpy.average([RFR10cvsElkMean, RFR10cvsCattleMean, RFR10cvsDeerMean])

RFR20cvsElkMean = numpy.mean(RFR20cvsScoreElk)
RFR20cvsCattleMean = numpy.mean(RFR20cvsScoreCattle)
RFR20cvsDeerMean = numpy.mean(RFR20cvsScoreDeer)
RFR20Avg = numpy.average([RFR20cvsElkMean, RFR20cvsCattleMean, RFR20cvsDeerMean])

RFR50cvsElkMean = numpy.mean(RFR50cvsScoreElk)
RFR50cvsCattleMean = numpy.mean(RFR50cvsScoreCattle)
RFR50cvsDeerMean = numpy.mean(RFR50cvsScoreDeer)
RFR50Avg = numpy.average([RFR50cvsElkMean, RFR50cvsCattleMean, RFR50cvsDeerMean])

RFR100cvsElkMean = numpy.mean(RFR100cvsScoreElk)
RFR100cvsCattleMean = numpy.mean(RFR100cvsScoreCattle)
RFR100cvsDeerMean = numpy.mean(RFR100cvsScoreDeer)
RFR100Avg = numpy.average([RFR100cvsElkMean, RFR100cvsCattleMean, RFR100cvsDeerMean])



###Ttest b/w best RFR and other classifiers###
concatRFR100 = numpy.concatenate([RFR100cvsScoreElk, RFR100cvsScoreCattle, RFR100cvsScoreDeer]);
statistic, RFR100vsDTtTest = ttest_ind(concatRFR100, concatDT)
statistic, RFR100vsLRtTest = ttest_ind(concatRFR100, concatLR)
statistic, RFR100vsGNBtTest = ttest_ind(concatRFR100, concatGNB)



###Frontend###
print "RFRscore:"
print('ELK')
print RFRscoreElk
print('')
print('CATTLE')
print RFRscoreCattle
print('')
print('DEER')
print RFRscoreDeer
print('')
print('AVG')
print RFRAvg
print('')
print('')
print "RFRcmatrixMean:"
print('')
print RFRcmatrixMean
print('')
print('')
print "RFRTrainscore:"
print('ELK')
print RFRTrainscoreElk
print('')
print('CATTLE')
print RFRTrainscoreCattle
print('')
print('DEER')
print RFRTrainscoreDeer
print('')
print('AVG')
print numpy.average([RFRTrainscoreElk,RFRTrainscoreCattle,RFRTrainscoreDeer])
print('')
print('')
print "RFRTraincmatrixMean:"
print('')
print RFRTraincmatrixMean
print('')
print('')
print "RFR Cross Validation Score:"
print('ELK')
print RFRcvsScoreElk
print('')
print RFRcvsElkMean
print RFRcvsElkStd
print('')
print('CATTLE')
print RFRcvsScoreCattle
print('')
print RFRcvsCattleMean
print RFRcvsCattleStd
print('')
print('DEER')
print RFRcvsScoreDeer
print('')
print RFRcvsDeerMean
print RFRcvsDeerStd
print('')
print('AVG')
print numpy.average([RFRcvsElkMean, RFRcvsCattleMean, RFRcvsDeerMean])
print numpy.average([RFRcvsElkStd, RFRcvsCattleStd, RFRcvsDeerStd])
print('')
print('')
print "DTscore:"
print('ELK')
print DTscoreElk
print('')
print('CATTLE')
print DTscoreCattle
print('')
print('DEER')
print DTscoreDeer
print('')
print('AVG')
print DTAvg
print('')
print('')
print "DTcmatrixMean:"
print('')
print DTcmatrixMean
print('')
print('')
print "DTTrainscore:"
print('ELK')
print DTTrainscoreElk
print('')
print('CATTLE')
print DTTrainscoreCattle
print('')
print('DEER')
print DTTrainscoreDeer
print('')
print('AVG')
print numpy.average([DTTrainscoreElk,DTTrainscoreCattle,DTTrainscoreDeer])
print('')
print('')
print "DTTraincmatrix:"
print('')
print numpy.mean([DTTraincmatrixElk, DTTraincmatrixCattle, DTTraincmatrixDeer], axis=0)
print('')
print('')
print "DT Cross Validation Score:"
print('ELK')
print DTcvsScoreElk
print('')
print DTcvsElkMean
print DTcvsElkStd
print('')
print('CATTLE')
print DTcvsScoreCattle
print('')
print DTcvsCattleMean
print DTcvsCattleStd
print('')
print('DEER')
print DTcvsScoreDeer
print('')
print DTcvsDeerMean
print DTcvsDeerStd
print('')
print('AVG')
print numpy.average([DTcvsElkMean, DTcvsCattleMean, DTcvsDeerMean])
print numpy.average([DTcvsElkStd, DTcvsCattleStd, DTcvsDeerStd])
print('')
print('')
print "LRscore:"
print('ELK')
print LRscoreElk
print('')
print('CATTLE')
print LRscoreCattle
print('')
print('DEER')
print LRscoreDeer
print('')
print('AVG')
print LRAvg
print('')
print('')
print "LRcmatrixMean:"
print('')
print LRcmatrixMean
print('')
print('')
print "LRTrainscore:"
print('ELK')
print LRTrainscoreElk
print('')
print('CATTLE')
print LRTrainscoreCattle
print('')
print('DEER')
print LRTrainscoreDeer
print('')
print('AVG')
print numpy.average([LRTrainscoreElk,LRTrainscoreCattle,LRTrainscoreDeer])
print('')
print('')
print "LRTraincmatrix:"
print('')
print numpy.mean([LRTraincmatrixElk, LRTraincmatrixCattle, LRTraincmatrixDeer], axis=0)
print('')
print('')
print "LR Cross Validation Score:"
print('ELK')
print LRcvsScoreElk
print('')
print LRcvsElkMean
print LRcvsElkStd
print('')
print('CATTLE')
print LRcvsScoreCattle
print('')
print LRcvsCattleMean
print LRcvsCattleStd
print('')
print('DEER')
print LRcvsScoreDeer
print('')
print LRcvsDeerMean
print LRcvsDeerStd
print('')
print('AVG')
print numpy.average([LRcvsElkMean, LRcvsCattleMean, LRcvsDeerMean])
print numpy.average([LRcvsElkStd, LRcvsCattleStd, LRcvsDeerStd])
print('')
print('')
print "GNBscore:"
print('ELK')
print GNBscoreElk
print('')
print('CATTLE')
print GNBscoreCattle
print('')
print('DEER')
print GNBscoreDeer
print('')
print('AVG')
print GNBAvg
print('')
print('')
print "GNBcmatrixMean:"
print('')
print GNBcmatrixMean
print('')
print('')
print "GNBTrainscore:"
print('ELK')
print GNBTrainscoreElk
print('')
print('CATTLE')
print GNBTrainscoreCattle
print('')
print('DEER')
print GNBTrainscoreDeer
print('')
print('AVG')
print numpy.average([GNBTrainscoreElk,GNBTrainscoreCattle,GNBTrainscoreDeer])
print('')
print('')
print "GNBTraincmatrix:"
print('')
print numpy.mean([GNBTraincmatrixElk, GNBTraincmatrixCattle, GNBTraincmatrixDeer], axis=0)
print('')
print('')
print "GNB Cross Validation Score:"
print('ELK')
print GNBcvsScoreElk
print('')
print GNBcvsElkMean
print GNBcvsElkStd
print('')
print('CATTLE')
print GNBcvsScoreCattle
print('')
print GNBcvsCattleMean
print GNBcvsCattleStd
print('')
print('DEER')
print GNBcvsScoreDeer
print('')
print GNBcvsDeerMean
print GNBcvsDeerStd
print('')
print('AVG')
print numpy.average([GNBcvsElkMean, GNBcvsCattleMean, GNBcvsDeerMean])
print numpy.average([GNBcvsElkStd, GNBcvsCattleStd, GNBcvsDeerStd])
print('')
print('')
print "T-tests:"
print "RFR vs DT"
print RFRvsDTtTest
print('')
print "RFR vs LR"
print RFRvsLRtTest
print('')
print "RFR vs GNB"
print RFRvsGNBtTest
print('')
print('')
print "n_estimators:"
print "RFR 10"
print RFR10Avg
print('')
print "RFR 20"
print RFR20Avg
print('')
print "RFR 50"
print RFR50Avg
print('')
print "RFR 100"
print RFR100Avg
print('')
print('')
print "T-tests:"
print "RFR100 vs DT"
print RFR100vsDTtTest
print('')
print "RFR100 vs LR"
print RFR100vsLRtTest
print('')
print "RFR100 vs GNB"
print RFR100vsGNBtTest
print('')
print('')