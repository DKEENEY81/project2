import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pylab as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', 100)


Cases = pd.read_csv('data/Cases2018.csv', encoding='iso-8859-1')

##Examine data, drop columns and labels
unneeded = [['caseId','docketId','lexisCite','term','naturalCourt','docket','chief','caseName','caseIssuesId','voteId','dateDecision', 'decisionType','usCite','sctCite','ledCite', 'issue','issueArea','decisionDirection','decisionDirectionDissent','authorityDecision1','authorityDecision2','lawType','lawSupp','lawMinor','majOpinWriter','majOpinAssigner']]
labels = [['declarationUncon', 'caseDisposition', 'caseDispositionUnusual','partyWinning', 'precedentAlteration','voteUnclear','splitVote','majVotes','minVotes']]

Cases.drop(unneeded, axis=1, inplace=True)
Cases.drop(labels, axis=1, inplace=True)

#Cases.head()

Cases[['dateArgument','dateRearg']]=Cases[['dateArgument', 'dateRearg']].notnull().astype(int)

#Cases.info()
# Cases.groupby('respondent').count()

# Cases[Cases['respondent'].isnull()]

## Filling NaNs with either a newly defined varible 0 to represent unspecified, or using an already defined unspecified/unclear variable
Cases[['petitionerState','adminActionState','threeJudgeFdc','respondentState',]] = Cases[['petitionerState','adminActionState','threeJudgeFdc','respondentState',]].fillna(value=0.0)
Cases['respondent'] = Cases[['respondent']].fillna(value=501)# There was already a code for unidentfiable which i reused for NaN here
Cases['adminAction'] = Cases[['adminAction']].fillna(value=118.0)# There was already a code for unidentfiable which i reused for NaN here
Cases['caseOrigin'] = Cases[['caseOrigin']].fillna(value=0.0)#meaning originated in supreme court
Cases['caseSource'] = Cases[['caseSource']].fillna(value=0.0)##meaning originated in supreme court
Cases['caseOriginState'] = Cases[['caseOriginState']].fillna(value=0.0)
Cases['caseSourceState'] = Cases[['caseSourceState']].fillna(value=0.0)
Cases['lcDisagreement'] = Cases[['lcDisagreement']].fillna(value=0.0)
Cases['certReason'] = Cases[['certReason']].fillna(value=12)
Cases['lcDisposition'] = Cases[['lcDisposition']].fillna(value=0.0)
Cases['lcDispositionDirection'] = Cases[['lcDispositionDirection']].fillna(value=3.0)


# Cases.info()

finalLabels = labels['partyWinning']
Cases['result'] = labels['partyWinning']
Cases.groupby('dateArgument').count()


Cases[Cases.result.notnull()]

##filling NaNs in result with 2.0 = unclear, there are so few of these, I might drop these ~20 rows
Cases['result'] = Cases[['result']].fillna(value=2.0)

#dropping 19 rows with no discernible winner
Cases.drop(Cases[Cases.result==2.0].index, inplace = True) 

#preparing model for training
labels2 = Cases[['result']]
Cases.drop('result', axis=1, inplace=True)


def cats(DF):
    for i in DF.columns:
        DF[i] = pd.Categorical(DF[i])
    return DF


cats(Cases)

#check to see if it worked
Cases.info()

#all good
#dummying columns that aren't binary already
binCols = ['dateArgument', 'dateRearg','threeJudgeFdc','lcDisagreement']
# Takes all 14 other columns
dummy_cols = list(set(Cases2.columns) - set(binCols))
Cases2 = pd.get_dummies(Cases2, columns=dummy_cols, drop_first=True)

##preparing for model
X_train, X_test, y_train, y_test = train_test_split(Cases2, labels2, test_size=0.2, random_state=23)

# kept getting warned about this column vector error so using ravel to address
y_test = np.ravel(y_test)
y_train = np.ravel(y_train)

params = {'n_estimators': 4000, 'learning_rate': 0.01, 'random_state':23}
GBC = GradientBoostingClassifier(**params)

GBC.fit(X_train, y_train)

#GBC.score(X_test, y_test)
	##.6630985
#GBC.score(X_train, y_train)
	##.737991

#5679/8874 = .64 cases labeled 1




##plotting test train deviance for n_estimator parameter tuning
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(GBC.staged_predict(X_test)):
    test_score[i] = GBC.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, GBC.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

##looking at feature importance but i have too many
feature_importance = GBC.feature_importances_

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, Cases2.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()