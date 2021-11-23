import pandas as pd 
import joblib

def runGridSearch(gs, X_train,X_test, y_train, y_test, file, message):
    import joblib
    print('joblib imported')
    print('change made')
    gs.fit(X_train, y_train)
    best_estimator = gs.best_estimator_
    train_score = best_estimator.score(X_train, y_train)
    test_score = best_estimator.score(X_test, y_test)
    

    with open(file + '.txt', 'w') as f:
        f.writelines('ReadMe : \n')
        f.writelines(message + '\n\n\n\n')
        f.writelines(f'Model : {best_estimator}\n')
        f.writelines(f'TRAIN SCORE : {str(train_score)} \n')
        f.writelines(f'TEST SCORE : {str(test_score)} \n\n\n')
        
        f.writelines(f'Train Size : {len(X_train)}\n')
        f.writelines(f'Test Size : {len(X_test)}\n\n')
        
        f.writelines('GRID SEARCH PARAMATER DICTIONARY\n')
        f.writelines(str(gs.get_params()) + '\n\n')
        f.writelines('BEST ESTIMATOR PARAMATER DICTIONARY\n')
        f.writelines(pd.DataFrame.from_dict([best_estimator.get_params()]).T.to_string() + '\n\n')
    joblib.dump(best_estimator,file + '.joblib')

    return gs



def calc_vif(data):
    for feature in data.columns:
        X = [f for f in data.columns if f != feature]
        X = data[X]
        y = data[feature]
        r2 = LinearRegression().fit(X,y).score(X,y)
        vif[feature] = 1/(1-r2)
    return pd.DataFrame.from_dict([vif]).T.sort_values(ascending = False, by = 0)
	


