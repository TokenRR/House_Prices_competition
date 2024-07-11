import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax, skew
from sklearn.preprocessing import MinMaxScaler


log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

squared_features = ['YearRemodAdd', 'LotFrontage_log', 
                    'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
                    'GarageCars_log', 'GarageArea_log']


def handle_missing(features):
        features['Functional'] = features['Functional'].fillna('Typ')
        features['Electrical'] = features['Electrical'].fillna("SBrkr")
        features['KitchenQual'] = features['KitchenQual'].fillna("TA")
        features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
        features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
        features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
        features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

        features["PoolQC"] = features["PoolQC"].fillna("None")
        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
            features[col] = features[col].fillna(0)
        for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            features[col] = features[col].fillna('None')
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            features[col] = features[col].fillna('None')

        features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

        objects = [col for col in features.columns if features[col].dtype == object]
        features.update(features[objects].fillna('None'))

        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric = [col for col in features.columns if features[col].dtype in numeric_dtypes]
        features.update(features[numeric].fillna(0))
        return features


def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res


def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 


def preprocess_data(data):
    data_ID = data['Id']
    data.drop(['Id'], axis=1, inplace=True)

    data["SalePrice"] = np.log1p(data["SalePrice"])

    data.drop(data[(data['OverallQual']<5) & (data['SalePrice']>200000)].index, inplace=True)
    data.drop(data[(data['GrLivArea']>4500) & (data['SalePrice']<300000)].index, inplace=True)
    data.reset_index(drop=True, inplace=True)

    data_labels = data['SalePrice'].reset_index(drop=True)
    data_features = data.drop(['SalePrice'], axis=1)

    data_features['MSSubClass'] = data_features['MSSubClass'].apply(str)
    data_features['YrSold'] = data_features['YrSold'].astype(str)
    data_features['MoSold'] = data_features['MoSold'].astype(str)

    data_features = handle_missing(data_features)

    # Отримати всі числові характеристики
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in data_features.columns:
        if data_features[i].dtype in numeric_dtypes:
            numeric.append(i)

    # Знайти перекіс числових функцій
    skew_features = data_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    # Попереднє масштабування даних
    scaler = MinMaxScaler()
    data_features[skew_index] = scaler.fit_transform(data_features[skew_index])

    # Нормалізувати викривлені елементи
    for i in skew_index:
        try:
            data_features[i] = boxcox1p(data_features[i], boxcox_normmax(data_features[i] + 1))
        except Exception as e:
            print(f"Помилка для функції {i}: {e}")

    data_features['BsmtFinType1_Unf'] = 1*(data_features['BsmtFinType1'] == 'Unf')
    data_features['HasWoodDeck'] = (data_features['WoodDeckSF'] == 0) * 1
    data_features['HasOpenPorch'] = (data_features['OpenPorchSF'] == 0) * 1
    data_features['HasEnclosedPorch'] = (data_features['EnclosedPorch'] == 0) * 1
    data_features['Has3SsnPorch'] = (data_features['3SsnPorch'] == 0) * 1
    data_features['HasScreenPorch'] = (data_features['ScreenPorch'] == 0) * 1
    data_features['YearsSinceRemodel'] = data_features['YrSold'].astype(int) - data_features['YearRemodAdd'].astype(int)
    data_features['Total_Home_Quality'] = data_features['OverallQual'] + data_features['OverallCond']
    data_features = data_features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
    data_features['TotalSF'] = data_features['TotalBsmtSF'] + data_features['1stFlrSF'] + data_features['2ndFlrSF']
    data_features['YrBltAndRemod'] = data_features['YearBuilt'] + data_features['YearRemodAdd']

    data_features['Total_sqr_footage'] = (data_features['BsmtFinSF1'] + data_features['BsmtFinSF2'] +
                                    data_features['1stFlrSF'] + data_features['2ndFlrSF'])
    data_features['Total_Bathrooms'] = (data_features['FullBath'] + (0.5 * data_features['HalfBath']) +
                                data_features['BsmtFullBath'] + (0.5 * data_features['BsmtHalfBath']))
    data_features['Total_porch_sf'] = (data_features['OpenPorchSF'] + data_features['3SsnPorch'] +
                                data_features['EnclosedPorch'] + data_features['ScreenPorch'] +
                                data_features['WoodDeckSF'])
    data_features['TotalBsmtSF'] = data_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    data_features['2ndFlrSF'] = data_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    data_features['GarageArea'] = data_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    data_features['GarageCars'] = data_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    data_features['LotFrontage'] = data_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    data_features['MasVnrArea'] = data_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    data_features['BsmtFinSF1'] = data_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

    data_features['haspool'] = data_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    data_features['has2ndfloor'] = data_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    data_features['hasgarage'] = data_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    data_features['hasbsmt'] = data_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    data_features['hasfireplace'] = data_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    data_features = logs(data_features, log_features)
    
    data_features = squares(data_features, squared_features)

    data_features = pd.get_dummies(data_features).reset_index(drop=True)
    # print(data_features.shape)
    data_features = data_features.loc[:,~data_features.columns.duplicated()]
    # print(data_features.shape)

    return data
