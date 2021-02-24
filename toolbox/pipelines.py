from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn import set_config; set_config(display='diagram')

def basic_preprocessor_pipeline():
    preprocessor = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', RobustScaler())
        ])
    return preprocessor
