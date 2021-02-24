from toolbox.pipelines import basic_preprocessor_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def test_basic_preprocessor_pipeline():
    pipe = basic_preprocessor_pipeline()
    imputer = SimpleImputer()
    scaler = RobustScaler()
    assert pipe.get_params()['imputer__strategy'] == 'mean'
    assert pipe.get_params()['scaler__with_centering'] == True


