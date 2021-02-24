from toolbox.mltools import init_linreg

def test_init_linreg():
    model = init_linreg()

    assert model.fit_intercept
