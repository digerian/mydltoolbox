from toolbox.mltools import init_nn_binary_classifier

def test_init_nn_binary_classifier():
    model = init_nn_binary_classifier()
    assert model.count_params() == 401
