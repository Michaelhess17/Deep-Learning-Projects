from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda



class LambdaLayer(Layer):

    def __init__(self, lambda_func, output_dim, **kwargs):
        self.output_dim = output_dim
        self.lambda_func = lambda_func
        super(LambdaLayer, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'lambda_func': self.lambda_func,
        })
        return config
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_shape),
        #                               initializer='uniform',
        #                               trainable=True)
        super(LambdaLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return Lambda(self.lambda_func, output_shape=(self.output_dim,))(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim