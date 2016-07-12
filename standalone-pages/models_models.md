# Building Probabilistic Models

A probabilistic model is specified by a joint distribution `p(x,z)` of data `x` and latent variables `z`. All models in Edward are written as a class; to implement a new model, it can be written in any of the currently supported modeling languages: Stan, TensorFlow, and NumPy/SciPy.

To use Stan, simply write a Stan program in the form of a file or string. Then call it with `StanModel(file)` or `StanModel(model_code)`. Here is an example:
```{Python}
model_code = """
    data {
      int<lower=0> N;
      int<lower=0,upper=1> y[N];
    }
    parameters {
      real<lower=0,upper=1> theta;
    }
    model {
      theta ~ beta(1.0, 1.0);
      for (n in 1:N)
        y[n] ~ bernoulli(theta);
    }
"""
model = ed.StanModel(model_code=model_code)
```
Here is a [toy script](https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_stan.py) that uses this model. Stan programs are convenient as [there are many online examples](https://github.com/stan-dev/example-models/wiki), although they are limited to probability models with differentiable latent variables and they can be quite slow to call in practice over TensorFlow.

To use TensorFlow, PyMC3, or NumPy/SciPy, write a class with the method `log_prob(xs, zs)`. This method takes as input a mini-batch of data `xs` and a mini-batch of the latent variables `zs`; the method outputs a vector of the joint density evaluations `[log p(xs, zs[0,:]), log p(xs, zs[1,:]), ...]` with size being the size of the latent variables' mini-batch. Here is an example:
```{Python}
class BetaBernoulli:
    """
    p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
    """
    def __init__(self):
        self.num_vars = 1

    def log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs, a=1.0, b=1.0)
        log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs, z))
                           for z in tf.unpack(zs)])
        return log_lik + log_prior

model = BetaBernoulli()
```
Here is a [toy script](https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_tf.py) that uses this model which is written in TensorFlow. Here is another [toy script](https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_np.py) that uses the same model written in NumPy/SciPy and [another](https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_pymc3.py) written in PyMC3.

For efficiency during future inferences or criticisms, we recommend using the modeling language which contains the most structure about the model; this enables the inference algorithms to automatically take advantage of any available structure if they are implemented to do so. TensorFlow will be most efficient as Edward uses it as the backend for computation.

