There is one data abstraction to rule all them all. It is the simplest abstraction satisfying all desired features: a Python dictionary.

Data as a Python dictionary bodes nicely with TensorFlow's `feed_dict` concept: in the same way `feed_dict` is a dictionary binding TensorFlow placeholders to NumPy arrays, `data` is a dictionary binding random variables to realizations. This connection will be exploited even more as we implement the modeling language.

How do the different languages interact with the data dictionary?
+ TensorFlow: The dictionary carries whatever keys and values the user accesses in `log_prob()` (or in the other user-defined methods). Key is a string. Value is a NumPy array or TensorFlow placeholder.
```python
class BetaBernoulli:
    def log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs, a=1.0, b=1.0)
        log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs['x'], z))
                           for z in tf.unpack(zs)])
        return log_lik + log_prior

model = BetaBernoulli()
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
```
+ Python: The dictionary carries whatever keys and values the user accesses in `log_prob()` (or in the other user-defined methods). Key is a string. Value is a NumPy array or TensorFlow placeholder.
```python
class BetaBernoulli(PythonModel):
    def _py_log_prob(self, xs, zs):
        n_minibatch = zs.shape[0]
        lp = np.zeros(n_minibatch, dtype=np.float32)
        for b in range(n_minibatch):
            lp[b] = beta.logpdf(zs[b, :], a=1.0, b=1.0)
            for n in range(len(xs['x'])):
                lp[b] += bernoulli.logpmf(xs['x'][n], p=zs[b, :])

        return lp

model = BetaBernoulli()
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
```
+ PyMC3: The dictionary binds Theano shared variables, which are used to mark the observed PyMC3 random variables, to their realizations. Key is a Theano shared variable. Value is a NumPy array or TensorFlow placeholder.
```python
x_obs = theano.shared(np.zeros(1))
with pm.Model() as pm_model:
    beta = pm.Beta('beta', 1, 1, transform=None)
    x = pm.Bernoulli('x', beta, observed=x_obs)

model = PyMC3Model(pm_model)
data = {x_obs: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
```
+ Stan: It's the usual dictionary according to the Stan program's data block. Key is a string. Value is whatever type is used for the data block.
```python
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
data = {'N': 10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
```
How do we use the data during training? Let's detail the three use cases.
+ Initialize training with full data. Loop over all data per iteration. (supported for all languages)

  Pass in the data via `inference = ed.MFVI(model, variational, data)`, then call `inference.run()`. See `examples/beta_bernoulli_tf.py` as an example.
+ Initialize training with full data. Loop over a batch per iteration. (scale inference in terms of computational complexity; supported for all but Stan)

  Pass in the data via `inference = ed.MFVI(model, variational, data)`, then call `inference.run(n_data=5)`. By default, we will subsample by slicing along the first dimension of every data structure in the data dictionary. See `examples/mixture_gaussian.py` as an example.
+ Initialize training with no data. Manually pass in a batch per iteration. (scale inference in terms of computational complexity and memory complexity; supported for all but Stan)

  Define your data dictionary by using `tf.placeholder()`'s. Pass in the data via `inference = ed.MFVI(model, variational, data)`. Initialize via `inference.initialize()`. Then in a loop run `sess.run(inference.train, feed_dict={...})` where in the `feed_dict` you pass in the values for the `tf.placeholder()`'s. See `examples/mixture_density_network.py` as an example.

What kind of black magic is done internally to get this to work? All methods—model methods, inference methods, criticism methods—work with the data dictionary. For inference, we internally use a data abstraction which you can think of now as a data generator. This data generator simply outputs batches of data during training. The typical Edward user will not have to learn about it (the typical Edward inference researcher would).

## Data Generators
