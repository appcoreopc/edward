Data in Edward is stored as a Python dictionary. It is usually
comprised of strings binded to NumPy arrays such as a key `'x'` with
value `np.array([0.23512, 13.2])`. Some modeling languages require the data
to have a different key or value type. We detail each below.

+ __TensorFlow.__ The data carries whatever keys and values the user accesses in the user-defined model. Key is a string. Value is a NumPy array or TensorFlow tensor.
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
+ __Python.__ The data carries whatever keys and values the user accesses in the user-defined model. Key is a string. Value is a NumPy array or TensorFlow tensor.
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
+ __PyMC3.__ The data binds Theano shared variables, which are used to mark the observed PyMC3 random variables, to their realizations. Key is a Theano shared variable. Value is a NumPy array or TensorFlow tensor.
```python
x_obs = theano.shared(np.zeros(1))
with pm.Model() as pm_model:
    beta = pm.Beta('beta', 1, 1, transform=None)
    x = pm.Bernoulli('x', beta, observed=x_obs)

model = PyMC3Model(pm_model)
data = {x_obs: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
```
+ __Stan.__ The data is according to the Stan program's data block. Key is a string. Value is whatever type is used for the data block.
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

## Reading Data in Edward

There are three ways to read data in Edward, following the
[three ways to read data in TensorFlow](https://www.tensorflow.org/versions/r0.9/how_tos/reading_data/index.html).

1. __Preloaded data.__ A constant or variable in the TensorFlow graph holds all the data.

   For inference, pass in the data as a dictionary of NumPy arrays. Internally, we will store them in TensorFlow variables to prevent copying data more than once in memory. Batch training is available internally via `tf.train.slice_input_producer` and `tf.train.batch`. (As an example, see `examples/mixture_gaussian.py`.)
2. __Feeding.__ Manual code provides the data when running each step of inference.

   For inference, pass in the data as a dictionary of TensorFlow placeholders. The user must manually feed the placeholders at each step of inference: initialize via `inference.initialize()`; then in a loop call `sess.run(inference.train, feed_dict={...})` where in `feed_dict` you pass in the values for the `tf.placeholder`'s. (As an example, see `examples/mixture_density_network.py` or `examples/convolutional_vae.py`.)
3. __Reading from files.__ An input pipeline reads the data from files at the beginning of a TensorFlow graph.

   For inference, pass in the data as a dictionary of TensorFlow tensors, where the tensors are the output of data readers. (No current example is available.)

## Training Models with Data

How do we use the data during training? In general there are three use cases:

1. Train over the full data per step. (supported for all languages)

   Follow the setting of preloaded data.
2. Train over a batch per step when the full data fits in memory. (scale inference in terms of computational complexity; supported for all but Stan)


   Follow the setting of preloaded data. Specify the batch size with `n_data` in `Inference`. By default, we will subsample by slicing along the first dimension of every data structure in the data dictionary. Alternatively, follow the setting of feeding. Manually deal with the batch behavior at each training step.
3. Train over batches per step when the full data does not fit in memory. (scale inference in terms of computational complexity and memory complexity; supported for all but Stan)

   Follow the setting of reading from files. Alternatively, follow the setting of feeding, and use a generator to create and destroy NumPy arrays on the fly for feeding the placeholders.
