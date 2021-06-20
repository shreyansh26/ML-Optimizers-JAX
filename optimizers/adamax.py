import jax
import jax.numpy as np

def adamax(J, X, y, delta1=0.9, delta2=0.999):
    """Performs linear regression using adamax
    
    Args:
        J: cost function
        X: training data
        y: training labels
        delta1: decay parameter 1
        delta2: decay parameter 2
        
    Returns:
        params: the weights and bias after performing the optimization
    """
    # Some configurations
    LOG = False
    lr = 0.1  # Learning rate
    e = 1e-7  # Epsilon value to prevent the fractions going to infinity when denominator is zero
    
    # The weights and bias terms we will be computing
    params = {
        'w': np.zeros(X.shape[1:]),
        'b': 0.
    }

    # To keep track of velocity parameters
    params_v = {
        'wv': np.zeros(X.shape[1:]),
        'bv': 0.
    }

    # Instead of sum of squares (similar to l2 norm), adamax uses l-infinity norm
    infinity_grad = {
        'w': np.zeros(X.shape[1:]),
        'b': 0.
    }

    # Define the gradient function w.r.t w and b
    grad_W = jax.jit(jax.grad(J, argnums=1))   # argnums indicates which variable to differentiate with from the parameters list passed to the function
    grad_B = jax.jit(jax.grad(J, argnums=2))

    # Run once to compile JIT (Just In Time). The next uses of grad_W and grad_B will now be fast
    grad_W(X, params['w'], params['b'], y)
    grad_B(X, params['w'], params['b'], y)
    
    for i in range(1000):
        # Gradient w.r.t. argumnet index 1 i.e., w
        grad_w = grad_W(X, params['w'], params['b'], y)
        # Gradient w.r.t. argumnet index 2 i.e., b
        grad_b = grad_B(X, params['w'], params['b'], y)

        # Momements update
        params_v['wv'] = delta1 * params_v['wv'] + (1 - delta1) * grad_w
        params_v['bv'] = delta1 * params_v['bv'] + (1 - delta1) * grad_b

        infinity_grad['w'] = np.maximum(delta2 * infinity_grad['w'], np.abs(grad_w))
        infinity_grad['b'] = np.maximum(delta2 * infinity_grad['b'], np.abs(grad_b))

        # Bias correction
        moment_w = params_v['wv'] / (1. - delta1**(i+1))
        moment_b = params_v['bv'] / (1. - delta1**(i+1))


        params['w'] -= (lr / (np.sqrt(infinity_grad['w']) + e)) * moment_w
        params['b'] -= (lr / (np.sqrt(infinity_grad['b']) + e)) * moment_b

        if LOG and i % 20 == 0:
            print(J(X, params['w'], params['b'], y))
            
    return params