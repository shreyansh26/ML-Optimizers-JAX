import jax
import jax.numpy as np

def nesterov_momentum(J, X, y, gamma=0.95):
    """Performs linear regression using batch nesterov momentum
    
    Args:
        J: cost function
        X: training data
        y: training labels
        gamm: Decay parameter for vecloity sum
        
    Returns:
        params: the weights and bias after performing the optimization
    """
    # Some configurations
    LOG = False
    lr = 0.05    # Learning rate
    
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

        # Update velocity
        params_v['wv'] = gamma * params_v['wv'] - lr * grad_W(X, params['w'] + gamma * params_v['wv'], params['b'] + gamma * params_v['bv'], y)
        params_v['bv'] = gamma * params_v['bv'] - lr * grad_B(X, params['w'] + gamma * params_v['wv'], params['b'] + gamma * params_v['bv'], y)

        # Parameter update
        params['w'] += params_v['wv']
        params['b'] += params_v['bv']

        if LOG and i % 20 == 0:
            print(J(X, params['w'], params['b'], y))
            
    return params