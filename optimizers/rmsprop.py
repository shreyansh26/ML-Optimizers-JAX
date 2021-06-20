import jax
import jax.numpy as np

def rmsprop(J, X, y, delta=0.95):
    """Performs linear regression using rmsprop
    
    Args:
        J: cost function
        X: training data
        y: training labels
        delta: decay parameter
        
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

    # To keep running sum of squares of gradients with decay
    squared_grad = {
        'w': np.zeros(X.shape[1:]),
        'b': 0.
    }

    # Define the gradient function w.r.t w and b
    grad_W = jax.jit(jax.grad(J, argnums=1))   # argnums indicates which variable to differentiate with from the parameters list passed to the function
    grad_B = jax.jit(jax.grad(J, argnums=2))

    # Run once to compile JIT (Just In Time). The next uses of grad_W and grad_B will now be fast
    grad_W(X, params['w'], params['b'], y)
    grad_B(X, params['w'], params['b'], y)
    
    for i in range(5000):
        # Gradient w.r.t. argumnet index 1 i.e., w
        grad_w = grad_W(X, params['w'], params['b'], y)
        # Gradient w.r.t. argumnet index 2 i.e., b
        grad_b = grad_B(X, params['w'], params['b'], y)

        # Running decaying sum of squares of gradients
        squared_grad['w'] = delta * squared_grad['w'] + (1 - delta) * grad_w * grad_w
        squared_grad['b'] = delta * squared_grad['b'] + (1 - delta) * grad_b * grad_b

        # Parameter update
        params['w'] -= (lr / (np.sqrt(squared_grad['w']) + e)) * grad_w
        params['b'] -= (lr / (np.sqrt(squared_grad['b']) + e)) * grad_b

        if LOG and i % 20 == 0:
            print(J(X, params['w'], params['b'], y))
            
    return params
