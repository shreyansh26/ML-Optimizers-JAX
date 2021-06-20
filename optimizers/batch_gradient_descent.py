import jax
import jax.numpy as np

def batch_gradient_descent(J, X, y):
    """Performs linear regression using batch gradient descent optimizer
    
    Args:
        J: cost function
        X: training data
        y: training labels
        
    Returns:
        params: the weights and bias after performing the optimization
    """
    # Some configurations
    LOG = False  # To print loss after every n epochs
    lr = 0.05    # Learning rate
    
    # The weights and bias terms we will be computing
    params = {
        'w': np.zeros(X.shape[1:]),
        'b': 0.
    }
    
    # Define the gradient function w.r.t w and b
    grad_W = jax.jit(jax.grad(J, argnums=1))    # argnums indicates which variable to differentiate with from the parameters list passed to the function
    grad_B = jax.jit(jax.grad(J, argnums=2))

    # Run once to compile JIT (Just In Time). The next uses of grad_W and grad_B will now be fast
    grad_W(X, params['w'], params['b'], y)
    grad_B(X, params['w'], params['b'], y)


    for i in range(1000):
        # Gradient w.r.t. argumnet index 1 i.e., w and parameter update
        params['w'] -= lr * grad_W(X, params['w'], params['b'], y)
        # Gradient w.r.t. argumnet index 2 i.e., b and parameter update
        params['b'] -= lr * grad_B(X, params['w'], params['b'], y)

        if LOG and i % 20 == 0:
            print(J(X, params['w'], params['b'], y))
    
    return params
