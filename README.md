# MSCi 641: Text Analytics Assignment 4

# Results Over 3 Runs

| Activation Fn | Trial 1 | Trial 2 | Trial 3 |
| ------------- |---------|---------|---------|
| Sigmoid       | 0.77037 | 0.77111 | 0.77133 |
| Tanh          | 0.77364 | 0.77217 | 0.76618 |
| ReLU          | 0.76818 | 0.77427 | 0.76482 |

| Dropout Rate  | Trial 1 | Trial 2 | Trial 3 |
| ------------- |---------|---------|---------|
| 0.0           | 0.77107 | 0.77393 | 0.77192 |
| 0.1           | 0.77111 | 0.77192 | 0.77268 |
| 0.3           | 0.77096 | 0.77438 | 0.77157 |
| 0.5           | 0.77020 | 0.77351 | 0.77232 |

| Regularization Param | Trial 1 | Trial 2 | Trial 3 |
| -------------------- |---------|---------|---------|
| 0                    | 0.76894 | 0.76833 | 0.77184 |
| 0.1                  | 0.67628 | 0.68557 | 0.68928 |
| 0.01                 | 0.71403 | 0.71982 | 0.71747 |
| 0.001                | 0.74657 | 0.74616 | 0.74610 |


# Observations & Analysis

1) As we can see from the results, changing the activation function had little to no effect across the trials of the experiment. I hypothesize that this is expected due to the lack of depth of the constructed neural network. It is known that the hyperbolic activation functions suffer from the vanishing gradient problem (here it is to be noted that sigmoid is just a rescaled tanh). The max value of the derivative of sigmoid is 0.25 of the original sigmoid value. Since ReLU does not suffer
from the vanishing gradient problem (its derivative is either 0 or 1), it is used as the popular activation function in deep networks. Since the network we constructed has only 1 hidden layer, all the activation functions are expected to perform roughly the same as the gradients are strong enough when they reach the weights to be updated.

2) I have used 1024 units in the hidden layer. As we can see, dropout rate of 0 - 0.5 have no visible effect on the result. This means that the hidden layer can do well given that at max 50% of the units have been dropped randomly during training. We can infer from this that having 1024 units in the hidden layer is extraneous and can be reduced without causing a significant reduction in the accuracy.

3) We can see that adding L2 regularization with varying regularization parameter degrades performance significantly.
