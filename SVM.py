import numpy as np

#kernel function add to be done

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.w) - self.b
                y_predicted = np.sign(linear_output)

                if y[idx] * linear_output <= 1:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    db = -y[idx]
                else:
                    dw = 2 * self.lambda_param * self.w
                    db = 0

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            print(f"Iteration {_+1}/{self.n_iters}, Weights: {self.w}, Bias: {self.b}")

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        y_predicted = np.sign(linear_output)
        return y_predicted
    

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([-1, 1, 1, -1])  

mySVM = SVM(learning_rate=0.1, n_iters=1000)
mySVM.fit(X, y)
predictions = mySVM.predict(X)
print(predictions)