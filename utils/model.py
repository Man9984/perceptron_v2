import numpy as np

class Perceptron:

    def __init__(self,eta, epochs):
        self.weights = np.random.randn(3) #  "standard normal" distribution.
        self.eta = eta
        self.epochs = epochs # one itteration that means :- (forward + backward) = one epoch
        print(f"Initial weights before training : \n {self.weights}")

    def activation(self,input, weights):
        z = np.dot(input,weights)
        return np.where(z>0,1,0)

    def fit(self,X,y):
        self.X = X
        self.y = y
        X_with_bias = np.c_[self.X,(-np.ones(len(self.X)))] #np.c_[self.X,(-np.ones(len(self.X)))] # Concatinate with input with bias.
        print(f"X_with bias : \n {X_with_bias}")

        for epoch in range(self.epochs):
            print("___"*20)
            print(f"for Epoch : {epoch}")
            print("___" * 20)

            y_had = self.activation(X_with_bias, self.weights) # Forword Propogation
            print(f"Predicted Value After Forword propagation : \n {y_had}")

            self.error = self.y - y_had
            print("Error :")
            print(self.error)
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) # This is the Backward Propogation.
            print(f"updated weights after epoch : \n {epoch}/{self.epochs}: {self.weights}")
            print("######"*10)

    def predict(self,X):
        X_with_bias = np.c_[X,(-np.ones(len(X)))]
        return self.activation(X_with_bias, self.weights)

    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"Total Loss = {total_loss}")
        return total_loss

    
