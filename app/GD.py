import numpy as np

class RL :
    def __init__(self) :
        self.w = 0.0
        self.b = 0.0
        x, y = np.loadtxt("Advertising2.tsv", delimiter= "\t", unpack = True)
        self.train(x, y, 0.001, 15000)
    
    def update_w_and_b(self, spendings, sales, alpha):
        dr_dw = 0.0
        dr_db = 0.0
        N = len(spendings)

        for i in range(N):
            dr_dw += -2 * spendings[i] * (sales[i] - (self.w * spendings[i] + self.b))
            dr_db += -2 * (sales[i] - (self.w * spendings[i] + self.b))

            # update w and b
            self.w = self.w - (dr_dw/float(N)) * alpha
            self.b = self.b - (dr_db/float(N)) * alpha

        return self.w, self.b 

    def train(self, spendings, sales, alpha, epochs):
        for e in range(epochs):
            self.update_w_and_b(spendings, sales, alpha)

            # log the progress
            if (e == 0) or (e < 3000 and e % 400 == 0) or (e % 3000 == 0):
                print("epoch: ", str(e), "loss: "+str(self.loss(spendings, sales)))
                print("w, b: ", self.w, self.b)
        return self.w, self.b

    def loss(self, spendings, sales):
        N = len(spendings)
        total_error = 0.0
        for i in range(N):
            total_error += (sales[i] - (self.w*spendings[i] + self.b))**2
        return total_error / N
    
    def predict(self, x):
        return self.w*x + self.b








