import numpy as np

# X Reservierungen
# Y Pizzabestellungen
# w Gerade definiert
# lr Lernrate - Schrittgröße

def predict(X, w):
    return X * w

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

def train(X,Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w
    raise Exception("Could not converge within %d iterations" % iterations)

# Importiert den Datensatz
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
# Trainiert das System
w = train(X,Y, iterations=10000, lr=0.01)
print("w=%.3f" % w)
# Sagt die Anzahl der Pizzas voraus
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w)))