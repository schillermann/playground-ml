import numpy as np

# X Reservierungen
# Y Pizzabestellungen
# w Gerade definiert
# lr Lernrate - Schrittgröße
# b Bias - Verschiebung

def predict(X: int, w: float, b: float) -> float:
    return X * w + b

def loss(X: int, Y: int, w: float, b: float) -> float:
    return np.average((predict(X, w, b) - Y) ** 2)

def train(X: int,Y: int, iterations: int, lr: float):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b
    raise Exception("Could not converge within %d iterations" % iterations)

# Importiert den Datensatz
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
# Trainiert das System
w, b = train(X,Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))
# Sagt die Anzahl der Pizzas voraus
print("Prediction: Bestellungen x=%d => Pizzas y=%.2f" % (20, predict(20, w, b)))

# Plot the chart
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
plt.plot(X, Y, "bo")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
plt.show()