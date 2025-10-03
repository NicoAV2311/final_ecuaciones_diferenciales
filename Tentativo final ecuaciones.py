"""
Proyecto mejorado: Red XOR + EDOs
- Modo Manual: Euler / RK4 implementados a mano.
- Modo Keras: entrenamiento usando tensorflow.keras.
- Modo SciPy ODE: integra dW/dt = -eta * grad L(W) con solve_ivp.
Guardar como: nn_edos_gui_improved.py
"""

import sys
import numpy as np
from typing import Tuple, List, Optional

# GUI and plotting
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QTextEdit, QLineEdit, QHBoxLayout, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Try to import optional libs (tensorflow, scipy). If missing, we'll notify user.
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    TF_OK = True
except Exception as e:
    TF_OK = False
    tf = None

try:
    from scipy.integrate import solve_ivp
    SCIPY_OK = True
except Exception as e:
    SCIPY_OK = False
    solve_ivp = None


# --------------------------
# Implementación manual (SimpleNN) - métodos numéricos hechos a mano
# --------------------------
class SimpleNN:
    def __init__(self, eta: float = 0.1, epochs: int = 2000, method: str = "Euler") -> None:
        # XOR data
        self.X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        self.y = np.array([[0],[1],[1],[0]], dtype=float)
        self.eta = float(eta)
        self.epochs = int(epochs)
        self.method = method

        np.random.seed(42)
        self.W1 = np.random.randn(2, 3)
        self.b1 = np.zeros((1,3))
        self.W2 = np.random.randn(3, 1)
        self.b2 = np.zeros((1,1))
        self.loss_history: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return s * (1 - s)

    def forward(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = self.X @ self.W1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)
        return z1, a1, z2, a2

    def compute_gradients(self):
        z1, a1, z2, a2 = self.forward()
        loss = np.mean((self.y - a2) ** 2)
        dL_da2 = 2 * (a2 - self.y) / self.y.shape[0]
        da2_dz2 = self._sigmoid_derivative(z2)
        dL_dz2 = dL_da2 * da2_dz2
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        dL_da1 = dL_dz2 @ self.W2.T
        da1_dz1 = self._sigmoid_derivative(z1)
        dL_dz1 = dL_da1 * da1_dz1
        dL_dW1 = self.X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        return float(loss), (dL_dW1, dL_db1, dL_dW2, dL_db2)

    def step_euler(self):
        loss, grads = self.compute_gradients()
        dW1, db1, dW2, db2 = grads
        self.W1 -= self.eta * dW1
        self.b1 -= self.eta * db1
        self.W2 -= self.eta * dW2
        self.b2 -= self.eta * db2
        return loss, dW1, dW2

    def step_rk4(self):
        # RK4 for the vector of parameters (applied componentwise using gradients)
        # k1
        loss1, grads1 = self.compute_gradients()
        dW1_1, db1_1, dW2_1, db2_1 = grads1
        # save original
        W1_orig, b1_orig, W2_orig, b2_orig = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()

        # k2
        self.W1 = W1_orig - 0.5 * self.eta * dW1_1
        self.b1 = b1_orig - 0.5 * self.eta * db1_1
        self.W2 = W2_orig - 0.5 * self.eta * dW2_1
        self.b2 = b2_orig - 0.5 * self.eta * db2_1
        _, grads2 = self.compute_gradients()
        dW1_2, db1_2, dW2_2, db2_2 = grads2

        # k3
        self.W1 = W1_orig - 0.5 * self.eta * dW1_2
        self.b1 = b1_orig - 0.5 * self.eta * db1_2
        self.W2 = W2_orig - 0.5 * self.eta * dW2_2
        self.b2 = b2_orig - 0.5 * self.eta * db2_2
        _, grads3 = self.compute_gradients()
        dW1_3, db1_3, dW2_3, db2_3 = grads3

        # k4
        self.W1 = W1_orig - self.eta * dW1_3
        self.b1 = b1_orig - self.eta * db1_3
        self.W2 = W2_orig - self.eta * dW2_3
        self.b2 = b2_orig - self.eta * db2_3
        _, grads4 = self.compute_gradients()
        dW1_4, db1_4, dW2_4, db2_4 = grads4

        # restore original and update using RK4 combination
        self.W1, self.b1, self.W2, self.b2 = W1_orig, b1_orig, W2_orig, b2_orig
        self.W1 -= (self.eta/6.0) * (dW1_1 + 2*dW1_2 + 2*dW1_3 + dW1_4)
        self.b1 -= (self.eta/6.0) * (db1_1 + 2*db1_2 + 2*db1_3 + db1_4)
        self.W2 -= (self.eta/6.0) * (dW2_1 + 2*dW2_2 + 2*dW2_3 + dW2_4)
        self.b2 -= (self.eta/6.0) * (db2_1 + 2*db2_2 + 2*db2_3 + db2_4)

        return loss1, dW1_1, dW2_1

    def step(self):
        if self.method == "Euler":
            return self.step_euler()
        elif self.method == "RK4":
            return self.step_rk4()
        else:
            # default to Euler if unknown
            return self.step_euler()

    def train(self) -> np.ndarray:
        self.loss_history = []
        for epoch in range(self.epochs):
            loss, _, _ = self.step()
            self.loss_history.append(float(loss))
        _, _, _, a2 = self.forward()
        return a2


# --------------------------
# Helpers for TensorFlow <-> flat vector conversion (for SciPy ODE mode)
# --------------------------
def get_flat_weights_from_keras(model: "tf.keras.Model") -> np.ndarray:
    """Return flattened weights vector from a tf.keras model (numpy)."""
    w_list = model.get_weights()
    flat = np.concatenate([w.flatten() for w in w_list]).astype(np.float64)
    return flat

def set_flat_weights_to_keras(model: "tf.keras.Model", flat: np.ndarray):
    """Set model weights from flat vector (numpy)."""
    shapes = [w.shape for w in model.get_weights()]
    sizes = [int(np.prod(s)) for s in shapes]
    parts = []
    idx = 0
    for s in sizes:
        parts.append(flat[idx: idx + s])
        idx += s
    new_ws = []
    for arr, shape in zip(parts, shapes):
        new_ws.append(arr.reshape(shape))
    model.set_weights(new_ws)


# --------------------------
# GUI
# --------------------------
class NNApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Red Neuronal + EDOs (Mejorada)")
        self.setGeometry(150, 150, 1000, 700)

        layout = QVBoxLayout()

        # Params layout
        params_layout = QHBoxLayout()
        self.mode_label = QLabel("Modo:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Keras", "SciPy ODE"])
        params_layout.addWidget(self.mode_label)
        params_layout.addWidget(self.mode_combo)

        self.lr_label = QLabel("η:")
        self.lr_input = QLineEdit("0.1")
        params_layout.addWidget(self.lr_label)
        params_layout.addWidget(self.lr_input)

        self.epochs_label = QLabel("Épocas:")
        self.epochs_input = QLineEdit("5000")
        params_layout.addWidget(self.epochs_label)
        params_layout.addWidget(self.epochs_input)

        # method selector for manual mode
        self.method_label = QLabel("Método (manual):")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Euler", "RK4"])
        params_layout.addWidget(self.method_label)
        params_layout.addWidget(self.method_combo)

        layout.addLayout(params_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Entrenar / Ejecutar")
        self.train_btn.clicked.connect(self.train_network)
        btn_layout.addWidget(self.train_btn)

        self.step_btn = QPushButton("Un Paso (solo Manual)")
        self.step_btn.clicked.connect(self.one_step)
        btn_layout.addWidget(self.step_btn)

        self.reset_btn = QPushButton("Resetear")
        self.reset_btn.clicked.connect(self.reset_network)
        btn_layout.addWidget(self.reset_btn)

        layout.addLayout(btn_layout)

        self.results = QTextEdit()
        self.results.setReadOnly(True)
        layout.addWidget(self.results)

        # Plotting
        self.figure = Figure(figsize=(6,3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # internal state
        self.manual_nn: Optional[SimpleNN] = None
        self.keras_model = None
        self.keras_loss_history: List[float] = []
        self.scipy_solution = None

    def train_network(self) -> None:
        mode = self.mode_combo.currentText()
        try:
            eta = float(self.lr_input.text())
            epochs = int(self.epochs_input.text())
        except ValueError:
            self.results.setText("Parámetros inválidos (η y épocas deben ser numéricos).")
            return

        if mode == "Manual":
            method = self.method_combo.currentText()
            self.manual_nn = SimpleNN(eta=eta, epochs=epochs, method=method)
            outputs = self.manual_nn.train()
            preds = (outputs > 0.5).astype(int)
            txt = f"Manual mode ({method}) - Resultados finales:\n"
            for i in range(len(self.manual_nn.X)):
                txt += f"{self.manual_nn.X[i]} -> pred: {int(preds[i][0])} (real {int(self.manual_nn.y[i][0])}) [{outputs[i][0]:.3f}]\n"
            txt += f"\nÚltima pérdida: {self.manual_nn.loss_history[-1]:.8f}\n"
            self.results.setText(txt)

            # plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.manual_nn.loss_history, label=f"Manual {method}")
            ax.set_title(f"Pérdida ({mode} - {method})")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("MSE")
            ax.legend()
            ax.grid(True)
            self.canvas.draw()

        elif mode == "Keras":
            if not TF_OK:
                self.results.setText("TensorFlow / Keras no está disponible en este entorno.")
                return
            # Build a small model and train with Keras (SGD) - we will keep sigmoid activations like manual
            model = Sequential([Dense(3, input_shape=(2,), activation='sigmoid'), Dense(1, activation='sigmoid')])
            opt = tf.keras.optimizers.SGD(learning_rate=eta)
            model.compile(optimizer=opt, loss='mse', metrics=[])
            # fit
            history = model.fit(np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[0],[1],[1],[0]]),
                                epochs=epochs, verbose=0)
            self.keras_model = model
            self.keras_loss_history = history.history['loss']

            preds = (model.predict(np.array([[0,0],[0,1],[1,0],[1,1]])) > 0.5).astype(int)
            txt = f"Keras mode (SGD lr={eta}) - Resultados:\n"
            for i, x in enumerate([[0,0],[0,1],[1,0],[1,1]]):
                prob = model.predict(np.array([x]))[0,0]
                txt += f"{x} -> pred: {int(preds[i,0])} (real {int(self.manual_nn.y[i,0]) if self.manual_nn else [0,1,1,0][i]}) [{prob:.3f}]\n"
            txt += f"\nÚltima pérdida: {self.keras_loss_history[-1]:.8f}\n"
            self.results.setText(txt)

            # plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.keras_loss_history, label="Keras SGD")
            ax.set_title("Pérdida (Keras SGD)")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("MSE")
            ax.legend()
            ax.grid(True)
            self.canvas.draw()

        elif mode == "SciPy ODE":
            if not (TF_OK and SCIPY_OK):
                self.results.setText("Para SciPy ODE necesitas: scipy y tensorflow instalados.")
                return

            # Build a Keras model to reuse structure and gradients computation
            model = Sequential([Dense(3, input_shape=(2,), activation='sigmoid'), Dense(1, activation='sigmoid')])
            # We will integrate weights as a flat vector
            # initial flat weights
            flat0 = get_flat_weights_from_keras(model)

            # define ODE: dy/dt = -eta * grad(loss) where y is flat weights
            X_train = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
            y_train = np.array([[0],[1],[1],[0]], dtype=np.float64)

            def ode_fun(t, y_flat):
                # set model weights
                set_flat_weights_to_keras(model, y_flat.astype(np.float32))
                # compute loss and gradients using TF
                with tf.GradientTape() as tape:
                    x_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
                    y_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
                    preds = model(x_tf, training=False)
                    loss_tf = tf.reduce_mean(tf.square(y_tf - preds))
                grads = tape.gradient(loss_tf, model.trainable_weights)
                # flatten grads
                grad_flat = np.concatenate([g.numpy().flatten() for g in grads]).astype(np.float64)
                # dy/dt = -eta * grad_flat
                return (-eta * grad_flat)

            # integrate from t=0..T where T = epochs (we interpret epochs as time units)
            t_span = (0.0, float(epochs))
            # choose solver (RK45 default)
            sol = solve_ivp(ode_fun, t_span, flat0, method='RK45', rtol=1e-6, atol=1e-9, max_step=1.0)
            self.scipy_solution = sol

            # final weights -> set and predict
            final_flat = sol.y[:, -1]
            set_flat_weights_to_keras(model, final_flat.astype(np.float32))
            preds = (model.predict(X_train) > 0.5).astype(int)
            losses = []
            # compute loss timeline by sampling solution at returned times
            for i_t, y_flat in enumerate(sol.y.T):
                set_flat_weights_to_keras(model, y_flat.astype(np.float32))
                preds_t = model.predict(X_train)
                losses.append(float(np.mean((y_train - preds_t) ** 2)))

            txt = f"SciPy ODE mode (integrador RK45) - Resultados:\n"
            for i, x in enumerate(X_train):
                prob = model.predict(np.array([x]))[0,0]
                txt += f"{x.tolist()} -> pred: {int(preds[i,0])} (real {int(y_train[i,0])}) [{prob:.3f}]\n"
            txt += f"\nPérdida final (último paso): {losses[-1]:.8f}\n"
            self.results.setText(txt)

            # plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(sol.t, losses, label='Loss (SciPy solve_ivp)')
            ax.set_title('Pérdida (SciPy ODE integration)')
            ax.set_xlabel('Tiempo (epocas equivalentes)')
            ax.set_ylabel('MSE')
            ax.legend()
            ax.grid(True)
            self.canvas.draw()

        else:
            self.results.setText("Modo no reconocido.")

    def one_step(self) -> None:
        mode = self.mode_combo.currentText()
        if mode != "Manual":
            self.results.setText("El modo 'Un paso' está disponible solo en Modo Manual.")
            return

        try:
            eta = float(self.lr_input.text())
        except ValueError:
            self.results.setText("η inválido.")
            return

        if self.manual_nn is None:
            method = self.method_combo.currentText()
            self.manual_nn = SimpleNN(eta=eta, epochs=1, method=method)

        loss, gradW1, gradW2 = self.manual_nn.step()
        self.manual_nn.loss_history.append(float(loss))
        txt = self.results.toPlainText()
        txt += f"\nPaso manual ({self.manual_nn.method}) #{len(self.manual_nn.loss_history)}\n"
        txt += f"dW1 (muestra):\n{gradW1}\n"
        txt += f"dW2 (muestra):\n{gradW2}\n"
        txt += f"Loss: {loss:.8f}\n"
        self.results.setText(txt)

        # update plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.manual_nn.loss_history, label=f"Manual {self.manual_nn.method}")
        ax.set_title("Pérdida (Manual step)")
        ax.set_xlabel("Pasos")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

    def reset_network(self) -> None:
        # reset all internal objects
        self.manual_nn = None
        self.keras_model = None
        self.keras_loss_history = []
        self.scipy_solution = None
        self.results.clear()
        self.figure.clear()
        self.canvas.draw()
        self.results.setText("Estado reiniciado. Selecciona modo y parámetros y presiona 'Entrenar / Ejecutar'.")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NNApp()
    window.show()
    sys.exit(app.exec_())
