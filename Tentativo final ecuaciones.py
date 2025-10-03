

import sys
import numpy as np
from typing import Tuple, List, Optional

# GUI and plotting
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QTextEdit, QLineEdit, QHBoxLayout, QComboBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Try to import optional libs (tensorflow, scipy). If missing, we'll notify user.
try:
    import tensorflow as tf
    from keras import Sequential
    from keras.layers import Dense
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
    """Red neuronal simple para XOR con métodos de integración manuales (Euler, RK4)."""
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
        """
        Propagación hacia adelante de la red.
        Returns: z1, a1, z2, a2
        """
        z1 = self.X @ self.W1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)
        return z1, a1, z2, a2

    def compute_gradients(self) -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calcula gradientes de la función de pérdida respecto a todos los parámetros.
        Returns: loss, (dW1, db1, dW2, db2)
        """
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

    def step_euler(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Un paso de Euler explícito.
        Returns: loss, gradW1, gradW2
        """
        loss, grads = self.compute_gradients()
        dW1, db1, dW2, db2 = grads
        self.W1 -= self.eta * dW1
        self.b1 -= self.eta * db1
        self.W2 -= self.eta * dW2
        self.b2 -= self.eta * db2
        return loss, dW1, dW2

    def step_rk4(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Un paso de Runge-Kutta 4 clásico.
        Returns: loss, gradW1, gradW2
        """
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

    def step(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Ejecuta un paso del método seleccionado.
        """
        if self.method == "Euler":
            return self.step_euler()
        elif self.method == "RK4":
            return self.step_rk4()
        else:
            # default to Euler if unknown
            return self.step_euler()

    def train(self) -> np.ndarray:
        """
        Entrena la red usando el método seleccionado.
        Returns: predicciones finales
        """
        self.loss_history = []
        for epoch in range(self.epochs):
            loss, _, _ = self.step()
            self.loss_history.append(float(loss))
        _, _, _, a2 = self.forward()
        return a2


# --------------------------
# Helpers for TensorFlow <-> flat vector conversion (for SciPy ODE mode)
# --------------------------
def get_flat_weights_from_keras(model) -> np.ndarray:
    """
    Devuelve el vector de pesos aplanado de un modelo Keras.
    """
    w_list = model.get_weights()
    flat = np.concatenate([w.flatten() for w in w_list]).astype(np.float64)
    return flat

def set_flat_weights_to_keras(model, flat: np.ndarray) -> None:
    """
    Asigna los pesos a un modelo Keras desde un vector plano.
    """
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
    """
    Interfaz gráfica para comparar modos de entrenamiento de la red XOR.
    """
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

        self.train_btn = QPushButton("Entrenar rápido")
        self.train_btn.clicked.connect(self.train_network)
        btn_layout.addWidget(self.train_btn)

        self.step_btn = QPushButton("Un Paso (solo Manual)")
        self.step_btn.clicked.connect(self.one_step)
        btn_layout.addWidget(self.step_btn)

        self.epoch_btn = QPushButton("Siguiente época (interactivo)")
        self.epoch_btn.clicked.connect(self.next_epoch)
        btn_layout.addWidget(self.epoch_btn)

        self.pause_btn = QPushButton("Pausa")
        self.pause_btn.clicked.connect(self.pause_training)
        btn_layout.addWidget(self.pause_btn)

        self.reset_btn = QPushButton("Resetear")
        self.reset_btn.clicked.connect(self.reset_network)
        btn_layout.addWidget(self.reset_btn)

        # Frecuencia de actualización de la visualización
        self.update_freq_label = QLabel("Frecuencia actualización gráfica:")
        self.update_freq_input = QLineEdit("100")
        btn_layout.addWidget(self.update_freq_label)
        btn_layout.addWidget(self.update_freq_input)

        layout.addLayout(btn_layout)

        self.results = QTextEdit()
        self.results.setReadOnly(True)
        layout.addWidget(self.results)

        # Plotting: curva de pérdida
        self.figure = Figure(figsize=(6,3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Plotting: visualización de la red y pesos
        self.nn_fig = Figure(figsize=(6,3))
        self.nn_canvas = FigureCanvas(self.nn_fig)
        layout.addWidget(self.nn_canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # internal state
        self.manual_nn = None
        self.keras_model = None
        self.keras_loss_history = []
        self.scipy_solution = None
        self.interactive_training = False
        self.current_epoch = 0
        self.max_epochs = 0
        self.training_paused = False
        self.reset_network()
        mode = self.mode_combo.currentText()
        if mode != "Manual":
            self.results.setText("El modo interactivo está disponible solo en Modo Manual.")
            return
        try:
            eta = float(self.lr_input.text())
            epochs = int(self.epochs_input.text())
        except ValueError:
            self.results.setText("Parámetros inválidos (η y épocas deben ser numéricos).")
            return
        method = self.method_combo.currentText()
        if self.manual_nn is None or self.max_epochs != epochs:
            self.manual_nn = SimpleNN(eta=eta, epochs=epochs, method=method)
            self.manual_nn.loss_history = []
            self.current_epoch = 0
            self.max_epochs = epochs
        if self.current_epoch < self.max_epochs:
            loss, _, _ = self.manual_nn.step()
            self.manual_nn.loss_history.append(float(loss))
            self.draw_nn_weights(self.manual_nn.W1, self.manual_nn.b1, self.manual_nn.W2, self.manual_nn.b2)
            self.current_epoch += 1
            txt = self.results.toPlainText()
            txt += f"\nÉpoca {self.current_epoch}/{self.max_epochs} - Loss: {loss:.8f}"
            self.results.setText(txt)
            # plot curva de pérdida
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.manual_nn.loss_history, label=f"Manual {method}")
            ax.set_title(f"Pérdida (interactivo)")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("MSE")
            ax.legend()
            ax.grid(True)
            self.canvas.draw()
        else:
            self.results.setText("Entrenamiento interactivo finalizado.")

    def pause_training(self):
    # Pausa el entrenamiento interactivo automático.
        self.training_paused = True
        self.results.setText("Entrenamiento pausado. Puedes continuar con 'Siguiente época'.")

    def next_epoch(self):
        # Ejecuta la siguiente época en modo interactivo (solo Manual)
        mode = self.mode_combo.currentText()
        if mode != "Manual":
            self.results.setText("El modo interactivo está disponible solo en Modo Manual.")
            return
        try:
            eta = float(self.lr_input.text())
            epochs = int(self.epochs_input.text())
        except ValueError:
            self.results.setText("Parámetros inválidos (η y épocas deben ser numéricos).")
            return
        method = self.method_combo.currentText()
        if self.manual_nn is None or self.max_epochs != epochs:
            self.manual_nn = SimpleNN(eta=eta, epochs=epochs, method=method)
            self.manual_nn.loss_history = []
            self.current_epoch = 0
            self.max_epochs = epochs
        if self.current_epoch < self.max_epochs:
            loss, _, _ = self.manual_nn.step()
            self.manual_nn.loss_history.append(float(loss))
            self.draw_nn_weights(self.manual_nn.W1, self.manual_nn.b1, self.manual_nn.W2, self.manual_nn.b2)
            self.current_epoch += 1
            txt = self.results.toPlainText()
            txt += f"\nÉpoca {self.current_epoch}/{self.max_epochs} - Loss: {loss:.8f}"
            self.results.setText(txt)
            # plot curva de pérdida
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.manual_nn.loss_history, label=f"Manual {method}")
            ax.set_title(f"Pérdida (interactivo)")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("MSE")
            ax.legend()
            ax.grid(True)
            self.canvas.draw()
        else:
            self.results.setText("Entrenamiento interactivo finalizado.")

    def draw_nn_weights(self, W1, b1, W2, b2, activations=None):
    # Dibuja la red neuronal y los pesos actuales.
        self.nn_fig.clear()
        ax = self.nn_fig.add_subplot(111)
        ax.axis('off')

        # Coordenadas de nodos
        x_in, y_in = [0.1]*2, [0.3, 0.7]
        x_hid, y_hid = [0.5]*3, [0.2, 0.5, 0.8]
        x_out, y_out = [0.9], [0.5]

        # Nodos
        for i in range(2):
            ax.plot(x_in[i], y_in[i], 'o', color='skyblue', markersize=18)
            ax.text(x_in[i]-0.05, y_in[i], f'X{i+1}', fontsize=12)
        for j in range(3):
            color = 'orange'
            if activations is not None and len(activations)>0:
                val = activations[0][j]
                color = plt.get_cmap('Reds')(val)
            ax.plot(x_hid[j], y_hid[j], 'o', color=color, markersize=18)
            ax.text(x_hid[j], y_hid[j]+0.07, f'H{j+1}', fontsize=12)
        ax.plot(x_out[0], y_out[0], 'o', color='lime', markersize=18)
        ax.text(x_out[0]+0.02, y_out[0], 'Y', fontsize=12)

        # Conexiones input-hidden
        for i in range(2):
            for j in range(3):
                w = W1[i,j]
                lw = 2 + 6*abs(w)/np.max(np.abs(W1)) if np.max(np.abs(W1))>0 else 2
                color = 'red' if w<0 else 'green'
                ax.plot([x_in[i], x_hid[j]], [y_in[i], y_hid[j]], '-', color=color, linewidth=lw, alpha=0.7)
                ax.text((x_in[i]+x_hid[j])/2, (y_in[i]+y_hid[j])/2, f'{w:.2f}', fontsize=8)

        # Conexiones hidden-output
        for j in range(3):
            w = W2[j,0]
            lw = 2 + 6*abs(w)/np.max(np.abs(W2)) if np.max(np.abs(W2))>0 else 2
            color = 'red' if w<0 else 'green'
            ax.plot([x_hid[j], x_out[0]], [y_hid[j], y_out[0]], '-', color=color, linewidth=lw, alpha=0.7)
            ax.text((x_hid[j]+x_out[0])/2, (y_hid[j]+y_out[0])/2, f'{w:.2f}', fontsize=8)

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        self.nn_canvas.draw()

    def train_network(self) -> None:
        # Optimización: desactiva botones durante entrenamiento
        self.train_btn.setEnabled(False)
        self.step_btn.setEnabled(False)
        self.epoch_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        mode = self.mode_combo.currentText()
        try:
            eta = float(self.lr_input.text())
            epochs = int(self.epochs_input.text())
            update_freq = int(self.update_freq_input.text())
        except ValueError:
            self.results.setText("Parámetros inválidos (η, épocas o frecuencia deben ser numéricos).")
            self.train_btn.setEnabled(True)
            self.step_btn.setEnabled(True)
            self.epoch_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            return

        if mode == "Manual":
            method = self.method_combo.currentText()
            self.manual_nn = SimpleNN(eta=eta, epochs=epochs, method=method)
            self.manual_nn.loss_history = []
            self.current_epoch = 0
            self.max_epochs = epochs
            self.training_paused = False
            # Entrenamiento en bloques para eficiencia
            while self.current_epoch < self.max_epochs and not self.training_paused:
                block = min(update_freq, self.max_epochs - self.current_epoch)
                for _ in range(block):
                    loss, _, _ = self.manual_nn.step()
                    self.manual_nn.loss_history.append(float(loss))
                    self.current_epoch += 1
                self.draw_nn_weights(self.manual_nn.W1, self.manual_nn.b1, self.manual_nn.W2, self.manual_nn.b2)
                self.results.append(f"Época {self.current_epoch}/{self.max_epochs} - Loss: {loss:.8f}")
                # plot curva de pérdida (solo últimas N)
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                N = 200
                ax.plot(self.manual_nn.loss_history[-N:], label=f"Manual {method}")
                ax.set_title(f"Pérdida (interactivo)")
                ax.set_xlabel("Épocas")
                ax.set_ylabel("MSE")
                ax.legend()
                ax.grid(True)
                self.canvas.draw()
            outputs = self.manual_nn.forward()[3]
            preds = (outputs > 0.5).astype(int)
            self.results.append(f"Manual mode ({method}) - Resultados finales:")
            for i in range(len(self.manual_nn.X)):
                self.results.append(f"{self.manual_nn.X[i]} -> pred: {int(preds[i][0])} (real {int(self.manual_nn.y[i][0])}) [{outputs[i][0]:.3f}]")
            self.results.append(f"Última pérdida: {self.manual_nn.loss_history[-1]:.8f}")
        # Reactiva botones
        self.train_btn.setEnabled(True)
        self.step_btn.setEnabled(True)
        self.epoch_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        if mode == "Keras":
            # Visualización de pesos no implementada para Keras por diferencias de API
            if not TF_OK or tf is None:
                self.results.setText("TensorFlow / Keras no está disponible en este entorno.")
                return
            try:
                # Defensive: check keras attribute
                keras_mod = getattr(tf, "keras", None)
                if keras_mod is None:
                    self.results.setText("TensorFlow instalado, pero 'keras' no está disponible.")
                    return
                optimizers_mod = getattr(keras_mod, "optimizers", None)
                if optimizers_mod is None:
                    self.results.setText("TensorFlow instalado, pero 'keras.optimizers' no está disponible.")
                    return
                model = Sequential([Dense(3, input_shape=(2,), activation='sigmoid'), Dense(1, activation='sigmoid')])
                opt = optimizers_mod.SGD(learning_rate=eta)
                model.compile(optimizer=opt, loss='mse', metrics=[])
                history = model.fit(np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[0],[1],[1],[0]]),
                                    epochs=epochs, verbose='auto')
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
            except Exception as ex:
                self.results.setText(f"Error ejecutando Keras: {ex}")
        elif mode == "SciPy ODE":
            # Visualización de pesos no implementada para SciPy ODE por diferencias de API
            if not (TF_OK and SCIPY_OK) or tf is None or solve_ivp is None:
                self.results.setText("Para SciPy ODE necesitas: scipy y tensorflow instalados.")
                return
            try:
                keras_mod = getattr(tf, "keras", None)
                if keras_mod is None:
                    self.results.setText("TensorFlow instalado, pero 'keras' no está disponible.")
                    return
                model = Sequential([Dense(3, input_shape=(2,), activation='sigmoid'), Dense(1, activation='sigmoid')])
                flat0 = get_flat_weights_from_keras(model)
                X_train = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
                y_train = np.array([[0],[1],[1],[0]], dtype=np.float64)

                def ode_fun(t, y_flat):
                    set_flat_weights_to_keras(model, y_flat.astype(np.float32))
                    GradientTape = getattr(tf, "GradientTape", None)
                    convert_to_tensor = getattr(tf, "convert_to_tensor", None)
                    reduce_mean = getattr(tf, "reduce_mean", None)
                    square = getattr(tf, "square", None)
                    if GradientTape is None or convert_to_tensor is None or reduce_mean is None or square is None:
                        raise RuntimeError("TensorFlow no tiene los métodos necesarios para SciPy ODE.")
                    with GradientTape() as tape:
                        x_tf = convert_to_tensor(X_train, dtype=getattr(tf, "float32", None))
                        y_tf = convert_to_tensor(y_train, dtype=getattr(tf, "float32", None))
                        preds = model(x_tf, training=False)
                        loss_tf = reduce_mean(square(y_tf - preds))
                    grads = tape.gradient(loss_tf, model.trainable_weights)
                    if grads is None:
                        raise RuntimeError("No se pudieron calcular los gradientes con TensorFlow.")
                    grad_flat = np.concatenate([g.numpy().flatten() for g in grads]).astype(np.float64)
                    return (-eta * grad_flat)

                t_span = (0.0, float(epochs))
                sol = solve_ivp(ode_fun, t_span, flat0, method='RK45', rtol=1e-6, atol=1e-9, max_step=1.0)
                self.scipy_solution = sol

                final_flat = sol.y[:, -1]
                set_flat_weights_to_keras(model, final_flat.astype(np.float32))
                preds = (model.predict(X_train) > 0.5).astype(int)
                losses = []
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

                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.plot(sol.t, losses, label='Loss (SciPy solve_ivp)')
                ax.set_title('Pérdida (SciPy ODE integration)')
                ax.set_xlabel('Tiempo (epocas equivalentes)')
                ax.set_ylabel('MSE')
                ax.legend()
                ax.grid(True)
                self.canvas.draw()
            except Exception as ex:
                self.results.setText(f"Error ejecutando SciPy ODE: {ex}")
        else:
            self.results.setText("Modo no reconocido.")

    def one_step(self) -> None:
    # Ejecuta un paso manual (solo modo Manual).
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
    # Reinicia el estado interno y la interfaz.
        # reset all internal objects
        self.manual_nn = None
        self.keras_model = None
        self.keras_loss_history = []
        self.scipy_solution = None
        self.interactive_training = False
        self.current_epoch = 0
        self.max_epochs = 0
        self.training_paused = False
        self.results.clear()
        self.figure.clear()
        self.canvas.draw()
        self.results.setText("Estado reiniciado. Selecciona modo y parámetros y presiona 'Entrenar / Ejecutar'.")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # Main entry point
    app = QApplication(sys.argv)
    window = NNApp()
    window.show()
    sys.exit(app.exec_())
