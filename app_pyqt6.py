import sys
import numpy as np
from typing import List
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget, QTextEdit, QSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# --- SimpleNN copied from Streamlit app (same logic) ---
class SimpleNN:
    def __init__(self, eta: float = 0.1, epochs: int = 2000, method: str = "Euler") -> None:
        self.X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        self.y = np.array([[0],[1],[1],[0]], dtype=float)
        self.eta = float(eta)
        self.epochs = int(epochs)
        self.method = method
        np.random.seed(42)
        # Architecture: 2 -> 3 -> 3 -> 3 -> 1
        self.W1 = np.random.randn(2, 3)
        self.b1 = np.zeros((1, 3))
        self.W2 = np.random.randn(3, 3)
        self.b2 = np.zeros((1, 3))
        self.W3 = np.random.randn(3, 3)
        self.b3 = np.zeros((1, 3))
        self.W4 = np.random.randn(3, 1)
        self.b4 = np.zeros((1, 1))
        self.loss_history: List[float] = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(z):
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return s * (1 - s)

    def forward(self):
        z1 = self.X @ self.W1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)
        z3 = a2 @ self.W3 + self.b3
        a3 = self._sigmoid(z3)
        z4 = a3 @ self.W4 + self.b4
        a4 = self._sigmoid(z4)
        return z1, a1, z2, a2, z3, a3, z4, a4

    def compute_gradients(self):
        z1, a1, z2, a2, z3, a3, z4, a4 = self.forward()
        loss = np.mean((self.y - a4) ** 2)

        # Output layer
        dL_da4 = 2 * (a4 - self.y) / self.y.shape[0]
        da4_dz4 = self._sigmoid_derivative(z4)
        dL_dz4 = dL_da4 * da4_dz4
        dL_dW4 = a3.T @ dL_dz4
        dL_db4 = np.sum(dL_dz4, axis=0, keepdims=True)

        # Layer 3
        dL_da3 = dL_dz4 @ self.W4.T
        da3_dz3 = self._sigmoid_derivative(z3)
        dL_dz3 = dL_da3 * da3_dz3
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

        # Layer 2
        dL_da2 = dL_dz3 @ self.W3.T
        da2_dz2 = self._sigmoid_derivative(z2)
        dL_dz2 = dL_da2 * da2_dz2
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        # Layer 1
        dL_da1 = dL_dz2 @ self.W2.T
        da1_dz1 = self._sigmoid_derivative(z1)
        dL_dz1 = dL_da1 * da1_dz1
        dL_dW1 = self.X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        return float(loss), (dL_dW1, dL_db1, dL_dW2, dL_db2, dL_dW3, dL_db3, dL_dW4, dL_db4)

    def step_euler(self):
        loss, grads = self.compute_gradients()
        dW1, db1, dW2, db2, dW3, db3, dW4, db4 = grads
        self.W1 -= self.eta * dW1
        self.b1 -= self.eta * db1
        self.W2 -= self.eta * dW2
        self.b2 -= self.eta * db2
        self.W3 -= self.eta * dW3
        self.b3 -= self.eta * db3
        self.W4 -= self.eta * dW4
        self.b4 -= self.eta * db4
        return loss, dW1, dW4

    def step_rk4(self):
        loss1, grads1 = self.compute_gradients()
        dW1_1, db1_1, dW2_1, db2_1, dW3_1, db3_1, dW4_1, db4_1 = grads1
        W1_orig, b1_orig = self.W1.copy(), self.b1.copy()
        W2_orig, b2_orig = self.W2.copy(), self.b2.copy()
        W3_orig, b3_orig = self.W3.copy(), self.b3.copy()
        W4_orig, b4_orig = self.W4.copy(), self.b4.copy()

        # k2
        self.W1 = W1_orig - 0.5 * self.eta * dW1_1
        self.b1 = b1_orig - 0.5 * self.eta * db1_1
        self.W2 = W2_orig - 0.5 * self.eta * dW2_1
        self.b2 = b2_orig - 0.5 * self.eta * db2_1
        self.W3 = W3_orig - 0.5 * self.eta * dW3_1
        self.b3 = b3_orig - 0.5 * self.eta * db3_1
        self.W4 = W4_orig - 0.5 * self.eta * dW4_1
        self.b4 = b4_orig - 0.5 * self.eta * db4_1
        _, grads2 = self.compute_gradients()
        dW1_2, db1_2, dW2_2, db2_2, dW3_2, db3_2, dW4_2, db4_2 = grads2

        # k3
        self.W1 = W1_orig - 0.5 * self.eta * dW1_2
        self.b1 = b1_orig - 0.5 * self.eta * db1_2
        self.W2 = W2_orig - 0.5 * self.eta * dW2_2
        self.b2 = b2_orig - 0.5 * self.eta * db2_2
        self.W3 = W3_orig - 0.5 * self.eta * dW3_2
        self.b3 = b3_orig - 0.5 * self.eta * db3_2
        self.W4 = W4_orig - 0.5 * self.eta * dW4_2
        self.b4 = b4_orig - 0.5 * self.eta * db4_2
        _, grads3 = self.compute_gradients()
        dW1_3, db1_3, dW2_3, db2_3, dW3_3, db3_3, dW4_3, db4_3 = grads3

        # k4
        self.W1 = W1_orig - self.eta * dW1_3
        self.b1 = b1_orig - self.eta * db1_3
        self.W2 = W2_orig - self.eta * dW2_3
        self.b2 = b2_orig - self.eta * db2_3
        self.W3 = W3_orig - self.eta * dW3_3
        self.b3 = b3_orig - self.eta * db3_3
        self.W4 = W4_orig - self.eta * dW4_3
        self.b4 = b4_orig - self.eta * db4_3
        _, grads4 = self.compute_gradients()
        dW1_4, db1_4, dW2_4, db2_4, dW3_4, db3_4, dW4_4, db4_4 = grads4

        # restore originals
        self.W1, self.b1 = W1_orig, b1_orig
        self.W2, self.b2 = W2_orig, b2_orig
        self.W3, self.b3 = W3_orig, b3_orig
        self.W4, self.b4 = W4_orig, b4_orig

        # update with weighted average
        self.W1 -= (self.eta/6.0) * (dW1_1 + 2*dW1_2 + 2*dW1_3 + dW1_4)
        self.b1 -= (self.eta/6.0) * (db1_1 + 2*db1_2 + 2*db1_3 + db1_4)
        self.W2 -= (self.eta/6.0) * (dW2_1 + 2*dW2_2 + 2*dW2_3 + dW2_4)
        self.b2 -= (self.eta/6.0) * (db2_1 + 2*db2_2 + 2*db2_3 + db2_4)
        self.W3 -= (self.eta/6.0) * (dW3_1 + 2*dW3_2 + 2*dW3_3 + dW3_4)
        self.b3 -= (self.eta/6.0) * (db3_1 + 2*db3_2 + 2*db3_3 + db3_4)
        self.W4 -= (self.eta/6.0) * (dW4_1 + 2*dW4_2 + 2*dW4_3 + dW4_4)
        self.b4 -= (self.eta/6.0) * (db4_1 + 2*db4_2 + 2*db4_3 + db4_4)
        return loss1, dW1_1, dW4_1

    def step(self):
        if self.method == "Euler":
            return self.step_euler()
        elif self.method == "RK4":
            return self.step_rk4()
        else:
            return self.step_euler()

    def train(self):
        self.loss_history = []
        for epoch in range(self.epochs):
            loss, _, _ = self.step()
            self.loss_history.append(float(loss))
        # forward now returns (z1,a1,z2,a2,z3,a3,z4,a4)
        *_, a4 = self.forward()
        return a4


# Worker thread to run training without blocking UI
class TrainWorker(QThread):
    # epoch, loss, loss_history, weights_tuple
    progress = pyqtSignal(int, float, list, object)
    finished = pyqtSignal(object, object)   # nn, outputs

    def __init__(self, eta, epochs, method, update_freq=100):
        super().__init__()
        self.eta = eta
        self.epochs = epochs
        self.method = method
        self.update_freq = update_freq
        self._stop = False

    def run(self):
        nn = SimpleNN(eta=self.eta, epochs=self.epochs, method=self.method)
        N = self.update_freq
        for i in range(0, self.epochs, self.update_freq):
            if self._stop:
                break
            block = min(self.update_freq, self.epochs - i)
            for _ in range(block):
                loss, _, _ = nn.step()
                nn.loss_history.append(float(loss))
            epoch = min(i + block, self.epochs)
            weights_snapshot = (
                nn.W1.copy(), nn.b1.copy(),
                nn.W2.copy(), nn.b2.copy(),
                nn.W3.copy(), nn.b3.copy(),
                nn.W4.copy(), nn.b4.copy()
            )
            self.progress.emit(epoch, float(loss), nn.loss_history.copy(), weights_snapshot)
        outputs = nn.forward()[-1]
        self.finished.emit(nn, outputs)

    def stop(self):
        self._stop = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Red Neuronal XOR - PyQt6")
        self.resize(1100, 700)

        container = QWidget()
        main_layout = QHBoxLayout()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Left controls
        left = QWidget()
        left_layout = QVBoxLayout()
        left.setLayout(left_layout)
        left_layout.addWidget(QLabel("Parámetros de entrenamiento"))

        left_layout.addWidget(QLabel("Tasa de aprendizaje (η):"))
        self.eta_input = QLineEdit("0.1")
        left_layout.addWidget(self.eta_input)

        left_layout.addWidget(QLabel("Épocas:"))
        self.epochs_input = QLineEdit("5000")
        left_layout.addWidget(self.epochs_input)

        left_layout.addWidget(QLabel("Método de integración:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Euler", "RK4"])
        left_layout.addWidget(self.method_combo)

        left_layout.addWidget(QLabel("Frecuencia actualización gráfica:"))
        self.update_freq_input = QLineEdit("100")
        left_layout.addWidget(self.update_freq_input)

        self.run_btn = QPushButton("Entrenar red")
        self.run_btn.clicked.connect(self.start_training)
        left_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Detener")
        self.stop_btn.clicked.connect(self.stop_training)
        left_layout.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("Resetear")
        self.reset_btn.clicked.connect(self.reset_state)
        left_layout.addWidget(self.reset_btn)

        left_layout.addStretch()

        # Right area: tabs
        right = QWidget()
        right_layout = QVBoxLayout()
        right.setLayout(right_layout)

        self.tabs = QTabWidget()

        # Loss tab
        self.loss_fig = Figure(figsize=(5,3))
        self.loss_canvas = FigureCanvas(self.loss_fig)
        loss_tab = QWidget()
        loss_layout = QVBoxLayout()
        loss_tab.setLayout(loss_layout)
        loss_layout.addWidget(self.loss_canvas)
        self.tabs.addTab(loss_tab, "Curva de pérdida")

        # Results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout()
        results_tab.setLayout(results_layout)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        self.tabs.addTab(results_tab, "Resultados finales")

        # Network tab
        self.net_fig = Figure(figsize=(5,3))
        self.net_canvas = FigureCanvas(self.net_fig)
        net_tab = QWidget()
        net_layout = QHBoxLayout()
        net_tab.setLayout(net_layout)
        # Left: network drawing
        left_net_w = QWidget()
        left_net_layout = QVBoxLayout()
        left_net_w.setLayout(left_net_layout)
        left_net_layout.addWidget(self.net_canvas)
        net_layout.addWidget(left_net_w, 2)
        # Right: weights/text
        self.weights_text = QTextEdit()
        self.weights_text.setReadOnly(True)
        self.weights_text.setMinimumWidth(280)
        net_layout.addWidget(self.weights_text, 1)
        self.tabs.addTab(net_tab, "Red y pesos")

        right_layout.addWidget(self.tabs)

        main_layout.addWidget(left, 1)
        main_layout.addWidget(right, 3)

        # internal
        self.worker = None
        self.nn = None

    def start_training(self):
        try:
            eta = float(self.eta_input.text())
            epochs = int(self.epochs_input.text())
            update_freq = int(self.update_freq_input.text())
        except ValueError:
            self.results_text.append("Parámetros inválidos.")
            return
        method = self.method_combo.currentText()
        self.results_text.append(f"Iniciando entrenamiento: método={method}, eta={eta}, epochs={epochs}\n")
        # disable buttons
        self.run_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

        self.worker = TrainWorker(eta=eta, epochs=epochs, method=method, update_freq=update_freq)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, epoch, loss, loss_history, weights):
        # update loss plot and status, and update network drawing/weights live
        self.update_loss_plot(loss_history)
        self.results_text.append(f"Época {epoch} - Loss: {loss:.8f}")
        try:
            W1, b1, W2, b2, W3, b3, W4, b4 = weights
            self.update_net_plot(W1, b1, W2, b2, W3, b3, W4, b4)
        except Exception:
            pass

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.results_text.append("Deteniendo entrenamiento...")

    def reset_state(self):
        self.nn = None
        self.results_text.clear()
        self.clear_loss_plot()
        self.clear_net_plot()
        try:
            self.weights_text.clear()
        except Exception:
            pass


    def on_finished(self, nn, outputs):
        self.nn = nn
        preds = (outputs > 0.5).astype(int)
        self.results_text.append("Entrenamiento finalizado. Resultados:")
        for i in range(len(nn.X)):
            self.results_text.append(f"{nn.X[i]} -> pred: {int(preds[i,0])} (real {int(nn.y[i,0])}) [{outputs[i,0]:.3f}]")
        self.update_loss_plot(nn.loss_history)
        self.update_net_plot(nn.W1, nn.b1, nn.W2, nn.b2, nn.W3, nn.b3, nn.W4, nn.b4)
        self.run_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

    def update_loss_plot(self, loss_history):
        self.loss_fig.clear()
        ax = self.loss_fig.add_subplot(111)
        ax.plot(loss_history, label='Loss')
        ax.set_title('Curva de pérdida')
        ax.set_xlabel('Épocas')
        ax.set_ylabel('MSE')
        ax.grid(True)
        self.loss_canvas.draw()

    def clear_loss_plot(self):
        self.loss_fig.clear()
        self.loss_canvas.draw()

    def clear_net_plot(self):
        self.net_fig.clear()
        self.net_canvas.draw()

    def update_net_plot(self, W1, b1, W2, b2, W3, b3, W4, b4):
        self.net_fig.clear()
        ax = self.net_fig.add_subplot(111)
        ax.axis('off')
        x_in, y_in = [0.05]*2, [0.35, 0.65]
        x_h1, y_h1 = [0.35]*3, [0.2, 0.5, 0.8]
        x_h2, y_h2 = [0.5]*3, [0.2, 0.5, 0.8]
        x_h3, y_h3 = [0.65]*3, [0.2, 0.5, 0.8]
        x_out, y_out = [0.95], [0.5]
        for i in range(2):
            ax.plot(x_in[i], y_in[i], 'o', color='skyblue', markersize=18)
            ax.text(x_in[i]-0.05, y_in[i], f'X{i+1}', fontsize=12)
        for j in range(3):
            ax.plot(x_h1[j], y_h1[j], 'o', color='orange', markersize=18)
            ax.text(x_h1[j], y_h1[j]+0.07, f'H1-{j+1}', fontsize=10)
            ax.plot(x_h2[j], y_h2[j], 'o', color='orange', markersize=18)
            ax.text(x_h2[j], y_h2[j]+0.07, f'H2-{j+1}', fontsize=10)
            ax.plot(x_h3[j], y_h3[j], 'o', color='orange', markersize=18)
            ax.text(x_h3[j], y_h3[j]+0.07, f'H3-{j+1}', fontsize=10)
        ax.plot(x_out[0], y_out[0], 'o', color='lime', markersize=18)
        ax.text(x_out[0]+0.02, y_out[0], 'Y', fontsize=12)
        # Input -> H1
        for i in range(2):
            for j in range(3):
                w = W1[i,j]
                lw = 2 + 6*abs(w)/np.max(np.abs(W1)) if np.max(np.abs(W1))>0 else 2
                color = 'red' if w<0 else 'green'
                ax.plot([x_in[i], x_h1[j]], [y_in[i], y_h1[j]], '-', color=color, linewidth=lw, alpha=0.7)
                ax.text((x_in[i]+x_h1[j])/2, (y_in[i]+y_h1[j])/2, f'{w:.2f}', fontsize=8)

        # H1 -> H2
        for i in range(3):
            for j in range(3):
                w = W2[i,j]
                lw = 2 + 6*abs(w)/np.max(np.abs(W2)) if np.max(np.abs(W2))>0 else 2
                color = 'red' if w<0 else 'green'
                ax.plot([x_h1[i], x_h2[j]], [y_h1[i], y_h2[j]], '-', color=color, linewidth=lw, alpha=0.7)
                ax.text((x_h1[i]+x_h2[j])/2, (y_h1[i]+y_h2[j])/2, f'{w:.2f}', fontsize=8)

        # H2 -> H3
        for i in range(3):
            for j in range(3):
                w = W3[i,j]
                lw = 2 + 6*abs(w)/np.max(np.abs(W3)) if np.max(np.abs(W3))>0 else 2
                color = 'red' if w<0 else 'green'
                ax.plot([x_h2[i], x_h3[j]], [y_h2[i], y_h3[j]], '-', color=color, linewidth=lw, alpha=0.7)
                ax.text((x_h2[i]+x_h3[j])/2, (y_h2[i]+y_h3[j])/2, f'{w:.2f}', fontsize=8)

        # H3 -> Out
        for j in range(3):
            w = W4[j,0]
            lw = 2 + 6*abs(w)/np.max(np.abs(W4)) if np.max(np.abs(W4))>0 else 2
            color = 'red' if w<0 else 'green'
            ax.plot([x_h3[j], x_out[0]], [y_h3[j], y_out[0]], '-', color=color, linewidth=lw, alpha=0.7)
            ax.text((x_h3[j]+x_out[0])/2, (y_h3[j]+y_out[0])/2, f'{w:.2f}', fontsize=8)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        self.net_canvas.draw()

        # Prepare a labeled textual view of weights and biases
        lines = []
        def fmt(arr):
            return np.array2string(arr, precision=4, floatmode='fixed')

        lines.append('Layer W1 (input -> H1)')
        for j in range(W1.shape[1]):
            lines.append(f'  H1-{j+1}: weights <- {fmt(W1[:, j])}, bias = {b1[0, j]:.6f}')

        lines.append('\nLayer W2 (H1 -> H2)')
        for j in range(W2.shape[1]):
            lines.append(f'  H2-{j+1}: weights <- {fmt(W2[:, j])}, bias = {b2[0, j]:.6f}')

        lines.append('\nLayer W3 (H2 -> H3)')
        for j in range(W3.shape[1]):
            lines.append(f'  H3-{j+1}: weights <- {fmt(W3[:, j])}, bias = {b3[0, j]:.6f}')

        lines.append('\nLayer W4 (H3 -> Out)')
        for j in range(W4.shape[1]):
            lines.append(f'  Out: weights <- {fmt(W4[:, j])}, bias = {b4[0, j]:.6f}')

        try:
            self.weights_text.setPlainText('\n'.join(lines))
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
