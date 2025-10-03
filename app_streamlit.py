import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# CSS personalizado para mejorar apariencia
def local_css():
    st.markdown("""
        <style>
        .main-title {color: #1f77b4; font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem;}
        .sidebar .sidebar-content {background: #f0f4f8;}
        .result-box {background: #eaf6ff; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;}
        .stTabs [data-baseweb="tab"] {background: #eaf6ff; border-radius: 10px 10px 0 0;}
        </style>
    """, unsafe_allow_html=True)

local_css()

# --------------------------
# Implementación manual (SimpleNN) - métodos numéricos hechos a mano
# --------------------------
class SimpleNN:
    """Red neuronal simple para XOR con métodos de integración manuales (Euler, RK4)."""
    def __init__(self, eta: float = 0.1, epochs: int = 2000, method: str = "Euler") -> None:
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
        self.loss_history = []

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
        loss1, grads1 = self.compute_gradients()
        dW1_1, db1_1, dW2_1, db2_1 = grads1
        W1_orig, b1_orig, W2_orig, b2_orig = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()
        self.W1 = W1_orig - 0.5 * self.eta * dW1_1
        self.b1 = b1_orig - 0.5 * self.eta * db1_1
        self.W2 = W2_orig - 0.5 * self.eta * dW2_1
        self.b2 = b2_orig - 0.5 * self.eta * db2_1
        _, grads2 = self.compute_gradients()
        dW1_2, db1_2, dW2_2, db2_2 = grads2
        self.W1 = W1_orig - 0.5 * self.eta * dW1_2
        self.b1 = b1_orig - 0.5 * self.eta * db1_2
        self.W2 = W2_orig - 0.5 * self.eta * dW2_2
        self.b2 = b2_orig - 0.5 * self.eta * db2_2
        _, grads3 = self.compute_gradients()
        dW1_3, db1_3, dW2_3, db2_3 = grads3
        self.W1 = W1_orig - self.eta * dW1_3
        self.b1 = b1_orig - self.eta * db1_3
        self.W2 = W2_orig - self.eta * dW2_3
        self.b2 = b2_orig - self.eta * db2_3
        _, grads4 = self.compute_gradients()
        dW1_4, db1_4, dW2_4, db2_4 = grads4
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
            return self.step_euler()

    def train(self):
        self.loss_history = []
        for epoch in range(self.epochs):
            loss, _, _ = self.step()
            self.loss_history.append(float(loss))
        _, _, _, a2 = self.forward()
        return a2

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Red Neuronal XOR + EDOs", layout="wide")
st.markdown('<div class="main-title">Red Neuronal XOR con Métodos Numéricos (Euler/RK4)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1,2])
with col1:
    st.header("Parámetros de entrenamiento")
    eta = st.number_input("Tasa de aprendizaje (η)", min_value=0.001, max_value=2.0, value=0.1, step=0.01)
    epochs = st.number_input("Épocas", min_value=1, max_value=20000, value=5000, step=100)
    method = st.selectbox("Método de integración", ["Euler", "RK4"])
    update_freq = st.number_input("Frecuencia de actualización gráfica", min_value=1, max_value=1000, value=100, step=1)
    run_btn = st.button("Entrenar red")
    reset_btn = st.button("Resetear red")

if 'nn' not in st.session_state or reset_btn:
    st.session_state.nn = SimpleNN(eta=eta, epochs=epochs, method=method)
    st.session_state.loss_history = []
    st.session_state.current_epoch = 0
    st.session_state.outputs = None

with col2:
    tabs = st.tabs(["Curva de pérdida", "Resultados finales", "Red y pesos"])
    if run_btn:
        nn = SimpleNN(eta=eta, epochs=epochs, method=method)
        N = 200
        for i in range(0, epochs, update_freq):
            block = min(update_freq, epochs - i)
            for _ in range(block):
                loss, _, _ = nn.step()
                nn.loss_history.append(float(loss))
        with tabs[0]:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.plot(nn.loss_history[-N:], label=f"{method}")
            ax.set_title("Curva de pérdida (últimas N)")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("MSE")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        outputs = nn.forward()[3]
        preds = (outputs > 0.5).astype(int)
        with tabs[1]:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("Resultados finales")
            for i in range(len(nn.X)):
                st.write(f"{nn.X[i]} → pred: {int(preds[i][0])} (real {int(nn.y[i][0])}) [{outputs[i][0]:.3f}]")
            st.write(f"Última pérdida: {nn.loss_history[-1]:.8f}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.nn = nn
        st.session_state.loss_history = nn.loss_history
        st.session_state.outputs = outputs
    if st.session_state.nn:
        nn = st.session_state.nn
        W1, b1, W2, b2 = nn.W1, nn.b1, nn.W2, nn.b2
        with tabs[2]:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.axis('off')
            x_in, y_in = [0.1]*2, [0.3, 0.7]
            x_hid, y_hid = [0.5]*3, [0.2, 0.5, 0.8]
            x_out, y_out = [0.9], [0.5]
            for i in range(2):
                ax.plot(x_in[i], y_in[i], 'o', color='skyblue', markersize=18)
                ax.text(x_in[i]-0.05, y_in[i], f'X{i+1}', fontsize=12)
            for j in range(3):
                ax.plot(x_hid[j], y_hid[j], 'o', color='orange', markersize=18)
                ax.text(x_hid[j], y_hid[j]+0.07, f'H{j+1}', fontsize=12)
            ax.plot(x_out[0], y_out[0], 'o', color='lime', markersize=18)
            ax.text(x_out[0]+0.02, y_out[0], 'Y', fontsize=12)
            for i in range(2):
                for j in range(3):
                    w = W1[i,j]
                    lw = 2 + 6*abs(w)/np.max(np.abs(W1)) if np.max(np.abs(W1))>0 else 2
                    color = 'red' if w<0 else 'green'
                    ax.plot([x_in[i], x_hid[j]], [y_in[i], y_hid[j]], '-', color=color, linewidth=lw, alpha=0.7)
                    ax.text((x_in[i]+x_hid[j])/2, (y_in[i]+y_hid[j])/2, f'{w:.2f}', fontsize=8)
            for j in range(3):
                w = W2[j,0]
                lw = 2 + 6*abs(w)/np.max(np.abs(W2)) if np.max(np.abs(W2))>0 else 2
                color = 'red' if w<0 else 'green'
                ax.plot([x_hid[j], x_out[0]], [y_hid[j], y_out[0]], '-', color=color, linewidth=lw, alpha=0.7)
                ax.text((x_hid[j]+x_out[0])/2, (y_hid[j]+y_out[0])/2, f'{w:.2f}', fontsize=8)
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
