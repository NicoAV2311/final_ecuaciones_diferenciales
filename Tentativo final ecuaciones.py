"""
Red Neuronal para XOR con Múltiples Métodos de Integración Numérica
===================================================================

Este módulo implementa una red neuronal simple que resuelve el problema XOR
utilizando diferentes métodos de integración numérica para la optimización de pesos.
El proyecto combina conceptos de ecuaciones diferenciales ordinarias (EDO) con 
aprendizaje automático, permitiendo comparar la efectividad de distintos métodos.

Métodos de integración disponibles:
- Euler Explícito: Método de primer orden, simple pero menos preciso
- Runge-Kutta 2 (RK2): Método de segundo orden, balance entre precisión y costo
- Runge-Kutta 4 (RK4): Método de cuarto orden, alta precisión

Funcionalidades principales:
- Red neuronal de 2 capas para resolver XOR
- Entrenamiento usando múltiples métodos de integración numérica
- Interfaz gráfica interactiva con PyQt5
- Selección dinámica del método de integración
- Visualización comparativa de curvas de pérdida
- Modo paso a paso para observar diferencias entre métodos
- Análisis de convergencia y estabilidad numérica

Autor: [Tu nombre]
Fecha: Septiembre 2025
Materia: Ecuaciones Diferenciales
"""

import sys
from typing import Tuple, List, Optional
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QTextEdit, QLineEdit, QHBoxLayout, QComboBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# --------------------------
# Red neuronal XOR con entrenamiento por Euler
# --------------------------

class SimpleNN:
    """
    Red Neuronal con Múltiples Métodos de Integración Numérica para XOR.
    
    Esta clase implementa una red neuronal feedforward de 2 capas que utiliza
    diferentes métodos de integración numérica para la actualización de pesos.
    Todos los métodos tratan la optimización como un sistema de EDO:
    
    dW/dt = -η * ∇L(W)
    
    Métodos disponibles:
    - Euler: W(t+h) = W(t) + h * f(t, W(t))
    - RK2: Método de punto medio con dos evaluaciones
    - RK4: Método clásico de cuarto orden con cuatro evaluaciones
    
    Donde W son los pesos, η es la tasa de aprendizaje, h es el paso de 
    integración y ∇L es el gradiente de la función de pérdida.
    
    Attributes:
        X (np.ndarray): Datos de entrada del problema XOR (4x2)
        y (np.ndarray): Etiquetas objetivo del problema XOR (4x1)
        eta (float): Tasa de aprendizaje (paso de integración)
        epochs (int): Número de épocas de entrenamiento
        method (str): Método de integración ('Euler', 'RK2', 'RK4')
        W1 (np.ndarray): Pesos de la capa oculta (2x3)
        b1 (np.ndarray): Sesgos de la capa oculta (1x3)
        W2 (np.ndarray): Pesos de la capa de salida (3x1)
        b2 (np.ndarray): Sesgos de la capa de salida (1x1)
        loss_history (List[float]): Historia de la función de pérdida
    """
    
    def __init__(self, eta: float = 0.1, epochs: int = 2000, method: str = "Euler") -> None:
        """
        Inicializa la red neuronal con parámetros y datos del problema XOR.
        
        Args:
            eta (float, optional): Tasa de aprendizaje para la integración numérica.
                                 Controla el tamaño del paso en la actualización
                                 de pesos. Por defecto 0.1.
            epochs (int, optional): Número de iteraciones de entrenamiento.
                                  Por defecto 2000.
            method (str, optional): Método de integración numérica a utilizar.
                                  Opciones: 'Euler', 'RK2', 'RK4'.
                                  Por defecto 'Euler'.
        
        Raises:
            ValueError: Si el método especificado no está soportado.
        
        Note:
            Los pesos se inicializan aleatoriamente usando semilla 42 para
            reproducibilidad. La arquitectura es 2-3-1 (entrada-oculta-salida).
        """
        # Datos del problema XOR
        # Matriz de entrada: cada fila representa una combinación de bits
        self.X = np.array([
            [0, 0],  # 0 XOR 0 = 0
            [0, 1],  # 0 XOR 1 = 1
            [1, 0],  # 1 XOR 0 = 1
            [1, 1]   # 1 XOR 1 = 0
        ])
        # Vector objetivo: resultados esperados del XOR
        self.y = np.array([[0], [1], [1], [0]])
        
        # Hiperparámetros del algoritmo de integración numérica
        self.eta = eta        # Tasa de aprendizaje (tamaño del paso)
        self.epochs = epochs  # Número de iteraciones
        self.method = method  # Método de integración elegido
        
        # Inicialización de pesos y sesgos
        # Semilla fija para reproducibilidad de resultados
        np.random.seed(42)
        
        # Capa oculta: 2 entradas -> 3 neuronas ocultas
        self.W1 = np.random.randn(2, 3)  # Pesos entrada-oculta
        self.b1 = np.zeros((1, 3))       # Sesgos capa oculta
        
        # Capa de salida: 3 neuronas ocultas -> 1 salida
        self.W2 = np.random.randn(3, 1)  # Pesos oculta-salida
        self.b2 = np.zeros((1, 1))       # Sesgo capa salida

        # Lista para almacenar el historial de pérdida durante entrenamiento
        self.loss_history: List[float] = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Función de activación sigmoide.
        
        La función sigmoide mapea cualquier número real al rango (0,1),
        siendo útil para problemas de clasificación binaria.
        
        Fórmula: σ(z) = 1 / (1 + e^(-z))
        
        Args:
            z (np.ndarray): Valores de entrada pre-activación
            
        Returns:
            np.ndarray: Valores activados en el rango (0,1)
            
        Note:
            Se usa clip para evitar overflow en exponenciales grandes.
        """
        # Clip para evitar overflow numérico
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivada de la función sigmoide.
        
        La derivada de σ(z) es σ(z) * (1 - σ(z)), útil para backpropagation.
        
        Args:
            z (np.ndarray): Valores de entrada pre-activación
            
        Returns:
            np.ndarray: Derivada de la sigmoide evaluada en z
            
        Note:
            Esta función es crucial para calcular gradientes en backpropagation.
        """
        s = self.sigmoid(z)
        return s * (1 - s)

    def compute_gradients(self) -> Tuple[np.floating, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calcula los gradientes de la función de pérdida respecto a todos los parámetros.
        
        Este método encapsula el cálculo de gradientes para ser reutilizado por
        diferentes métodos de integración numérica. Realiza:
        
        1. Propagación hacia adelante
        2. Cálculo de la función de pérdida
        3. Backpropagation para obtener gradientes
        
        Returns:
            Tuple[np.floating, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
                - loss: Valor actual de la función de pérdida (MSE)
                - grads: Tupla con gradientes (dL_dW1, dL_db1, dL_dW2, dL_db2)
                
        Note:
            Este método no modifica los pesos, solo calcula las derivadas
            necesarias para los métodos de integración.
        """
        # Propagación hacia adelante
        z1, a1, z2, a2 = self.forward()
        
        # Cálculo de la función de pérdida (MSE)
        loss = np.mean((self.y - a2) ** 2)

        # BACKPROPAGATION: Cálculo de gradientes
        # =====================================
        
        # Gradientes para la capa de salida
        dL_da2 = 2 * (a2 - self.y) / self.y.shape[0]
        da2_dz2 = self.sigmoid_derivative(z2)
        dL_dz2 = dL_da2 * da2_dz2

        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        # Gradientes para la capa oculta
        dL_da1 = dL_dz2 @ self.W2.T
        da1_dz1 = self.sigmoid_derivative(z1)
        dL_dz1 = dL_da1 * da1_dz1

        dL_dW1 = self.X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Empaquetar gradientes
        grads = (dL_dW1, dL_db1, dL_dW2, dL_db2)
        return loss, grads

    def forward(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagación hacia adelante (forward pass) de la red neuronal.
        
        Calcula las activaciones de todas las capas de la red, desde la entrada
        hasta la salida. Este proceso implementa:
        
        Capa oculta: z1 = X·W1 + b1, a1 = σ(z1)
        Capa salida:  z2 = a1·W2 + b2, a2 = σ(z2)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - z1: Pre-activaciones de la capa oculta (4x3)
                - a1: Activaciones de la capa oculta (4x3)
                - z2: Pre-activaciones de la capa de salida (4x1)
                - a2: Activaciones de la capa de salida (predicciones) (4x1)
                
        Note:
            Todas las muestras se procesan en paralelo (vectorización).
        """
        # Capa oculta: combinación lineal + activación
        z1 = self.X @ self.W1 + self.b1  # Pre-activación (4x3)
        a1 = self.sigmoid(z1)            # Activación (4x3)
        
        # Capa de salida: combinación lineal + activación
        z2 = a1 @ self.W2 + self.b2      # Pre-activación (4x1)
        a2 = self.sigmoid(z2)            # Activación final (4x1)
        
        return z1, a1, z2, a2

    def step(self) -> Tuple[np.floating, np.ndarray, np.ndarray]:
        """
        Ejecuta un solo paso del método de integración seleccionado.
        
        Este método actúa como dispatcher, seleccionando el método de integración
        numérica apropiado basado en self.method:
        
        - 'Euler': Método de Euler explícito (primer orden)
        - 'RK2': Runge-Kutta de segundo orden (método del punto medio)
        - 'RK4': Runge-Kutta de cuarto orden (método clásico)
        
        Todos resuelven el sistema de EDO: dW/dt = -η * ∇L(W)
        
        Returns:
            Tuple[np.floating, np.ndarray, np.ndarray]:
                - loss: Valor actual de la función de pérdida (MSE)
                - dL_dW1: Gradiente inicial respecto a pesos de capa oculta
                - dL_dW2: Gradiente inicial respecto a pesos de capa salida
                
        Raises:
            ValueError: Si el método especificado no está soportado.
        """
        if self.method == "Euler":
            return self.step_euler()
        elif self.method == "RK2":
            return self.step_rk2()
        elif self.method == "RK4":
            return self.step_rk4()
        else:
            raise ValueError(f"Método de integración '{self.method}' no soportado. "
                           f"Opciones válidas: 'Euler', 'RK2', 'RK4'")

    def step_euler(self) -> Tuple[np.floating, np.ndarray, np.ndarray]:
        """
        Implementa un paso del método de Euler explícito.
        
        El método de Euler es el más simple de los métodos de integración numérica.
        Aproxima la solución usando la fórmula:
        
        W(t+h) = W(t) + h * f(t, W(t))
        
        Donde f(t, W(t)) = -∇L(W(t)) en nuestro contexto de optimización.
        
        Características:
        - Orden de precisión: O(h)
        - Evaluaciones por paso: 1
        - Estabilidad: Condicional
        
        Returns:
            Tuple[np.floating, np.ndarray, np.ndarray]: Loss y gradientes iniciales
        """
        # Calcular gradientes en el punto actual
        loss, grads = self.compute_gradients()
        dW1, db1, dW2, db2 = grads

        # Aplicar actualización de Euler: W_nuevo = W_viejo - η * ∇L
        self.W1 -= self.eta * dW1
        self.b1 -= self.eta * db1
        self.W2 -= self.eta * dW2
        self.b2 -= self.eta * db2

        return loss, dW1, dW2

    def step_rk2(self) -> Tuple[np.floating, np.ndarray, np.ndarray]:
        """
        Implementa un paso del método de Runge-Kutta de segundo orden (RK2).
        
        También conocido como método del punto medio, RK2 mejora la precisión
        de Euler evaluando la derivada en un punto intermedio:
        
        k1 = h * f(t, W)
        k2 = h * f(t + h/2, W + k1/2)
        W(t+h) = W(t) + k2
        
        Características:
        - Orden de precisión: O(h²)
        - Evaluaciones por paso: 2
        - Mejor estabilidad que Euler
        
        Returns:
            Tuple[np.floating, np.ndarray, np.ndarray]: Loss y gradientes iniciales
        """
        # k1: Gradiente inicial en el punto actual
        loss, grads1 = self.compute_gradients()
        dW1_1, db1_1, dW2_1, db2_1 = grads1

        # Calcular punto intermedio (W + h/2 * k1)
        W1_temp = self.W1 - 0.5 * self.eta * dW1_1
        b1_temp = self.b1 - 0.5 * self.eta * db1_1
        W2_temp = self.W2 - 0.5 * self.eta * dW2_1
        b2_temp = self.b2 - 0.5 * self.eta * db2_1

        # Evaluar gradientes en el punto intermedio
        self.W1, self.b1, self.W2, self.b2 = W1_temp, b1_temp, W2_temp, b2_temp
        _, grads2 = self.compute_gradients()
        dW1_2, db1_2, dW2_2, db2_2 = grads2

        # Restaurar pesos originales y aplicar actualización RK2
        # W_nuevo = W_viejo + k2 (donde k2 usa el gradiente del punto medio)
        self.W1 += 0.5 * self.eta * dW1_1 - self.eta * dW1_2
        self.b1 += 0.5 * self.eta * db1_1 - self.eta * db1_2
        self.W2 += 0.5 * self.eta * dW2_1 - self.eta * dW2_2
        self.b2 += 0.5 * self.eta * db2_1 - self.eta * db2_2

        return loss, dW1_1, dW2_1

    def step_rk4(self) -> Tuple[np.floating, np.ndarray, np.ndarray]:
        """
        Implementa un paso del método de Runge-Kutta de cuarto orden (RK4).
        
        RK4 es el método clásico de alta precisión que evalúa la derivada en
        cuatro puntos y combina los resultados:
        
        k1 = h * f(t, W)
        k2 = h * f(t + h/2, W + k1/2)
        k3 = h * f(t + h/2, W + k2/2)
        k4 = h * f(t + h, W + k3)
        W(t+h) = W(t) + (k1 + 2*k2 + 2*k3 + k4)/6
        
        Características:
        - Orden de precisión: O(h⁴)
        - Evaluaciones por paso: 4
        - Excelente estabilidad y precisión
        
        Returns:
            Tuple[np.floating, np.ndarray, np.ndarray]: Loss y gradientes iniciales
        """
        # Guardar estado inicial
        W1_orig, b1_orig = self.W1.copy(), self.b1.copy()
        W2_orig, b2_orig = self.W2.copy(), self.b2.copy()
        
        # k1: Gradiente en el punto inicial
        loss, grads1 = self.compute_gradients()
        dW1_1, db1_1, dW2_1, db2_1 = grads1

        # k2: Gradiente en el punto W + h/2 * k1
        self.W1 = W1_orig - 0.5 * self.eta * dW1_1
        self.b1 = b1_orig - 0.5 * self.eta * db1_1
        self.W2 = W2_orig - 0.5 * self.eta * dW2_1
        self.b2 = b2_orig - 0.5 * self.eta * db2_1
        _, grads2 = self.compute_gradients()
        dW1_2, db1_2, dW2_2, db2_2 = grads2

        # k3: Gradiente en el punto W + h/2 * k2
        self.W1 = W1_orig - 0.5 * self.eta * dW1_2
        self.b1 = b1_orig - 0.5 * self.eta * db1_2
        self.W2 = W2_orig - 0.5 * self.eta * dW2_2
        self.b2 = b2_orig - 0.5 * self.eta * db2_2
        _, grads3 = self.compute_gradients()
        dW1_3, db1_3, dW2_3, db2_3 = grads3

        # k4: Gradiente en el punto W + h * k3
        self.W1 = W1_orig - self.eta * dW1_3
        self.b1 = b1_orig - self.eta * db1_3
        self.W2 = W2_orig - self.eta * dW2_3
        self.b2 = b2_orig - self.eta * db2_3
        _, grads4 = self.compute_gradients()
        dW1_4, db1_4, dW2_4, db2_4 = grads4

        # Aplicar combinación ponderada de RK4: (k1 + 2*k2 + 2*k3 + k4)/6
        self.W1 = W1_orig - (self.eta/6) * (dW1_1 + 2*dW1_2 + 2*dW1_3 + dW1_4)
        self.b1 = b1_orig - (self.eta/6) * (db1_1 + 2*db1_2 + 2*db1_3 + db1_4)
        self.W2 = W2_orig - (self.eta/6) * (dW2_1 + 2*dW2_2 + 2*dW2_3 + dW2_4)
        self.b2 = b2_orig - (self.eta/6) * (db2_1 + 2*db2_2 + 2*db2_3 + db2_4)

        return loss, dW1_1, dW2_1

    def train(self) -> np.ndarray:
        """
        Entrena la red neuronal usando el método de integración seleccionado.
        
        Ejecuta múltiples pasos del método de integración numérica elegido
        (Euler, RK2, o RK4) para resolver iterativamente el sistema de EDO
        asociado a la optimización de la red:
        
        dW/dt = -η * ∇L(W)
        
        Cada época corresponde a una aproximación numérica del sistema,
        donde la precisión y estabilidad dependen del método seleccionado.
        
        Returns:
            np.ndarray: Predicciones finales de la red (4x1)
            
        Note:
            El historial de pérdida se almacena en self.loss_history para
            posterior visualización y análisis comparativo de convergencia
            entre diferentes métodos de integración.
        """
        # Entrenamiento iterativo usando método de Euler
        for epoch in range(self.epochs):
            loss, _, _ = self.step()  # Un paso de Euler
            self.loss_history.append(float(loss))  # Guardar para visualización

        # Obtener predicciones finales después del entrenamiento
        _, _, _, a2 = self.forward()
        return a2


# --------------------------
# GUI con PyQt5
# --------------------------

class NNApp(QMainWindow):
    """
    Interfaz Gráfica para la Red Neuronal con Múltiples Métodos de Integración.
    
    Esta clase proporciona una interfaz interactiva que permite:
    - Configurar parámetros de entrenamiento (η, épocas)
    - Seleccionar método de integración (Euler, RK2, RK4)
    - Entrenar la red completa o paso a paso
    - Visualizar y comparar curvas de pérdida entre métodos
    - Observar gradientes y actualizaciones en tiempo real
    - Resetear la red para nuevos experimentos
    - Análisis comparativo de convergencia y estabilidad
    
    La interfaz está diseñada para facilitar la comprensión y comparación
    de diferentes métodos de integración numérica aplicados a la 
    optimización de redes neuronales.
    
    Attributes:
        nn (Optional[SimpleNN]): Instancia de la red neuronal
        lr_input (QLineEdit): Campo para tasa de aprendizaje
        epochs_input (QLineEdit): Campo para número de épocas
        method_combo (QComboBox): Selector de método de integración
        results (QTextEdit): Área de texto para mostrar resultados
        figure (Figure): Figura de matplotlib para gráficos
        canvas (FigureCanvas): Canvas para renderizar gráficos
    """
    
    def __init__(self) -> None:
        """
        Inicializa la interfaz gráfica con todos los componentes.
        
        Configura el layout, widgets de entrada, botones de control,
        área de resultados y canvas para gráficos.
        """
        super().__init__()
        # Configuración de la ventana principal
        self.setWindowTitle("Red Neuronal con EDO - XOR")
        self.setGeometry(200, 200, 900, 600)  # x, y, ancho, alto

        # Layout principal vertical
        layout = QVBoxLayout()

        # Sección de parámetros de entrada
        # Layout horizontal para organizar controles de parámetros
        params_layout = QHBoxLayout()
        
        # Control para tasa de aprendizaje (η - paso de Euler)
        self.lr_label = QLabel("Tasa de aprendizaje (η):")
        self.lr_input = QLineEdit("0.1")  # Valor por defecto
        
        # Control para número de épocas
        self.epochs_label = QLabel("Épocas:")
        self.epochs_input = QLineEdit("5000")  # Valor por defecto
        
        # Control para selección de método de integración
        self.method_label = QLabel("Método:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Euler", "RK2", "RK4"])
        self.method_combo.setCurrentText("Euler")  # Método por defecto

        # Agregar controles de parámetros al layout horizontal
        params_layout.addWidget(self.lr_label)
        params_layout.addWidget(self.lr_input)
        params_layout.addWidget(self.epochs_label)
        params_layout.addWidget(self.epochs_input)
        params_layout.addWidget(self.method_label)
        params_layout.addWidget(self.method_combo)

        # Agregar layout de parámetros al layout principal
        layout.addLayout(params_layout)

        # Botones de control de la aplicación
        # ===================================
        
        # Botón para entrenamiento completo
        self.train_btn = QPushButton("Entrenar Red")
        self.train_btn.clicked.connect(self.train_network)
        layout.addWidget(self.train_btn)

        # Botón para ejecutar un solo paso del método seleccionado (modo educativo)
        self.step_btn = QPushButton("Un Paso de Integración")
        self.step_btn.clicked.connect(self.one_step)
        layout.addWidget(self.step_btn)

        # Botón para reiniciar la red neuronal
        self.reset_btn = QPushButton("Resetear Red")
        self.reset_btn.clicked.connect(self.reset_network)
        layout.addWidget(self.reset_btn)

        # Área de texto para mostrar resultados y información detallada
        self.results = QTextEdit()
        self.results.setReadOnly(True)  # Solo lectura para evitar edición accidental
        layout.addWidget(self.results)

        # Canvas de matplotlib para visualización de gráficos
        self.figure = Figure()  # Figura de matplotlib
        self.canvas = FigureCanvas(self.figure)  # Canvas Qt5 para matplotlib
        layout.addWidget(self.canvas)

        # Configuración final del layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Instancia de red neuronal (inicialmente None)
        self.nn: Optional[SimpleNN] = None

    def train_network(self) -> None:
        """
        Entrena la red neuronal completa y muestra resultados.
        
        Este método:
        1. Lee parámetros de la interfaz (η, épocas, método)
        2. Crea una nueva instancia de red neuronal
        3. Ejecuta el entrenamiento completo usando el método seleccionado
        4. Muestra predicciones finales y precisión
        5. Grafica la curva de pérdida durante el entrenamiento
        
        Note:
            Los resultados se muestran en formato legible comparando
            predicciones con valores reales del problema XOR, incluyendo
            información sobre el método de integración utilizado.
        """
        # Obtener parámetros de la interfaz
        eta = float(self.lr_input.text())       # Tasa de aprendizaje
        epochs = int(self.epochs_input.text())  # Número de épocas
        method = self.method_combo.currentText() # Método de integración
        
        # Crear nueva instancia de red con parámetros especificados
        self.nn = SimpleNN(eta=eta, epochs=epochs, method=method)
        
        # Ejecutar entrenamiento completo
        outputs = self.nn.train()

        # Procesar y mostrar resultados finales
        # Convertir probabilidades a predicciones binarias (umbral 0.5)
        preds = (outputs > 0.5).astype(int)
        
        # Generar reporte detallado de resultados
        result_text = f"Resultados finales - Método {method}:\n"
        result_text += "=" * 60 + "\n"
        result_text += f"Método de integración: {method}\n"
        result_text += f"Tasa de aprendizaje: {eta}\n"
        result_text += f"Épocas: {epochs}\n\n"
        
        # Mostrar cada caso del problema XOR
        for i in range(len(self.nn.X)):
            entrada = self.nn.X[i]
            predicho = preds[i][0]
            real = self.nn.y[i][0]
            probabilidad = outputs[i][0]
            
            # Indicar si la predicción es correcta
            correcto = "✓" if predicho == real else "✗"
            
            result_text += f"Entrada: {entrada} -> Predicho: {predicho} "
            result_text += f"(Real: {real}) [{probabilidad:.3f}] {correcto}\n"
        
        # Calcular y mostrar precisión
        accuracy = np.mean(preds.flatten() == self.nn.y.flatten()) * 100
        result_text += f"\nPrecisión: {accuracy:.1f}%\n"
        result_text += f"Pérdida final: {self.nn.loss_history[-1]:.6f}\n"
        
        # Información adicional sobre el método
        method_info = {
            "Euler": "Orden O(h), 1 evaluación por paso",
            "RK2": "Orden O(h²), 2 evaluaciones por paso", 
            "RK4": "Orden O(h⁴), 4 evaluaciones por paso"
        }
        result_text += f"Características del método: {method_info.get(method, 'N/A')}\n"
        
        self.results.setText(result_text)

        # Visualización de la curva de pérdida
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Graficar evolución de la pérdida
        colors = {"Euler": "blue", "RK2": "green", "RK4": "red"}
        color = colors.get(method, "blue")
        
        ax.plot(self.nn.loss_history, color=color, linewidth=2, 
                label=f"Pérdida (MSE) - {method}")
        ax.set_title(f"Evolución de la Función de Pérdida\nMétodo: {method}")
        ax.set_xlabel(f"Épocas (Pasos de {method})")
        ax.set_ylabel("Error Cuadrático Medio")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Actualizar canvas
        self.canvas.draw()

    def one_step(self) -> None:
        """
        Ejecuta un solo paso del método de integración seleccionado y muestra detalles.
        
        Este método educativo permite observar paso a paso cómo el
        método de integración elegido actualiza los pesos de la red. Muestra:
        
        - Gradientes calculados (∇W1, ∇W2)
        - Ecuaciones diferenciales aplicadas
        - Valor actual de la función de pérdida
        - Interpretación específica del método utilizado
        - Comparación de características entre métodos
        
        Si no existe una red, la crea con los parámetros actuales.
        
        Note:
            Útil para entender las diferencias entre métodos de integración
            numérica aplicados a la optimización.
        """
        # Crear red si no existe
        if self.nn is None:
            eta = float(self.lr_input.text())
            method = self.method_combo.currentText()
            # Crear con epochs=1 para uso paso a paso
            self.nn = SimpleNN(eta=eta, epochs=1, method=method)

        # Ejecutar un solo paso del método seleccionado
        loss, gradW1, gradW2 = self.nn.step()

        # Mostrar información detallada del paso de integración
        text = self.results.toPlainText()
        text += f"\n{'='*70}\n"
        text += f"PASO DE {self.nn.method.upper()} #{len(self.nn.loss_history)}\n"
        text += f"{'='*70}\n"
        
        # Explicación teórica específica del método
        text += f"Ecuación diferencial: dW/dt = -η * ∇L(W)\n"
        
        if self.nn.method == "Euler":
            text += f"Aproximación de Euler: W_nuevo = W_viejo - η * ∇L\n"
            text += f"Evaluaciones por paso: 1\n\n"
        elif self.nn.method == "RK2":
            text += f"Aproximación RK2: Usa punto medio para mayor precisión\n"
            text += f"Evaluaciones por paso: 2\n\n"
        elif self.nn.method == "RK4":
            text += f"Aproximación RK4: Combina 4 evaluaciones con pesos\n"
            text += f"Evaluaciones por paso: 4\n\n"
        
        # Valores numéricos de gradientes iniciales
        text += f"Gradientes iniciales calculados:\n"
        text += f"∇W1 (capa oculta):\n{gradW1}\n\n"
        text += f"∇W2 (capa salida):\n{gradW2}\n\n"
        
        # Estado actual
        text += f"Pérdida actual: {loss:.6f}\n"
        text += f"Tasa de aprendizaje (η): {self.nn.eta}\n"
        text += f"Método: {self.nn.method}\n"
        
        # Agregar a historial y mostrar
        self.nn.loss_history.append(float(loss))
        self.results.setText(text)

    def reset_network(self) -> None:
        """
        Reinicia completamente la aplicación.
        
        Este método:
        - Elimina la instancia actual de la red neuronal
        - Limpia el área de resultados
        - Borra todos los gráficos
        - Prepara la interfaz para un nuevo experimento
        
        Útil para comenzar experimentos frescos con diferentes parámetros.
        """
        # Eliminar instancia de red neuronal
        self.nn = None
        
        # Limpiar área de resultados
        self.results.clear()
        
        # Limpiar gráficos
        self.figure.clear()
        self.canvas.draw()
        
        # Mensaje informativo
        self.results.setText("Red neuronal reiniciada. Lista para nuevo entrenamiento.\n" +
                           "Seleccione método de integración y parámetros deseados.")


# --------------------------
# Punto de entrada principal
# --------------------------

if __name__ == "__main__":
    """
    Punto de entrada principal de la aplicación.
    
    Crea la aplicación Qt, instancia la ventana principal con soporte
    para múltiples métodos de integración numérica (Euler, RK2, RK4)
    e inicia el bucle de eventos de la interfaz gráfica.
    """
    # Crear aplicación Qt
    app = QApplication(sys.argv)
    
    # Crear y mostrar ventana principal
    window = NNApp()
    window.show()
    
    # Iniciar bucle de eventos (bloquea hasta cerrar ventana)
    sys.exit(app.exec_())
