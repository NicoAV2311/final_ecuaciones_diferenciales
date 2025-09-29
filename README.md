# Red Neuronal XOR con Métodos de Integración Numérica

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)](https://pypi.org/project/PyQt5/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-red.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción

Este proyecto implementa una **red neuronal feedforward** para resolver el problema XOR utilizando diferentes **métodos de integración numérica** aplicados a la optimización de pesos. El proyecto combina conceptos de **ecuaciones diferenciales ordinarias (EDO)** con **aprendizaje automático**, ofreciendo una perspectiva única sobre cómo los métodos numéricos influyen en la convergencia y estabilidad del entrenamiento.

### 🎯 Objetivo Académico

Demostrar y comparar cómo diferentes métodos de integración numérica afectan el proceso de optimización en redes neuronales, tratando el entrenamiento como la solución de un sistema de EDO:

```
dW/dt = -η * ∇L(W)
```

Donde:
- `W`: Pesos de la red neuronal
- `η`: Tasa de aprendizaje (paso de integración)
- `∇L`: Gradiente de la función de pérdida

## 🔬 Métodos de Integración Implementados

| Método | Orden | Evaluaciones/Paso | Precisión | Descripción |
|--------|-------|-------------------|-----------|-------------|
| **Euler Explícito** | O(h) | 1 | Básica | Método más simple, rápido pero menos preciso |
| **Runge-Kutta 2 (RK2)** | O(h²) | 2 | Media | Balance entre precisión y costo computacional |
| **Runge-Kutta 4 (RK4)** | O(h⁴) | 4 | Alta | Método clásico de alta precisión |

### 📊 Fórmulas Matemáticas

#### Euler Explícito
```
W(t+h) = W(t) + h * f(t, W(t))
```

#### Runge-Kutta 2 (Método del Punto Medio)
```
k₁ = h * f(t, W)
k₂ = h * f(t + h/2, W + k₁/2)
W(t+h) = W(t) + k₂
```

#### Runge-Kutta 4
```
k₁ = h * f(t, W)
k₂ = h * f(t + h/2, W + k₁/2)
k₃ = h * f(t + h/2, W + k₂/2)
k₄ = h * f(t + h, W + k₃)
W(t+h) = W(t) + (k₁ + 2k₂ + 2k₃ + k₄)/6
```

## 🏗️ Arquitectura de la Red Neuronal

```
Entrada (2) → Capa Oculta (3) → Salida (1)
```

- **Entrada**: 2 neuronas (bits para XOR)
- **Capa Oculta**: 3 neuronas con activación sigmoide
- **Salida**: 1 neurona con activación sigmoide
- **Función de Pérdida**: Error Cuadrático Medio (MSE)

### 📈 Problema XOR

| Entrada A | Entrada B | Salida Esperada |
|-----------|-----------|-----------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

## 🚀 Características

### ✨ Funcionalidades Principales

- **Selección de Método**: Interfaz para elegir entre Euler, RK2, y RK4
- **Entrenamiento Completo**: Optimización automática con el método seleccionado
- **Modo Paso a Paso**: Análisis educativo de cada iteración
- **Visualización en Tiempo Real**: Curvas de pérdida diferenciadas por color
- **Análisis Comparativo**: Métricas de convergencia y estabilidad
- **Interfaz Intuitiva**: GUI desarrollada en PyQt5

### 🎨 Interfaz Gráfica

- **Panel de Control**: Configuración de hiperparámetros
- **Selector de Método**: ComboBox para métodos de integración
- **Área de Resultados**: Información detallada del entrenamiento
- **Visualización**: Gráficos de convergencia con matplotlib
- **Controles**: Botones para entrenar, paso a paso, y reset

## 📦 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Dependencias

```bash
pip install numpy matplotlib PyQt5
```

O usando el archivo de requerimientos:

```bash
pip install -r requirements.txt
```

### Instalación Manual

```bash
# Clonar el repositorio
git clone https://github.com/NicoAV2311/final_ecuaciones_diferenciales.git

# Navegar al directorio
cd final_ecuaciones_diferenciales

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python "Tentativo final ecuaciones.py"
```

## 🎮 Uso

### Ejecución Básica

```bash
python "Tentativo final ecuaciones.py"
```

### Interfaz de Usuario

1. **Configurar Parámetros**:
   - Tasa de aprendizaje (η): 0.01 - 1.0
   - Épocas: 1000 - 10000
   - Método: Euler, RK2, o RK4

2. **Entrenar Red**:
   - Clic en "Entrenar Red" para optimización completa
   - Observar curva de convergencia en tiempo real

3. **Modo Educativo**:
   - Usar "Un Paso de Integración" para análisis detallado
   - Visualizar gradientes y actualizaciones paso a paso

4. **Análisis**:
   - Comparar diferentes métodos
   - Evaluar precisión y velocidad de convergencia

### Ejemplo de Uso Programático

```python
from Tentativo_final_ecuaciones import SimpleNN

# Crear red con método RK4
nn = SimpleNN(eta=0.1, epochs=2000, method="RK4")

# Entrenar
predictions = nn.train()

# Evaluar resultados
print(f"Pérdida final: {nn.loss_history[-1]:.6f}")
print(f"Predicciones: {predictions}")
```

## 📊 Resultados Esperados

### Convergencia Típica

- **Euler**: Convergencia lenta, posible inestabilidad con η alto
- **RK2**: Convergencia más suave, mejor estabilidad
- **RK4**: Convergencia rápida y estable, mayor costo computacional

### Métricas de Rendimiento

```
Método | Precisión Final | Épocas para 99% | Estabilidad
-------|----------------|-----------------|------------
Euler  | 95-98%         | 3000-5000      | Media
RK2    | 98-99%         | 2000-3000      | Alta
RK4    | 99-100%        | 1000-2000      | Muy Alta
```

## 🧪 Ejemplos de Experimentos

### Comparación de Métodos

```python
# Configuración experimental
methods = ["Euler", "RK2", "RK4"]
learning_rates = [0.05, 0.1, 0.2]
epochs = 2000

# Ejecutar comparaciones y analizar resultados
for method in methods:
    for lr in learning_rates:
        nn = SimpleNN(eta=lr, epochs=epochs, method=method)
        predictions = nn.train()
        # Analizar convergencia...
```

## 📁 Estructura del Proyecto

```
final_ecuaciones_diferenciales/
│
├── Tentativo final ecuaciones.py    # Archivo principal
├── README.md                        # Este archivo
├── requirements.txt                 # Dependencias
├── LICENSE                         # Licencia del proyecto
│
├── docs/                           # Documentación adicional
│   ├── mathematical_background.md  # Fundamentos matemáticos
│   └── user_guide.md              # Guía detallada de usuario
│
└── examples/                       # Ejemplos y experimentos
    ├── comparison_analysis.py      # Análisis comparativo
    └── parameter_tuning.py         # Optimización de parámetros
```

## 🔧 Configuración Avanzada

### Parámetros de Red

```python
# Arquitectura personalizable
self.W1 = np.random.randn(2, 3)  # Pesos entrada-oculta
self.b1 = np.zeros((1, 3))       # Sesgos capa oculta
self.W2 = np.random.randn(3, 1)  # Pesos oculta-salida
self.b2 = np.zeros((1, 1))       # Sesgos salida
```

### Hiperparámetros Recomendados

| Método | η (Tasa de Aprendizaje) | Épocas | Observaciones |
|--------|-------------------------|--------|---------------|
| Euler  | 0.05 - 0.1             | 3000+ | Reducir η si hay inestabilidad |
| RK2    | 0.1 - 0.2              | 2000+ | Balance óptimo |
| RK4    | 0.1 - 0.3              | 1500+ | Permite η más altos |

## 📖 Fundamentos Teóricos

### Ecuaciones Diferenciales en ML

El entrenamiento de redes neuronales puede modelarse como:

```
dW/dt = -∇L(W)
```

Este enfoque permite aplicar métodos de integración numérica desarrollados para EDO al contexto de optimización en aprendizaje automático.

### Ventajas del Enfoque EDO

1. **Perspectiva Continua**: Visualizar el entrenamiento como flujo continuo
2. **Análisis de Estabilidad**: Aplicar teoría de EDO para estudiar convergencia
3. **Métodos Adaptativos**: Potencial para control automático del paso
4. **Interpretabilidad**: Conexión clara entre matemáticas y ML

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear un Pull Request

### Áreas de Mejora

- [ ] Implementar métodos adaptativos (RK45, Dormand-Prince)
- [ ] Añadir soporte para problemas multi-clase
- [ ] Integrar métricas avanzadas de análisis numérico
- [ ] Desarrollar interfaz web con visualizaciones interactivas
- [ ] Implementar comparación automática de métodos

## 📚 Referencias

### Papers y Libros

1. Butcher, J.C. (2016). *Numerical Methods for Ordinary Differential Equations*
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*
3. Press, W.H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing*

### Recursos Adicionales

- [Documentación NumPy](https://numpy.org/doc/)
- [PyQt5 Documentation](https://doc.qt.io/qtforpython/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**Nicolás Andrés Villamizar**
- GitHub: [@NicoAV2311](https://github.com/NicoAV2311)
- Universidad: [Tu Universidad]
- Materia: Ecuaciones Diferenciales

## 📞 Contacto

Si tienes preguntas, sugerencias o encuentras algún problema:

- 🐛 **Issues**: [GitHub Issues](https://github.com/NicoAV2311/final_ecuaciones_diferenciales/issues)
- 📧 **Email**: [tu-email@universidad.edu]
- 💬 **Discusiones**: [GitHub Discussions](https://github.com/NicoAV2311/final_ecuaciones_diferenciales/discussions)

---

<div align="center">

**⭐ Si este proyecto te fue útil, por favor dale una estrella en GitHub ⭐**

*Proyecto desarrollado como trabajo final para la materia de Ecuaciones Diferenciales*

</div>
