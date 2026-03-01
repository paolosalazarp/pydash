🟩 Geometry Dash con Reinforcement Learning (Q-Learning)

Proyecto desarrollado para el curso de Inteligencia Computacional, donde se implementa un agente basado en Reinforcement Learning (Q-Learning tabular) para aprender a completar un nivel inspirado en Geometry Dash.

📌 Descripción

En este proyecto se modela una versión simplificada de Geometry Dash como un Markov Decision Process (MDP) y se entrena un agente mediante Q-Learning para que aprenda cuándo saltar y cuándo no, con el objetivo de evitar obstáculos y completar el nivel.

El entorno está representado mediante archivos .csv, y el agente aprende a través de múltiples episodios hasta converger a una política óptima.

🧠 Modelado del Problema
🎯 Estados

Los estados están definidos a partir de:

Posición actual del jugador

Información del siguiente obstáculo

Configuración del nivel (extraída de archivos .csv)

Cada estado es representado de manera discreta, permitiendo el uso de una Q-Table.

🎮 Acciones

El agente puede realizar únicamente:

0 → No saltar

1 → Saltar

🏆 Función de Recompensa

✅ Recompensa positiva por avanzar

❌ Penalización por colisión

🏁 Recompensa alta por completar el nivel

El diseño de la recompensa fue clave para lograr una convergencia estable.

🤖 Algoritmo Implementado

Se utilizó Q-Learning tabular, con:

Política ε-greedy (exploración vs explotación)

Actualización mediante la ecuación de Bellman:

𝑄
(
𝑠
,
𝑎
)
←
𝑄
(
𝑠
,
𝑎
)
+
𝛼
[
𝑟
+
𝛾
max⁡
𝑎
′
𝑄
(
𝑠
′
,
𝑎
′
)
−
𝑄
(
𝑠
,
𝑎
)
]
Q(s,a)←Q(s,a)+α[r+γ
a
′
max
	​
Q(s
′
,a
′
)−Q(s,a)]

Parámetros principales:

α → Learning rate

γ → Discount factor

ε → Exploration rate

Decaimiento progresivo de ε

📊 Resultados

Durante el entrenamiento se observó:

Incremento progresivo de la recompensa promedio

Reducción en muertes tempranas

Convergencia hacia una política estable

Capacidad de completar el nivel consistentemente


📚 Conceptos Aplicados

Markov Decision Process (MDP)

Q-Learning

Política ε-greedy

Exploración vs Explotación

Convergencia de Q-Table

Reward Shaping


📝 Conclusiones

Este proyecto demuestra cómo un agente basado en aprendizaje por refuerzo puede aprender comportamientos óptimos en un entorno dinámico y secuencial.

Se evidenció que:

El diseño de la función de recompensa es crítico.

El balance entre exploración y explotación impacta directamente en la convergencia.

Q-Learning es efectivo en entornos discretos bien definidos.
