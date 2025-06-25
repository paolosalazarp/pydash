import streamlit as st
import pygame
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pygame.math import Vector2
import time
import threading
import queue
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64

st.set_page_config(
    page_title="PyDash RL Training - Completo",
    page_icon="🎮",
    layout="wide"
)

# Header
st.markdown("""
<div style='background: linear-gradient(90deg, #FF6B6B, #4ECDC4); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>🎮 PyDash RL Training - VERSIÓN COMPLETA</h1>
    <p style='color: #ffffff; margin: 0;'>Con imágenes reales del juego y visualización de los últimos episodios</p>
</div>
""", unsafe_allow_html=True)

# Inicialización pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Constantes
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 50)
GRAVITY = Vector2(0, 0.86)
ACTIONS = [0, 1]

def load_game_images():
    """Cargar las imágenes reales del juego"""
    images = {}
    
    try:
        # Crear imágenes por defecto en caso de que no existan los archivos
        def create_fallback_image(color, size, shape="square"):
            surf = pygame.Surface(size, pygame.SRCALPHA)
            if shape == "square":
                pygame.draw.rect(surf, color, (0, 0, size[0], size[1]))
                pygame.draw.rect(surf, BLACK, (0, 0, size[0], size[1]), 2)
            elif shape == "spike":
                points = [(size[0]//2, 0), (0, size[1]), (size[0], size[1])]
                pygame.draw.polygon(surf, color, points)
                pygame.draw.polygon(surf, BLACK, points, 2)
            elif shape == "orb":
                center = (size[0]//2, size[1]//2)
                pygame.draw.circle(surf, color, center, size[0]//2)
                pygame.draw.circle(surf, BLACK, center, size[0]//2, 2)
            elif shape == "player":
                pygame.draw.rect(surf, color, (0, 0, size[0], size[1]))
                pygame.draw.rect(surf, BLACK, (0, 0, size[0], size[1]), 2)
                # Cara del cubo
                pygame.draw.circle(surf, BLACK, (size[0]//4, size[1]//4), 3)
                pygame.draw.circle(surf, BLACK, (3*size[0]//4, size[1]//4), 3)
                pygame.draw.arc(surf, BLACK, (size[0]//4, size[1]//2, size[0]//2, size[1]//4), 0, 3.14, 2)
            return surf
        
        # Intentar cargar imágenes reales
        image_paths = {
            'avatar': 'images/avatar.png',
            'block': 'images/block_1.png', 
            'spike': 'images/obj-spike.png',
            'orb': 'images/orb-yellow.png',
            'end': 'images/fin.png',
            'bg': 'images/bg.png'
        }
        
        for name, path in image_paths.items():
            try:
                if os.path.exists(path):
                    img = pygame.image.load(path)
                    if name != 'bg':  # Redimensionar sprites a 32x32
                        img = pygame.transform.smoothscale(img, (32, 32))
                    images[name] = img
                    st.sidebar.success(f"✅ Cargada: {name}")
                else:
                    # Usar imagen de respaldo
                    if name == 'avatar':
                        images[name] = create_fallback_image(BLUE, (32, 32), "player")
                    elif name == 'block':
                        images[name] = create_fallback_image(BLACK, (32, 32), "square")
                    elif name == 'spike':
                        images[name] = create_fallback_image(RED, (32, 32), "spike")
                    elif name == 'orb':
                        images[name] = create_fallback_image(YELLOW, (32, 32), "orb")
                    elif name == 'end':
                        images[name] = create_fallback_image(GREEN, (32, 32), "square")
                    elif name == 'bg':
                        bg_surf = pygame.Surface((800, 600))
                        bg_surf.fill((135, 206, 235))  # Sky blue
                        images[name] = bg_surf
                    st.sidebar.warning(f"⚠️ Usando respaldo para: {name}")
            except Exception as e:
                st.sidebar.error(f"❌ Error cargando {name}: {e}")
                # Usar imagen de respaldo
                images[name] = create_fallback_image(BLUE if name == 'avatar' else BLACK, (32, 32))
        
        return images
        
    except Exception as e:
        st.error(f"Error general cargando imágenes: {e}")
        return {}

class CompletPlayer(pygame.sprite.Sprite):
    """Jugador completo con las imágenes reales"""
    def __init__(self, image, platforms, pos, *groups):
        super().__init__(*groups)
        self.platforms = platforms
        self.image = image
        self.rect = self.image.get_rect(center=pos)
        self.onGround = False
        self.vel = Vector2(6, 0)  # ✅ Velocidad horizontal fija
        self.isjump = False
        self.jump_amount = 12
        self.win = False
        self.died = False
        self.distance_traveled = 0
        self.initial_x = pos[0]
        self.survival_time = 0

    def jump(self):
        self.vel.y = -self.jump_amount

    def collide(self, yvel):
        for p in self.platforms:
            if pygame.sprite.collide_rect(self, p):
                if isinstance(p, Spike):
                    self.died = True
                if isinstance(p, End):
                    self.win = True
                if isinstance(p, Orb) and self.isjump:
                    self.jump_amount = 12
                    self.jump()
                    self.jump_amount = 10

                if isinstance(p, Platform):
                    if yvel > 0:
                        self.rect.bottom = p.rect.top
                        self.vel.y = 0
                        self.onGround = True
                        self.isjump = False
                    elif yvel < 0:
                        self.rect.top = p.rect.bottom
                    else:
                        self.vel.x = 0
                        self.rect.right = p.rect.left
                        self.died = True

    def update(self):
        # Salto
        if self.isjump and self.onGround:
            self.jump()
        
        # Gravedad
        if not self.onGround:
            self.vel += GRAVITY
            if self.vel.y > 100:
                self.vel.y = 100

        # ✅ MOVIMIENTO HORIZONTAL CRÍTICO
        self.rect.x += self.vel.x
        self.distance_traveled = self.rect.x - self.initial_x
        self.survival_time += 1
        
        # Colisiones
        self.collide(0)
        self.rect.top += self.vel.y
        self.onGround = False
        self.collide(self.vel.y)

class CompletPlayer2(pygame.sprite.Sprite):
    """Player2 para referencia del estado"""
    def __init__(self, image, platforms, pos, *groups):
        super().__init__(*groups)
        self.platforms = platforms
        self.image = image
        self.rect = self.image.get_rect(center=pos)
        self.vel = Vector2(6, 0)

    def update(self):
        self.rect.x += self.vel.x

# Clases de objetos con imágenes reales
class Draw(pygame.sprite.Sprite):
    def __init__(self, image, pos, *groups):
        super().__init__(*groups)
        self.image = image
        self.rect = image.get_rect(topleft=pos)

class Platform(Draw): pass
class Spike(Draw): pass
class Orb(Draw): pass
class End(Draw): pass

def create_level_from_csv():
    """Crear nivel basado en tu level_1.csv"""
    try:
        # Intentar leer el CSV real
        if os.path.exists("level_1.csv"):
            with open("level_1.csv", newline='') as f:
                level_data = [row for row in csv.reader(f)]
            
            # Convertir el formato de tu CSV
            level = []
            for row in level_data:
                new_row = []
                for cell in row:
                    if cell == "0":
                        new_row.append("0")
                    elif cell == "Spike":
                        new_row.append("Spike")
                    elif cell == "Orb":
                        new_row.append("Orb")
                    elif cell == "End":
                        new_row.append("End")
                    else:
                        new_row.append("")
                level.append(new_row)
            return level
        else:
            # Crear nivel basado en tu CSV original
            level = []
            for y in range(19):
                row = []
                for x in range(85):
                    if y == 16:  # Fila 17 del CSV original (suelo)
                        row.append("0")
                    elif y == 15:  # Fila 16 del CSV original (obstáculos)
                        if x in [13, 27, 28, 52]:  # Posiciones de spikes del CSV
                            row.append("Spike")
                        elif x % 25 == 0 and x > 10:  # Algunos orbes
                            row.append("Orb")
                        else:
                            row.append("")
                    elif x == 83 and y == 15:  # Final
                        row.append("End")
                    else:
                        row.append("")
                level.append(row)
            return level
    
    except Exception as e:
        st.error(f"Error creando nivel: {e}")
        # Nivel básico de emergencia
        return [["" for _ in range(50)] for _ in range(19)]

def init_level_with_real_images(mapdata, elements, images):
    """Inicializar nivel con las imágenes reales cargadas"""
    for y, row in enumerate(mapdata):
        for x, col in enumerate(row):
            pos = (x*32, y*32)
            if col == "0":
                Platform(images['block'], pos, elements)
            elif col == "Spike":
                Spike(images['spike'], pos, elements)
            elif col == "Orb":
                Orb(images['orb'], pos, elements)
            elif col == "End":
                End(images['end'], pos, elements)

def get_state_enhanced(player, player2, next_obstacle_type):
    """Estado mejorado"""
    return (
        min(max(player.rect.x // 32, 0), 100),
        min(max(player.rect.y // 32, 0), 20),
        next_obstacle_type,
        int(player.isjump)
    )

def find_next_obstacle_enhanced(player, elements):
    """Encontrar siguiente obstáculo"""
    objs = [e for e in elements if isinstance(e, (Spike, Orb))]
    if not objs:
        return 0
    
    future_objs = [o for o in objs if o.rect.x > player.rect.x]
    if not future_objs:
        return 0
    
    nxt = min(future_objs, key=lambda o: o.rect.x - player.rect.x)
    distance = nxt.rect.x - player.rect.x
    
    if isinstance(nxt, Spike):
        return 1
    elif isinstance(nxt, Orb):
        return 2
    return 0

def step_env_enhanced(state, elements, player, player2, action):
    """Función de paso mejorada"""
    old_distance = player.distance_traveled
    
    # Aplicar acción
    player.isjump = (action == 1)
    
    # Actualizar jugadores
    player.update()
    player2.update()
    
    # Sistema de recompensas progresivo
    distance_progress = player.distance_traveled - old_distance
    base_reward = 0.1
    distance_reward = distance_progress * 0.05
    reward = base_reward + distance_reward
    
    # Verificar condiciones
    done = False
    
    if player.died:
        reward = -15  # Penalización por muerte
        done = True
    elif player.win:
        reward = 100 + player.distance_traveled * 0.02  # Gran recompensa por ganar
        done = True
    elif player.rect.x >= 2700:  # Meta realista
        reward = 50 + player.distance_traveled * 0.01
        done = True
    
    # Recompensas por comportamiento inteligente
    next_obs_type = find_next_obstacle_enhanced(player, elements)
    
    # Calcular distancia al siguiente obstáculo
    objs = [e for e in elements if isinstance(e, (Spike, Orb)) and e.rect.x > player.rect.x]
    if objs:
        nearest = min(objs, key=lambda o: o.rect.x - player.rect.x)
        distance_to_obstacle = nearest.rect.x - player.rect.x
        
        # Recompensar salto anticipado ante spikes
        if isinstance(nearest, Spike) and distance_to_obstacle < 120 and action == 1:
            reward += 3.0
        # Penalizar saltos innecesarios
        elif distance_to_obstacle > 200 and action == 1:
            reward -= 1.0
    
    # Nuevo estado
    next_state = get_state_enhanced(player, player2, next_obs_type)
    
    return reward, next_state, done

def train_complete_agent(episodes, alpha, gamma, epsilon, epsilon_decay, progress_queue, images):
    """Entrenamiento completo con imágenes reales"""
    q_table = defaultdict(lambda: [0.0, 0.0])
    reward_log = []
    max_distance_log = []
    
    for ep in range(episodes):
        try:
            # Inicializar episodio
            elements = pygame.sprite.Group()
            player = CompletPlayer(images['avatar'], elements, (150, 400))
            player2 = CompletPlayer2(images['avatar'], elements, (150, 400))
            
            # Usar nivel real
            leveldata = create_level_from_csv()
            init_level_with_real_images(leveldata, elements, images)
            
            obs_type = find_next_obstacle_enhanced(player, elements)
            state = get_state_enhanced(player, player2, obs_type)
            total_reward = 0
            
            # Mostrar últimos 10 episodios (más episodios visualizados)
            show_visualization = ep >= episodes - 10
            step_count = 0
            
            for step in range(800):  # Más pasos por episodio
                step_count = step
                
                # Epsilon-greedy mejorado
                if np.random.random() < epsilon:
                    # Exploración inteligente
                    if obs_type == 1:  # Spike detectado
                        action = np.random.choice([0, 1], p=[0.3, 0.7])  # Más probable saltar
                    else:
                        action = np.random.choice(ACTIONS)
                else:
                    action = np.argmax(q_table[state])
                
                # Ejecutar paso
                reward, next_state, done = step_env_enhanced(state, elements, player, player2, action)
                total_reward += reward
                
                # Actualizar Q-table
                old_value = q_table[state][action]
                next_max = max(q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state][action] = new_value
                
                # Enviar frame para visualización (más frecuente)
                if show_visualization and step % 6 == 0:
                    try:
                        frame = create_complete_game_frame(player, elements, ep + 1, step, 
                                                         total_reward, action, state, images)
                        if frame:
                            progress_queue.put({
                                'type': 'frame',
                                'episode': ep + 1,
                                'step': step,
                                'reward': total_reward,
                                'action': action,
                                'player_x': player.rect.x,
                                'distance': player.distance_traveled,
                                'survival_time': player.survival_time,
                                'frame': frame,
                                'q_table_size': len(q_table)
                            })
                    except Exception as e:
                        print(f"Error en visualización: {e}")
                
                state = next_state
                
                if done:
                    break
            
            reward_log.append(total_reward)
            max_distance_log.append(player.distance_traveled)
            epsilon = max(0.005, epsilon * epsilon_decay)
            
            # Progreso cada 15 episodios (más frecuente)
            if ep % 15 == 0:
                progress_queue.put({
                    'type': 'progress',
                    'episode': ep + 1,
                    'reward': total_reward,
                    'epsilon': epsilon,
                    'q_table_size': len(q_table),
                    'max_distance': player.distance_traveled,
                    'survival_time': player.survival_time,
                    'avg_reward_last_10': np.mean(reward_log[-10:]) if len(reward_log) >= 10 else total_reward
                })
            
        except Exception as e:
            print(f"Error en episodio {ep + 1}: {e}")
            continue
    
    # Completado
    progress_queue.put({
        'type': 'complete',
        'q_table': dict(q_table),
        'rewards': reward_log,
        'max_distances': max_distance_log,
        'final_epsilon': epsilon
    })

def create_complete_game_frame(player, elements, episode, step, reward, action, state, images):
    """Crear frame completo con imágenes reales y mejor HUD"""
    try:
        game_surface = pygame.Surface((800, 600))
        
        # Usar background real si está disponible
        if 'bg' in images:
            game_surface.blit(images['bg'], (0, 0))
        else:
            game_surface.fill((135, 206, 235))  # Sky blue fallback
        
        # Cámara que sigue al jugador suavemente
        camera_x = max(0, player.rect.x - 300)
        
        # Dibujar elementos del nivel con las imágenes reales
        for element in elements:
            screen_x = element.rect.x - camera_x
            if -50 < screen_x < 850:
                screen_rect = pygame.Rect(screen_x, element.rect.y, 32, 32)
                game_surface.blit(element.image, screen_rect)
        
        # Dibujar jugador con imagen real
        player_screen_x = player.rect.x - camera_x
        player_rect = pygame.Rect(player_screen_x, player.rect.y, 32, 32)
        
        # Rotar jugador si está saltando (como en el juego original)
        if player.isjump:
            rotated_player = pygame.transform.rotate(player.image, player.survival_time * 5)
            rotated_rect = rotated_player.get_rect(center=player_rect.center)
            game_surface.blit(rotated_player, rotated_rect)
        else:
            game_surface.blit(player.image, player_rect)
        
        # HUD mejorado con más información
        font_large = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 20)
        
        # Título del episodio
        title_text = font_large.render(f"🎮 EPISODIO {episode} - CUBO APRENDIENDO", True, WHITE)
        title_rect = title_text.get_rect()
        pygame.draw.rect(game_surface, BLACK, (5, 5, title_rect.width + 10, title_rect.height + 5))
        game_surface.blit(title_text, (10, 8))
        
        # Información principal
        info_texts = [
            f"🏆 Reward: {reward:.1f}",
            f"🎯 Acción: {'🦘 SALTANDO' if action == 1 else '🏃 CORRIENDO'}",
            f"📍 Posición X: {player.rect.x}",
            f"📏 Distancia: {player.distance_traveled:.0f}",
            f"⏱️ Tiempo: {player.survival_time}",
            f"🧠 Estados Q: {len(q_table) if 'q_table' in globals() else 'N/A'}"
        ]
        
        for i, text in enumerate(info_texts):
            color = WHITE
            if i == 1:  # Acción
                color = GREEN if action == 1 else BLUE
            
            text_surface = font_small.render(text, True, color)
            # Fondo semi-transparente
            bg_rect = pygame.Rect(10, 45 + i * 22, text_surface.get_width() + 10, 20)
            pygame.draw.rect(game_surface, (0, 0, 0, 128), bg_rect)
            game_surface.blit(text_surface, (15, 47 + i * 22))
        
        # Barra de progreso más detallada
        progress_y = 200
        max_distance = 2700  # Distancia objetivo
        progress_ratio = min(player.distance_traveled / max_distance, 1.0)
        progress_width = int(300 * progress_ratio)
        
        # Fondo de la barra
        pygame.draw.rect(game_surface, BLACK, (10, progress_y, 304, 14))
        pygame.draw.rect(game_surface, (50, 50, 50), (12, progress_y + 2, 300, 10))
        
        # Barra de progreso
        if progress_width > 0:
            color = GREEN if progress_ratio > 0.8 else YELLOW if progress_ratio > 0.4 else RED
            pygame.draw.rect(game_surface, color, (12, progress_y + 2, progress_width, 10))
        
        # Texto del progreso
        progress_text = font_small.render(f"Progreso: {progress_ratio*100:.1f}%", True, WHITE)
        game_surface.blit(progress_text, (320, progress_y))
        
        # Indicador de siguiente obstáculo
        next_obs_type = find_next_obstacle_enhanced(player, elements)
        if next_obs_type > 0:
            obs_text = "🔺 SPIKE AHEAD!" if next_obs_type == 1 else "🔴 ORB AHEAD!"
            obs_color = RED if next_obs_type == 1 else YELLOW
            obs_surface = font_small.render(obs_text, True, obs_color)
            game_surface.blit(obs_surface, (15, 240))
        
        # Convertir a imagen PIL
        frame_string = pygame.image.tostring(game_surface, 'RGB')
        frame_image = Image.frombytes('RGB', (800, 600), frame_string)
        
        return frame_image
        
    except Exception as e:
        print(f"Error creando frame completo: {e}")
        return None

def main():
    st.header("🎯 PyDash RL Training - Versión Completa con Imágenes Reales")
    
    # Cargar imágenes al inicio
    with st.spinner("🎨 Cargando imágenes del juego..."):
        images = load_game_images()
    
    if not images:
        st.error("❌ No se pudieron cargar las imágenes del juego. Usando imágenes de respaldo.")
        return
    
    # Sidebar con configuración expandida
    st.sidebar.header("⚙️ Configuración Completa")
    
    st.sidebar.subheader("🎮 Entrenamiento")
    episodes = st.sidebar.slider("Episodios", 100, 2000, 500, step=50)
    alpha = st.sidebar.slider("Tasa de Aprendizaje (α)", 0.05, 0.9, 0.25, step=0.05)
    gamma = st.sidebar.slider("Factor de Descuento (γ)", 0.8, 0.999, 0.95, step=0.01)
    epsilon = st.sidebar.slider("Exploración Inicial (ε)", 0.1, 1.0, 0.7, step=0.1)
    epsilon_decay = st.sidebar.slider("Decaimiento de ε", 0.990, 0.9999, 0.998, step=0.0001)
    
    st.sidebar.subheader("👁️ Visualización")
    st.sidebar.info("✅ Últimos 10 episodios se mostrarán automáticamente")
    
    # Layout principal
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📈 Evolución del Aprendizaje")
        reward_chart = st.empty()
        
        st.subheader("🎮 Visualización en Tiempo Real (Últimos 10 Episodios)")
        game_display = st.empty()
        game_info = st.empty()
        
        if st.button("🚀 Iniciar Entrenamiento Completo", type="primary", key="start_training"):
            run_complete_training(episodes, alpha, gamma, epsilon, epsilon_decay,
                                reward_chart, game_display, game_info, images)
    
    with col2:
        st.subheader("📊 Dashboard de Métricas")
        metrics_display = st.empty()
        
        st.subheader("🧠 Estado del Aprendizaje")
        learning_display = st.empty()
        
        if 'complete_results' in st.session_state:
            results = st.session_state.complete_results
            if results.get('rewards'):
                rewards = results['rewards']
                
                # Métricas principales
                with metrics_display.container():
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🏆 Mejor Reward", f"{max(rewards):.1f}")
                        st.metric("📈 Último Reward", f"{rewards[-1]:.1f}")
                    with col_b:
                        st.metric("📊 Promedio Final", f"{np.mean(rewards[-20:]):.1f}")
                        st.metric("🎯 Episodios", len(rewards))
                
                # Análisis de aprendizaje
                with learning_display.container():
                    improvement = rewards[-1] - rewards[0] if len(rewards) > 1 else 0
                    consistency = 1 - (np.std(rewards[-50:]) / np.mean(rewards[-50:])) if len(rewards) >= 50 else 0
                    
                    st.metric("📈 Mejora Total", f"{improvement:.1f}")
                    st.metric("🎯 Consistencia", f"{consistency:.2f}")
                    
                    if improvement > 20:
                        st.success("🎉 ¡El agente está aprendiendo exitosamente!")
                    elif improvement > 5:
                        st.info("📈 El agente muestra progreso...")
                    else:
                        st.warning("⚠️ El agente necesita más entrenamiento")

def run_complete_training(episodes, alpha, gamma, epsilon, epsilon_decay,
                         reward_chart, game_display, game_info, images):
    """Ejecutar entrenamiento completo"""
    
    progress_queue = queue.Queue()
    
    # Iniciar hilo de entrenamiento
    training_thread = threading.Thread(
        target=train_complete_agent,
        args=(episodes, alpha, gamma, epsilon, epsilon_decay, progress_queue, images),
        daemon=True
    )
    training_thread.start()
    
    # Variables de seguimiento
    progress_bar = st.progress(0)
    status_text = st.empty()
    rewards = []
    episodes_list = []
    max_distances = []
    last_frame = None
    
    # Contenedor para estadísticas en tiempo real
    stats_container = st.empty()
    
    while training_thread.is_alive() or not progress_queue.empty():
        try:
            data = progress_queue.get(timeout=0.8)
            
            if data['type'] == 'complete':
                # Entrenamiento terminado
                progress_bar.progress(1.0)
                status_text.success("🎉 ¡Entrenamiento completado! El agente ha dominado PyDash!")
                
                # Guardar resultados completos
                st.session_state.complete_results = {
                    'q_table': data['q_table'],
                    'rewards': data['rewards'],
                    'max_distances': data['max_distances'],
                    'final_epsilon': data['final_epsilon']
                }
                
                # Crear gráficos finales más detallados
                create_final_enhanced_plots(data, reward_chart)
                
                # Mostrar último frame
                if last_frame:
                    game_display.image(last_frame, 
                                     caption="🏆 ¡Entrenamiento Completado - Agente Entrenado!", 
                                     use_container_width=True)
                
                # Crear botón de descarga mejorado
                create_enhanced_download_button(data, episodes)
                
                break
            
            elif data['type'] == 'frame':
                # Actualizar visualización en tiempo real
                ep = data['episode']
                reward = data['reward']
                action = data['action']
                distance = data['distance']
                survival_time = data['survival_time']
                q_size = data['q_table_size']
                
                last_frame = data['frame']
                game_display.image(data['frame'], 
                                 caption=f"🎮 Episodio {ep} - Cubo Aprendiendo en Tiempo Real", 
                                 use_container_width=True)
                
                # Información detallada del juego
                with game_info.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🎯 Acción", "🦘 SALTAR" if action == 1 else "🏃 CORRER")
                    col2.metric("🏆 Reward", f"{reward:.1f}")
                    col3.metric("📏 Distancia", f"{distance:.0f}")
                    col4.metric("🧠 Estados", q_size)
            
            elif data['type'] == 'progress':
                # Actualizar progreso general
                ep = data['episode']
                reward = data['reward']
                epsilon_val = data['epsilon']
                max_distance = data['max_distance']
                avg_reward = data['avg_reward_last_10']
                
                progress = ep / episodes
                progress_bar.progress(progress)
                
                # Estado detallado
                status_text.info(
                    f"🎯 Episodio {ep}/{episodes} ({progress*100:.1f}%) | "
                    f"🏆 Reward: {reward:.1f} | "
                    f"📈 Promedio: {avg_reward:.1f} | "
                    f"📏 Distancia Máx: {max_distance:.0f} | "
                    f"🔍 Exploración: {epsilon_val:.3f}"
                )
                
                # Actualizar listas y gráficos
                rewards.append(reward)
                episodes_list.append(ep)
                max_distances.append(max_distance)
                
                # Actualizar gráfico cada 30 episodios
                if len(rewards) > 1 and ep % 30 == 0:
                    create_progress_enhanced_plots(episodes_list, rewards, max_distances, reward_chart)
                
                # Estadísticas en tiempo real
                with stats_container.container():
                    if len(rewards) >= 10:
                        recent_avg = np.mean(rewards[-10:])
                        improvement = rewards[-1] - rewards[0]
                        best_reward = max(rewards)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("📈 Promedio (10 últ.)", f"{recent_avg:.1f}")
                        col2.metric("🚀 Mejora Total", f"{improvement:.1f}")
                        col3.metric("🏆 Mejor Reward", f"{best_reward:.1f}")
        
        except queue.Empty:
            continue
        except Exception as e:
            st.error(f"Error en entrenamiento: {e}")
            break
    
    training_thread.join(timeout=10)

def create_final_enhanced_plots(data, reward_chart):
    """Crear gráficos finales mejorados"""
    rewards = data['rewards']
    max_distances = data.get('max_distances', [])
    
    if not rewards:
        return
    
    episodes_range = range(1, len(rewards) + 1)
    
    # Crear figura con subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Evolución del Reward', 'Distancia Máxima Alcanzada'),
        vertical_spacing=0.12
    )
    
    # Gráfico de rewards
    fig.add_trace(
        go.Scatter(x=list(episodes_range), y=rewards, mode='lines', 
                  name='Reward', line=dict(color='lightblue', width=1)),
        row=1, col=1
    )
    
    # Media móvil de rewards
    if len(rewards) >= 20:
        moving_avg = pd.Series(rewards).rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(x=list(episodes_range), y=moving_avg, mode='lines',
                      name='Media Móvil (20)', line=dict(color='blue', width=3)),
            row=1, col=1
        )
    
    # Gráfico de distancias si disponible
    if max_distances:
        fig.add_trace(
            go.Scatter(x=list(episodes_range), y=max_distances, mode='lines',
                      name='Distancia Máx.', line=dict(color='green', width=2)),
            row=2, col=1
        )
    
    fig.update_layout(
        title='🎯 Análisis Completo del Entrenamiento',
        height=600,
        showlegend=True
    )
    
    reward_chart.plotly_chart(fig, use_container_width=True)

def create_progress_enhanced_plots(episodes_list, rewards, max_distances, reward_chart):
    """Crear gráficos de progreso mejorados"""
    if len(rewards) < 2:
        return
    
    fig = go.Figure()
    
    # Reward principal
    fig.add_trace(go.Scatter(
        x=episodes_list, y=rewards, mode='lines+markers',
        name='Reward', line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Media móvil
    if len(rewards) >= 10:
        window = min(20, len(rewards)//2)
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=episodes_list, y=moving_avg, mode='lines',
            name=f'Media Móvil ({window})', line=dict(color='red', width=3)
        ))
    
    # Línea de tendencia
    if len(rewards) >= 20:
        z = np.polyfit(episodes_list, rewards, 1)
        trend_line = np.poly1d(z)(episodes_list)
        fig.add_trace(go.Scatter(
            x=episodes_list, y=trend_line, mode='lines',
            name='Tendencia', line=dict(color='orange', dash='dash', width=2)
        ))
    
    fig.update_layout(
        title=f'📊 Progreso del Entrenamiento - Episodio {episodes_list[-1]}',
        xaxis_title='Episodio',
        yaxis_title='Reward',
        template='plotly_white',
        hovermode='x unified'
    )
    
    reward_chart.plotly_chart(fig, use_container_width=True)

def create_enhanced_download_button(data, episodes):
    """Crear botón de descarga con datos completos"""
    try:
        rewards = data['rewards']
        max_distances = data.get('max_distances', [0] * len(rewards))
        
        # Crear DataFrame completo
        results_df = pd.DataFrame({
            'Episodio': range(1, len(rewards) + 1),
            'Reward': rewards,
            'Distancia_Maxima': max_distances,
            'Media_Movil_20': pd.Series(rewards).rolling(20).mean(),
            'Mejora_Acumulada': np.array(rewards) - rewards[0],
            'Percentil_Episode': [(i+1)/len(rewards)*100 for i in range(len(rewards))]
        })
        
        # Agregar estadísticas de resumen
        summary_stats = {
            'Total_Episodios': len(rewards),
            'Reward_Inicial': rewards[0],
            'Reward_Final': rewards[-1],
            'Mejor_Reward': max(rewards),
            'Mejora_Total': rewards[-1] - rewards[0],
            'Epsilon_Final': data.get('final_epsilon', 'N/A'),
            'Estados_Q_Aprendidos': len(data['q_table'])
        }
        
        # Crear CSV con datos y resumen
        csv_data = results_df.to_csv(index=False)
        csv_data += "\n\n# RESUMEN DEL ENTRENAMIENTO\n"
        for key, value in summary_stats.items():
            csv_data += f"# {key}: {value}\n"
        
        st.download_button(
            label="📥 Descargar Resultados Completos del Entrenamiento",
            data=csv_data,
            file_name=f"pydash_complete_training_{episodes}ep_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Descarga todos los datos del entrenamiento incluyendo estadísticas detalladas"
        )
        
        # Botón adicional para la Q-table
        q_table_data = "Estado,Accion_No_Saltar,Accion_Saltar,Mejor_Accion\n"
        for state, values in data['q_table'].items():
            best_action = "Saltar" if values[1] > values[0] else "No_Saltar"
            q_table_data += f'"{state}",{values[0]:.4f},{values[1]:.4f},{best_action}\n'
        
        st.download_button(
            label="🧠 Descargar Tabla Q (Conocimiento Aprendido)",
            data=q_table_data,
            file_name=f"pydash_q_table_{episodes}ep_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Descarga la tabla Q con todo el conocimiento que aprendió el agente"
        )
        
    except Exception as e:
        st.error(f"Error creando descarga: {e}")

if __name__ == "__main__":
    main()