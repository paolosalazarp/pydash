##
# Reinforcement Learning integrated Pydash
# Author modificado

import csv, os, random, pygame, numpy as np
from collections import defaultdict
from pygame.math import Vector2
from pygame.draw import rect
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Game Setup ----------------
pygame.init()
screen = pygame.display.set_mode([800, 600])
pygame.display.set_caption('Pydash RL Agent')
clock = pygame.time.Clock()

# Colores
WHITE = (255,255,255); BLACK = (0,0,0); BLUE = (0,0,255)

# Superficie con canal alpha para partículas (no usada pero puede servir para mejoras)
alpha_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

# Constantes físicas
GRAVITY = Vector2(0, 0.86)

# ---------------- Agent / Env Setup ----------------
ACTIONS = [0, 1]  # 0: no saltar, 1: saltar
q_table = defaultdict(lambda: [0.0, 0.0])  # Q-table inicializada con ceros

# Parámetros de aprendizaje por refuerzo
alpha = 0.1     # tasa de aprendizaje
gamma = 0.9     # factor de descuento
epsilon = 0.1   # tasa de exploración
episodes = 500
max_steps = 500
reward_log = []

# Visualización (define desde qué episodio se quiere ver el juego)
show_from_episode = 490  # Cambia a None si no quieres ver ningún episodio

# -------------- Clases del Juego --------------

class Player(pygame.sprite.Sprite):
    def __init__(self, image, platforms, pos, *groups):
        super().__init__(*groups)
        self.platforms = platforms
        self.image = pygame.transform.smoothscale(image,(32,32))
        self.rect = self.image.get_rect(center=pos)
        self.onGround = False
        self.vel = Vector2(0,0)
        self.isjump = False
        self.jump_amount = 11
        self.win = False
        self.died = False
        self.particles = []

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
                if yvel > 0 and isinstance(p, Platform):
                    self.rect.bottom = p.rect.top
                    self.vel.y = 0
                    self.onGround = True
                    self.isjump = False

    def update(self):
        if self.isjump and self.onGround:
            self.jump()
        if not self.onGround:
            self.vel += GRAVITY
            if self.vel.y > 100:
                self.vel.y = 100
        self.collide(0)
        self.rect.top += self.vel.y
        self.onGround = False
        self.collide(self.vel.y)

# Clases de objetos del juego
class Draw(pygame.sprite.Sprite):
    def __init__(self, image, pos, *groups):
        super().__init__(*groups)
        self.image = image
        self.rect = image.get_rect(topleft=pos)

class Platform(Draw): pass
class Spike(Draw): pass
class Orb(Draw): pass
class End(Draw): pass

# ---------------- Cargar Nivel ----------------

def block_map(level_file):
    with open(level_file, newline='') as f:
        return [row for row in csv.reader(f)]

def init_level(mapdata, elements):
    for y, row in enumerate(mapdata):
        for x, col in enumerate(row):
            pos = (x*32, y*32)
            if col == "0": Platform(block, pos, elements)
            if col == "Spike": Spike(spike, pos, elements)
            if col == "Orb": Orb(orb, pos, elements)
            if col == "End": End(avatar, pos, elements)

# ---------------- Funciones del entorno ----------------

# Estado del entorno: posición redondeada + tipo de obstáculo + si está saltando
def get_state(player, next_obstacle_type):
    return (round(player.rect.x, -1),
            round(player.rect.y, -1),
            next_obstacle_type,
            int(player.isjump))

# Encuentra el siguiente obstáculo (Spike u Orb)
def find_next_obstacle(player, elements):
    objs = [e for e in elements if isinstance(e, Spike) or isinstance(e, Orb)]
    if not objs: return 0
    nxt = min(objs, key=lambda o: o.rect.x - player.rect.x if o.rect.x >= player.rect.x else 9999)
    if isinstance(nxt, Spike): return 1
    if isinstance(nxt, Orb): return 2
    return 0

# Paso del entorno: ejecuta acción, mueve sprites, devuelve recompensa y nuevo estado
def step_env(env, render=False):
    state, elements, player = env
    # Política epsilon-greedy
    if random.random() < epsilon:
        action = random.choice(ACTIONS)
    else:
        action = np.argmax(q_table[state])

    # Ejecuta acción
    player.isjump = (action == 1)
    player.vel.x = 6
    player.update()
    for e in elements:
        e.rect.x -= player.vel.x

    # Renderiza el juego si se activa
    if render:
        screen.blit(bg, (0, 0))
        elements.draw(screen)
        screen.blit(player.image, player.rect)
        pygame.display.update()
        clock.tick(60)

    # Calcula recompensa
    done = player.died or player.win or player.rect.x >= 800
    reward = 1
    if player.died: reward = -20
    if player.win or player.rect.x >= 800: reward = 50
    if player.isjump and state[2] == 1:
        reward += 0.1

    # Nuevo estado
    next_obs_type = find_next_obstacle(player, elements)
    next_state = get_state(player, next_obs_type)
    return action, reward, next_state, done

# ---------------- Cargar imágenes ----------------

font = pygame.font.SysFont("lucidaconsole", 20)
avatar = pygame.image.load(os.path.join("images", "avatar.png"))
pygame.display.set_icon(avatar)
spike = pygame.transform.smoothscale(pygame.image.load(os.path.join("images", "obj-spike.png")), (32,32))
block = pygame.transform.smoothscale(pygame.image.load(os.path.join("images", "block_1.png")), (32,32))
orb = pygame.transform.smoothscale(pygame.image.load(os.path.join("images", "orb-yellow.png")), (32,32))
bg = pygame.image.load(os.path.join("images", "bg.png"))

# --------------- Entrenamiento Principal ----------------

for ep in range(episodes):
    # Reiniciar entorno
    elements = pygame.sprite.Group()
    player = Player(avatar, elements, (150,150))
    leveldata = block_map("level_1.csv")
    init_level(leveldata, elements)
    obs_type = find_next_obstacle(player, elements)
    state = get_state(player, obs_type)
    total_r = 0

    # ¿Renderizar este episodio?
    render = show_from_episode is not None and ep+1 >= show_from_episode

    for s in range(max_steps):
        # Manejo de eventos (cierre de ventana)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        # Ejecuta paso
        action, reward, next_state, done = step_env((state, elements, player), render)
        total_r += reward
        # Actualiza Q-table
        old = q_table[state][action]
        q_table[state][action] = (1 - alpha) * old + alpha * (reward + gamma * max(q_table[next_state]))
        state = next_state
        if done: break

    reward_log.append(total_r)
    print(f"Ep {ep+1}/{episodes}, reward={total_r:.2f}")

# ------------- Guardar Resultados ----------------

df = pd.DataFrame({"Episode": range(1, episodes+1), "Reward": reward_log})
df.to_csv("pydash_rl_results.csv", index=False)
plt.plot(df.Episode, df.Reward)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training progress")
plt.show()
