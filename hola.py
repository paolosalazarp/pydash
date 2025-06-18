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
alpha = 0.3     # tasa de aprendizaje
gamma = 0.9    # factor de descuento
epsilon = 0.1   # tasa de exploración
episodes = 500
max_steps = 5000
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
        self.jump_amount = 12
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

                if isinstance(p, Platform):

                    if yvel > 0:
                        """if player is going down(yvel is +)"""
                        self.rect.bottom = p.rect.top  # dont let the player go through the ground
                        self.vel.y = 0  # rest y velocity because player is on ground

                        # set self.onGround to true because player collided with the ground
                        self.onGround = True

                        # reset jump
                        self.isjump = False
                    elif yvel < 0:
                        """if yvel is (-),player collided while jumping"""
                        self.rect.top = p.rect.bottom  # player top is set the bottom of block like it hits it head
                    else:
                        """otherwise, if player collides with a block, he/she dies."""
                        self.vel.x = 0
                        self.rect.right = p.rect.left  # dont let player go through walls
                        self.died = True

    def update(self):
        if self.isjump and self.onGround:
            self.jump()
        if not self.onGround:
            self.vel += GRAVITY
            if self.vel.y > 100:
                self.vel.y = 100

    # Agregar movimiento en el eje X
        #self.rect.x += self.vel.x  # Mueve al jugador en el eje X

        self.collide(0)
        self.rect.top += self.vel.y  # Actualiza la posición en Y
        self.onGround = False
        self.collide(self.vel.y)

class Player2(pygame.sprite.Sprite):
    def __init__(self, image, platforms, pos, *groups):
        super().__init__(*groups)
        self.platforms = platforms
        self.image = pygame.transform.smoothscale(image,(32,32))
        self.rect = self.image.get_rect(center=pos)
        self.onGround = False
        self.vel = Vector2(0,0)
        self.isjump = False
        self.jump_amount = 12
        self.win = False
        self.died = False
        self.particles = []


    def update(self):

    # Agregar movimiento en el eje X
        self.rect.x += self.vel.x  # Mueve al jugador en el eje X


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
            if col == "End": End(fin, pos, elements)

# ---------------- Funciones del entorno ----------------

# Estado del entorno: posición redondeada + tipo de obstáculo + si está saltando
def get_state(player, next_obstacle_type):
    # Se actualiza el estado basado en la posición del jugador en múltiplos de 32
    return (player2.rect.x // 32,  # Dividir la posición por 32 para obtener el "cuadrado" del jugador
            player.rect.y//32,  # Mantener la precisión de la posición Y
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

import time  # Para poder utilizar time.sleep()

def step_env(env, render=False):
    state, elements, player = env
    
    # Política epsilon-greedy
    if random.random() < epsilon:
        action = random.choice(ACTIONS)  # Acción aleatoria (exploración)
    else:
        action = np.argmax(q_table[state])  # Acción según política aprendida (explotación)

    # Ejecuta acción
    player.isjump = (action == 1)
    player.vel.x = 6
    player2.vel.x = 6
    player.update()
    player2.update()
    
    for e in elements:
        e.rect.x -= player.vel.x

    # Renderiza el juego si se activa
    if render:
        screen.blit(bg, (0, 0))
        elements.draw(screen)
        screen.blit(player.image, player.rect)
        pygame.display.update()
        clock.tick(60)

    # Calcula recompensa basada en la distancia recorrida
    distance_reward = int(player.rect.x / 100)  # Recompensa por cada 100 píxeles avanzados

    # Recompensa por sobrevivir y llegar más lejos
    done = player.died or player.win or player.rect.x >= 800
    reward = distance_reward  # Recompensa por avanzar

    if player.died:
        reward = -20  # Penaliza si muere
    if player.win or player.rect.x >= 800: 
        reward = 50  # Recompensa grande por ganar el nivel

    if player.isjump and state[2] == 1:  # Si el siguiente obstáculo es un pincho y está saltando
        reward += 0.1

    # Nuevo estado
    next_obs_type = find_next_obstacle(player, elements)
    next_state = get_state(player, next_obs_type)

    if done:
        # Agregar un pequeño retraso para asegurar que los cálculos de la tabla Q se hagan antes de reiniciar
        time.sleep(0)  # Retraso de 1 segundo, puedes ajustarlo según sea necesario

    return action, reward, next_state, done




# ---------------- Cargar imágenes ----------------

font = pygame.font.SysFont("lucidaconsole", 20)
avatar = pygame.image.load(os.path.join("images", "avatar.png"))
fin = pygame.image.load(os.path.join("images", "fin.png"))
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
    player2 = Player2(avatar, elements, (150,150))
    leveldata = block_map("level_2.csv")
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
        state = get_state(player, obs_type)
        #time.sleep(0.1)
        #print(state)
        action, reward, next_state, done = step_env((state, elements, player), render)
        total_r += reward
        # Actualiza Q-table
        old = q_table[state][action]
        q_table[state][action] = (1 - alpha) * old + alpha * (reward + gamma * max(q_table[next_state]))
        state = next_state
        if done: break
    reward_log.append(total_r)
    print(f"Ep {ep+1}/{episodes}, reward={total_r:.2f}")
    epsilon = epsilon *0.99
# ------------- Guardar Resultados ----------------

df = pd.DataFrame({"Episode": range(1, episodes+1), "Reward": reward_log})
df.to_csv("pydash_rl_results.csv", index=False)
plt.plot(df.Episode, df.Reward)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training progress")
plt.show()

import csv

# Guardar la tabla Q en un archivo CSV
with open("q_table.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Escribir las cabeceras si deseas
    writer.writerow(["State", "Action 0 (No saltar)", "Action 1 (Saltar)"])
    
    # Iterar sobre cada estado y escribir las acciones correspondientes
    for state, q_values in q_table.items():
        # Escribir cada estado y sus valores Q correspondientes
        writer.writerow([state] + q_values)


