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
episodes = 500
max_steps = 5000
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
def get_state(player, player2, next_obstacle_type):
    return (player2.rect.x // 32,  # Posición horizontal del jugador2
            player.rect.y // 32,   # Posición vertical del jugador1
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

def step_env(env, player2, render=False):
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
    next_state = get_state(player,player2, next_obs_type)

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
# ---------------- Función para ejecutar experimentos ----------------

def run_experiment(alpha_val, gamma_val, epsilon_decay_val, eps_start=0.5, label="exp"):
    global alpha, gamma, epsilon, q_table, reward_log

    # Asigna los parámetros del experimento
    alpha = alpha_val
    gamma = gamma_val
    epsilon = eps_start
    reward_log = []
    q_table = defaultdict(lambda: [0.0, 0.0])  # Reinicia Q-table

    for ep in range(episodes):
        # Reiniciar entorno
        elements = pygame.sprite.Group()
        player = Player(avatar, elements, (150,150))
        player2 = Player2(avatar, elements, (150,150))
        leveldata = block_map("level_2.csv")
        init_level(leveldata, elements)
        obs_type = find_next_obstacle(player, elements)
        state = get_state(player, player2, obs_type)
        total_r = 0

        render = False  # No renderizar durante los experimentos

        for s in range(max_steps):
            # Manejo de eventos (cierre de ventana)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # Ejecuta paso
            state = get_state(player, player2, obs_type)
            action, reward, next_state, done = step_env((state, elements, player), player2, render)
            total_r += reward

            # Actualiza Q-table
            old = q_table[state][action]
            q_table[state][action] = (1 - alpha) * old + alpha * (reward + gamma * max(q_table[next_state]))

            state = next_state
            if done: break

        reward_log.append(total_r)
        
        # Decaimiento de epsilon para controlar exploración-explotación
        epsilon = max(0.01, epsilon * epsilon_decay_val)

        print(f"{label} | Ep {ep+1}/{episodes}, reward={total_r:.2f}")

    # ------------- Guardar Resultados ----------------

    df = pd.DataFrame({"Episode": range(1, episodes+1), "Reward": reward_log})
    df.to_csv(f"results_{label}.csv", index=False)

    import csv
    # Guardar la tabla Q en un archivo CSV
    with open(f"q_table_{label}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["State", "Action 0 (No saltar)", "Action 1 (Saltar)"])
        for state, q_values in q_table.items():
            writer.writerow([state] + q_values)


