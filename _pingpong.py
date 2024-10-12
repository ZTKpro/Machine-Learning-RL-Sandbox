import pygame
import random
import numpy as np
import pickle

# Game initialization
pygame.init()

# Game window settings
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WINDOW = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Ping Pong with AI")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RAINBOW_COLORS = [
    (255, 0, 0),  # Red
    (255, 127, 0),  # Orange
    (255, 255, 0),  # Yellow
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (75, 0, 130),  # Indigo
    (148, 0, 211)  # Violet
]

# Paddle and ball parameters
PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_SIZE = 20

# Paddle speed
PADDLE_SPEED = 8

# Ball speed
INITIAL_BALL_SPEED_X = 15
INITIAL_BALL_SPEED_Y = 15

# Q-learning parameters
LEARNING_RATE = 1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995

# Q-table initialization
q_table = {}

def get_discrete_state(player_y, ai_y, ball_x, ball_y, ball_speed_x, ball_speed_y):
    return (int(player_y), int(ai_y), int(ball_x), int(ball_y), int(ball_speed_x), int(ball_speed_y))

# Initial positions
def initialize_positions():
    player_x = 50
    player_y = (SCREEN_HEIGHT // 2) - (PADDLE_HEIGHT // 2)
    ball_x = (SCREEN_WIDTH // 2) - (BALL_SIZE // 2)
    ball_y = (SCREEN_HEIGHT // 2) - (BALL_SIZE // 2)
    ball_speed_x = INITIAL_BALL_SPEED_X * random.choice([-1, 1])
    ball_speed_y = INITIAL_BALL_SPEED_Y * random.choice([-1, 1])
    ai_paddles = []
    ai_x = SCREEN_WIDTH - 50 - PADDLE_WIDTH  # All AI paddles have the same x position
    for i in range(7):  # Adding 7 AI paddles with rainbow colors
        ai_y = random.randint(0, SCREEN_HEIGHT - PADDLE_HEIGHT)
        ai_color = RAINBOW_COLORS[i % len(RAINBOW_COLORS)]
        ai_paddles.append({'x': ai_x, 'y': ai_y, 'color': ai_color, 'best_score': -np.inf, })
    return player_x, player_y, ball_x, ball_y, ball_speed_x, ball_speed_y, ai_paddles

player_x, player_y, ball_x, ball_y, ball_speed_x, ball_speed_y, ai_paddles = initialize_positions()

# Handling AI movement
def handle_ai_movement(ai_paddle, action):
    if action == 0 and ai_paddle['y'] > 0:
        ai_paddle['y'] -= PADDLE_SPEED
    elif action == 1 and ai_paddle['y'] < SCREEN_HEIGHT - PADDLE_HEIGHT:
        ai_paddle['y'] += PADDLE_SPEED

# Handling ball movement
def handle_ball_movement(ball_x, ball_y, ball_speed_x, ball_speed_y):
    ball_x += ball_speed_x
    ball_y += ball_speed_y
    # Collision with top and bottom walls
    if ball_y <= 0 or ball_y >= SCREEN_HEIGHT - BALL_SIZE:
        ball_speed_y *= -1
    return ball_x, ball_y, ball_speed_x, ball_speed_y

# Handling collisions with paddles
def handle_paddle_collisions(player_x, player_y, ai_paddles, ball_x, ball_y, ball_speed_x):
    # Collision with player paddle
    if (player_x < ball_x < player_x + PADDLE_WIDTH and player_y < ball_y < player_y + PADDLE_HEIGHT):
        ball_speed_x *= -1
        return ball_speed_x, True, 'player'
    # Collision with AI paddles
    for ai_paddle in ai_paddles:
        if (ai_paddle['x'] < ball_x + BALL_SIZE < ai_paddle['x'] + PADDLE_WIDTH and ai_paddle['y'] < ball_y < ai_paddle['y'] + PADDLE_HEIGHT):
            ball_speed_x *= -1
            return ball_speed_x, True, ai_paddle
    return ball_speed_x, False, None

# Resetting the ball position after a point is scored
def reset_ball():
    return (SCREEN_WIDTH // 2) - (BALL_SIZE // 2), (SCREEN_HEIGHT // 2) - (BALL_SIZE // 2), INITIAL_BALL_SPEED_X * random.choice([-1, 1]), INITIAL_BALL_SPEED_Y * random.choice([-1, 1])

# Main game loop
speed_increase_interval = 5000  # Interval in milliseconds to increase ball speed
last_speed_increase_time = pygame.time.get_ticks()
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Player movement - player paddle follows the ball
    player_y = ball_y - (PADDLE_HEIGHT // 2)
    player_y = max(0, min(SCREEN_HEIGHT - PADDLE_HEIGHT, player_y))

    # Ball movement
    ball_x, ball_y, ball_speed_x, ball_speed_y = handle_ball_movement(ball_x, ball_y, ball_speed_x, ball_speed_y)

    # Paddle collisions
    ball_speed_x, collision, who_hit = handle_paddle_collisions(player_x, player_y, ai_paddles, ball_x, ball_y, ball_speed_x)

    # Discrete state for Q-learning
    for ai_paddle in ai_paddles:
        state = get_discrete_state(player_y, ai_paddle['y'], ball_x, ball_y, ball_speed_x, ball_speed_y)
        if state not in q_table:
            q_table[state] = [0, 0]  # Actions: 0 - move up, 1 - move down

        # Choose action using epsilon-greedy policy
        if np.random.random() > EXPLORATION_RATE:
            action = np.argmax(q_table[state])
        else:
            action = np.random.choice([0, 1])

        # AI movement
        handle_ai_movement(ai_paddle, action)

        # Get new state after action
        new_state = get_discrete_state(player_y, ai_paddle['y'], ball_x, ball_y, ball_speed_x, ball_speed_y)
        if new_state not in q_table:
            q_table[new_state] = [0, 0]

        # Update Q-value
        max_future_q = np.max(q_table[new_state])
        current_q = q_table[state][action]
        reward = 0  # Default reward is 0

        # Reward when AI is aligned with the ball in y-axis
        if abs(ai_paddle['y'] + PADDLE_HEIGHT / 2 - ball_y) < PADDLE_HEIGHT / 2:
            reward += 0.1
        else:
            reward -= 0.1

        # Reward for hitting the ball
        if collision and who_hit == ai_paddle and ball_speed_x > 0:  # Specific AI successfully hits the ball
            reward += 10
            
        q_table[state][action] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)

    # Decay exploration rate
    EXPLORATION_RATE *= EXPLORATION_DECAY
    EXPLORATION_RATE = max(0.01, EXPLORATION_RATE)

    # Increase ball speed over time
    current_time = pygame.time.get_ticks()
    if current_time - last_speed_increase_time > speed_increase_interval:
        ball_speed_x *= 1.1
        ball_speed_y *= 1.1
        last_speed_increase_time = current_time

    # Drawing game elements
    WINDOW.fill(BLACK)
    pygame.draw.rect(WINDOW, WHITE, (player_x, player_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    for ai_paddle in ai_paddles:
        pygame.draw.rect(WINDOW, ai_paddle['color'], (ai_paddle['x'], ai_paddle['y'], PADDLE_WIDTH, PADDLE_HEIGHT))
        font = pygame.font.SysFont(None, 24)
        pygame.draw.ellipse(WINDOW, WHITE, (ball_x, ball_y, BALL_SIZE, BALL_SIZE))
    pygame.draw.aaline(WINDOW, WHITE, (SCREEN_WIDTH // 2, 0), (SCREEN_WIDTH // 2, SCREEN_HEIGHT))

    # Display AI paddle scores at the bottom of the screen
    for idx, ai_paddle in enumerate(ai_paddles):
        color_name = ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet"][idx]
        
    # Update the display
    pygame.display.flip()
    pygame.time.delay(30)

pygame.quit()