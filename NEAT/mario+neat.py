# C:\Users\Micha\Anaconda3\envs\tensorflow\Lib\site-packages\retro
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import cv2
import neat
import pickle

env = gym_super_mario_bros.make('SuperMarioBros-8-3-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
imgarray = []
xpos_end = 0

resume = True
restore_file = "neat-checkpoint-692"


def eval_genomes(genome, config):
    ob = env.reset()
    ac = env.action_space.sample()

    inx, iny, inc = env.observation_space.shape

    inx = int(inx / 8)
    iny = int(iny / 8)

    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

    current_max_fitness = 0
    fitness_current = 0
    frame = 0
    counter = 0
    xpos = 0

    done = False

    while not done:

        # env.render()
        frame += 1
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx, iny))

        imgarray = np.ndarray.flatten(ob)

        nnOutput = net.activate(imgarray)
        ob, rew, done, info = env.step(int(np.argmax(nnOutput)))

        if info['flag_get']:
            fitness_current += 10000000
            done = True

        fitness_current += rew

        if fitness_current > current_max_fitness:
            current_max_fitness = float(fitness_current)
            counter = 0
        else:
            counter += 1

        if done or counter == 250:
            done = True
            print(fitness_current)

        genome.fitness = float(fitness_current)
    return genome.fitness


config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward.txt')
def run():
    if resume == True:
        p = neat.Checkpointer.restore_checkpoint(restore_file)
    else:
        p = neat.population.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(25))


    pe = neat.ParallelEvaluator(12, eval_genomes)
    winner = p.run(pe.evaluate, 300)
    pe.stop()

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
if __name__ == '__main__':
    run()