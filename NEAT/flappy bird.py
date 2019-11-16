# C:\Users\Micha\Anaconda3\envs\tensorflow\Lib\site-packages\retro
from nes_py.wrappers import JoypadSpace
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import cv2
import neat
import pickle
import gym
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward.txt')

game = FlappyBird()
imgarray = []
xpos_end = 0

resume = False
restore_file = "neat-checkpoint-678"


def eval_genomes(genomes, config):
    done = [False] * len(genomes)
    pl = []
    for i in range(len(genomes)):
        pl.append(PLE(game, fps=30, display_screen=True, force_fps=False))
        pl[i].init()
    while sum(done) != len(done):
        if len(pl) < len(genomes):
            pl.append(PLE(game, fps=30, display_screen=True, force_fps=False))
            done = done + [False]
            pl[-1].init()
        m = 0
        nets = []
        gid = []

        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            nets.append(net)
            gid.append(genome_id)

        nnOutput = [0] * len(genomes)
        rew = [0] * len(genomes)
        current_max_fitness = [0] * len(genomes)
        fitness_current = [0] * len(genomes)
        frame = [0] * len(genomes)
        counter = [0] * len(genomes)
        for i in range(len(genomes)):
            ob = list(np.zeros([288, 512, 3]) * len(genomes))
            ob.append(pl[i].getScreenRGB())
            frame[i] += 1
            ob[i] = cv2.resize(ob[i], (int(ob[i].shape[0]/8), int(ob[i].shape[1]/8)))
            ob[i] = cv2.cvtColor(ob[i], cv2.COLOR_BGR2GRAY)
            ob[i] = np.reshape(ob[i], (int(ob[i].shape[0]), int(ob[i].shape[1])))

            imgarray = np.ndarray.flatten(ob[i])

            nnOutput.append(np.argmax(nets[i].activate(imgarray)))
            rew[i] = (pl[i].act(119*np.argmax(nnOutput[i])))
            done[i] = pl[i].game_over()  # check if the game is over

            fitness_current[i] += float(rew[i])
            if fitness_current[i] > current_max_fitness[i]:
                current_max_fitness[i] = float(fitness_current[i])
                counter[i] = 0
            else:
                counter[i] += 1
            if sum(done) == len(done):
                m += config.pop_size - 1
                # print(gid[i], fitness_current)
                for k in range(len(pl)):
                    pl[k].reset_game()
            print(len(p.population), i+1)
            p.population[i+1].fitness = float(fitness_current[i])





if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.population.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(25))


winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
