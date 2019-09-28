'''
    Owner: Luis Eduardo Hernandez Ayala
    Email: luis.hdez97@hotmail.com
    Python Version: 3.7.x, but shouldn't have problems with 2.7.x
'''

from math import sqrt
from breeder import breeder_factory
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    plt_found = True
except:
    plt_found = False



def get_distance(p1, p2):
    '''
        return distance between 2 points
        distance = √((y2-y1)^2 + (x2-x1)^2)
    '''
    return sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)

def draw_route(route, title=""):
    fig, ax = plt.subplots()

    Path = mpath.Path
    path_data = [(Path.MOVETO, (route[0][0], route[0][1]))]
    for i in range(1, len(route)):
        path_data.append(
            (Path.LINETO, (route[i][0], route[i][1]))
        )

    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='w', alpha=1)
    ax.add_patch(patch)

    # plot control points and connecting lines
    x, y = zip(*path.vertices)
    line, = ax.plot(x, y, 'go-')

    ax.grid()
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    
    plt.title(title)
    plt.show()
    
class OGAmodel():
    def __init__(self, samples, population_size, generations):
        self.samples = samples
        self.population = None
        self.population_size = population_size
        self.generations = generations
        self.aptituds = None
        self.fittiests_history = None
        self.fittiest = None
    
    @classmethod
    def evaluate(cls, cromosome):
        '''
            Calculate the distance from one point to the next one and sum all the results. 
            Iterate through all the genes of the cromosome except the last one because it's the last gen and can't be compare to any 'next one'
        '''
        temp_sum = 0
        for i in range(len(cromosome)-1):
            temp_sum += get_distance(cromosome[i], cromosome[i+1])

        return temp_sum

    @classmethod
    def get_fittiest(cls, population, aptituds):
        fittiest = None
        fittiest_aptitud = 0
        for i in range(len(population)):
            if fittiest == None: 
                fittiest = population[i]
                fittiest_aptitud = aptituds[i]
                continue
            
            if aptituds[i] < fittiest_aptitud:
                fittiest = population[i]
                fittiest_aptitud = aptituds[i]

        return fittiest_aptitud, fittiest

    @classmethod
    def generate_population(cls, samples, population_size):  
        '''
            Take a list of samples which is all the posible genes in a chromosome and generate a 'n' number of new cromosomes which a 'random' order of the samples
            'n' is the size of the population
        '''  
        temp_population = [0] * population_size
        for i in range(population_size):
            # Use random library to generate a shuffled copy of the samples and add at the end the element '0' which will contain the 'aptitud function' value
            temp_population[i] = random.sample(samples, len(samples))

        return temp_population


    def aptitud_function(self):
        '''
            Calculate the aptitud function for each cromosome and store the result in the last position of the list
        '''
        for i in range(self.population_size):
            self.aptituds[i] = OGAmodel.evaluate(self.population[i])


    def tournament_compete(self, total_competidors):
        '''
            Get an 'n' number of 'randomly' selected cromosomes and get the one with the lowest value of the aptitud function
            'n' is given by the parameter 'total_competidors'
        '''
        winner = None
        winner_aptitud = None
        for i in range(total_competidors):
            rndindex = random.randrange(self.population_size)

            # If there's still no winner (first item in the loop) just assign it to be the temporal winner
            if winner == None:
                winner = self.population[rndindex]
                winner_aptitud = self.aptituds[rndindex]
                continue

            if self.aptituds[rndindex] < winner_aptitud:
                winner = self.population[rndindex]
                winner_aptitud = self.aptituds[rndindex]

        return winner


    def tournament(self, total_competidors):
        '''
            Make a tournament to get a new list of 'n' cromosomes. 'n' is the total population size. 
            The logic to get the cromosomes for this new list is delegated to the method 'tournament_compete'
        '''
        winners = [0] * self.population_size
        for i in range(self.population_size):
            winners[i] = self.tournament_compete(total_competidors)

        return winners


    def breed(self):
        '''
            'randomly' (50% chance) select a breeding method. 
        '''
        for i in range(self.population_size):
            rndnum = random.randrange(0,2)
            self.population[i] = breeder_factory(rndnum, self.population[i])
            self.aptituds[i] = 0


    def graph_history(self):
        if not plt_found:
            print("Matplotlib libary not found. Install it to see the graph. Install: 'pip install matplotlib'")
            return

        plt.rcParams.update({'font.size': 6})
        plt.plot(list(self.fittiests_history.keys()), list(self.fittiests_history.values()))
        plt.ylabel('Aptitud')
        plt.xlabel('Generations')
        plt.show()


    def fit(self, graph_routes=False):
        '''
            Main method of the algorithm. This method will coordinate each step to make the ordinary genetic algorithm work
        '''
        # Initialize aptituds list. This will store all the values of the aptitud function
        self.aptituds = [0] * self.population_size
        self.fittiests_history = {}

        # Get initial population randomly and store it in the global variable
        self.population = OGAmodel.generate_population(self.samples, self.population_size)

        # Calculate the aptitud function for each cromosome
        self.aptitud_function()

        # Get fittiest to make the graph
        fittiest_aptitud, fittiest = OGAmodel.get_fittiest(self.population, self.aptituds)
        self.fittiests_history["0"] = fittiest_aptitud
        
        for i in range(self.generations):    
            # Get a new list of cromosomes with a tournament
            self.population = self.tournament(total_competidors=5)

            # from the winners of the tournament, get new cromosomes by a 'breeding' process and overwrite the actual population with the new population originated from the 'winners'
            self.breed()

            # Calculate the aptitud function for each new cromosome
            self.aptitud_function()

            # Get fittiest to make the graph
            fittiest_aptitud, fittiest = OGAmodel.get_fittiest(self.population, self.aptituds)
            self.fittiests_history[str(i+1)] = fittiest_aptitud
               
            if graph_routes:
                draw_route(fittiest, title="Generation {:d}".format(i+1))

        self.fittiest = fittiest
        return fittiest


    def print_population(self):
        '''
            Used for debug
        '''
        for i in range(len(self.population)):
            print(str(self.population[i]) + " = " + str(self.aptituds[i]))