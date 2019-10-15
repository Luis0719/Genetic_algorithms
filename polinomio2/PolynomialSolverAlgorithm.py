'''
    Owner: Luis Eduardo Hernandez Ayala
    Email: luis.hdez97@hotmail.com
    Python Version: 3.7.x, but shouldn't have problems with 2.7.x
'''

from math import sqrt, cos, sin
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    plt_found = True
except:
    plt_found = False

logger = open('logs.log', 'w')


A = 0
B = 1
C = 2
D = 3
E = 4
F = 5

    
class PSmodel():
    def __init__(self, population_size, target_values, resolution=50, generations=100, competidors_percentage=0.05, gen_bit_length=8, x_bottom_limit=0, x_top_limit=100, x_step=0.1, debuglevel=0):
        # Validate population size to be an even number
        if population_size % 2 == 1:
            raise Exception("Population size must be an even number")
        self.population_size = population_size

        # Validate the competidors percentage to be between 1 and 100
        if competidors_percentage < 0.1 and competidors_percentage > 0.99:
            raise Exception("Competidor percentage must be a number between 0.1 and 0.99")
        self.competidors_percentage = competidors_percentage

        self.cromosome_size = len(target_values)
        self.generations = generations
        self.gen_bit_length = gen_bit_length
        self.resolution = resolution
        self.debuglevel = debuglevel
        self.x_bottom_limit = x_bottom_limit
        self.x_top_limit = x_top_limit
        self.x_step = x_step

        self.population = None
        self.fittiests_history = None
        self.fittiest = None

        self.target = self.get_function_values(target_values)


    @staticmethod
    def get_y(x, cromosome):
        return cromosome[A] * (cromosome[B] * sin(x/cromosome[C]) + cromosome[D] * sin(x/cromosome[E])) + cromosome[F] * x - cromosome[D]
        # return cromosome[A] * sin(x) + cromosome[B]


    def resolutionate(self, values):
        result = [0] * len(values)
        for i in range(len(values)):
            result[i] = values[i] / self.resolution

        return result


    def get_function_values(self, cromosome):
        result = [0] * (self.x_top_limit - self.x_bottom_limit)

        resolutionated_cromosome = self.resolutionate(cromosome)
        for x in range(self.x_bottom_limit, self.x_top_limit):
            result[x] = PSmodel.get_y(x*self.x_step, resolutionated_cromosome)

        return result
    

    def evaluate(self, cromosome):
        error = 0
        cromosome_values = self.get_function_values(cromosome)
        for i in range(len(self.target)):
            error += abs(cromosome_values[i] - self.target[i])
        
        return error


    def log(self, text, debuglevel=0, logtype="INFO"):
        if self.debuglevel <= debuglevel:
            msg = "{} - {}".format(logtype, text)
            print(msg)
            logger.write(msg + '\n')     


    def plot_functions(self, generation_fittiest, generation):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.target)
        axs[0].plot(self.get_function_values(generation_fittiest))
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].grid(True)

        # fig.tight_layout()
        fig.figsize=(20,21)
        plt.title(f"Generation {generation}")
        plt.show()
        print(generation_fittiest)
        

    def generate_population(self):  
        '''
            Generate a population where each individual consists of an array of 3 numbers which can go from 0 to 2^gen_bit_length-1 (gen_bit_length is received as a parameter, default=8)
        '''
        max_value = 2 ** self.gen_bit_length
        temp_population = [0] * self.population_size
        for i in range(self.population_size):
            temp_population[i] = [0] * (self.cromosome_size + 1)
            for j in range(self.cromosome_size):
                temp_population[i][j] = random.randrange(1, max_value)

        return temp_population


    def get_fittiest(self):
        fittiest = None
        for cromosome in self.population:
            if fittiest == None or cromosome[-1] < fittiest[-1]: 
                fittiest = cromosome

        return fittiest


    def calculate_aptitud_function(self):
        '''
            Calculate the aptitud function for each cromosome and store the result in the last position of the list
        '''
        for cromosome in self.population:
            cromosome[-1] = self.evaluate(cromosome)


    def tournament_compete(self, total_competidors):
        '''
            Get an 'n' number of 'randomly' selected cromosomes and get the one with the lowest value of the aptitud function
            'n' is given by the parameter 'total_competidors'
        '''
        self.log("-----start of tournament with {:d} competidors-----".format(total_competidors))
        winner = None
        for i in range(total_competidors):
            rndindex = random.randrange(self.population_size)
            self.log("{}".format(str(self.population[rndindex])))

            # If there's still no winner (first item in the loop) just assign it to be the temporal winner
            if winner == None or self.population[rndindex][-1] < winner[-1]:
                winner = self.population[rndindex]
                continue

        self.log("Winner = {}".format(str(winner)))
        self.log("-----end of tournament-----")
        return winner


    def tournament(self):
        '''
            Make a tournament to get a new list of 'n' cromosomes. 'n' is the total population size / 2. 
            The self.logic to get the cromosomes for this new list is delegated to the method 'tournament_compete'
        '''
        total_competidors = int(self.population_size * self.competidors_percentage)
        total_winners = int(self.population_size / 2)

        winners = [0] * total_winners
        for i in range(total_winners):
            winners[i] = self.tournament_compete(total_competidors)

        self.log('------winners tournament----')
        self.print_population(winners)
        return winners

    @staticmethod
    def is_valid_child(cromosome):
        # Because last position is always 0, we shouldn't compare this last index of the array
        return not 0 in cromosome[1:-1]

    def breeding_operator1(self, father, mother):
        '''
            This breeding method will get a pivot randomely and will use it to 'break' each cromosome (father's and mother's cromosomes)
            Child1 will consist on father's binary value from position 0 to pivot, and mother's binary value from pivot to last position
            Child2, on the other hand, will consist on mothers's binary value from position 0 to pivot, and fathers's binary value from pivot to last position
        '''
        def cromosome_to_binary(cromosome, gen_bit_length):
            result = ""
            format_schema = "{0:" + "{0:02d}".format(gen_bit_length) + "b}"
            for i in range(len(cromosome)-1):
                result += format_schema.format(cromosome[i])
            return result

        def binary_to_cromosome(binary, gen_bit_length):
            result = []
            for i in range(0, len(binary), gen_bit_length):
                result.append(int(binary[i:i+gen_bit_length], 2))

            return result


        # Get random pivot which will divide the cromosome
        pivot = random.randrange(1, self.gen_bit_length * self.cromosome_size)

        # Convert each cromsome to it's binary equivalent. This get the binary value of each gen and will merge them into one
        father_binary = cromosome_to_binary(father, self.gen_bit_length)
        mother_binary = cromosome_to_binary(mother, self.gen_bit_length)

        # Do the breeding
        child1_binary = father_binary[:pivot] + mother_binary[pivot:]
        child2_binary = mother_binary[:pivot] + father_binary[pivot:]

        # Split the final binary value into each gen
        child1 = binary_to_cromosome(child1_binary, self.gen_bit_length) + [0]
        child2 = binary_to_cromosome(child2_binary, self.gen_bit_length) + [0]

        self.log("{} & {} ({:d})= {} & {}".format(str(father), str(mother), pivot, str(child1), str(child2)), 2)
        
        return child1, child2


    def breed(self, fathers, mothers):
        '''
            select the breeding method
        '''
        self.log('------breeding-------')
        newpopulation = [0] * self.population_size

        for i in range(int(self.population_size/2)):
            valid_children = False
            while not valid_children:
                child1, child2 = self.breeding_operator1(fathers[i], mothers[i])
                valid_children = PSmodel.is_valid_child(child1) and PSmodel.is_valid_child(child2)

            newpopulation[i*2], newpopulation[i*2+1] = child1, child2

        self.log('------end of breeding-------')
        return newpopulation


    def graph_history(self):
        if not plt_found:
            print("Matplotlib libary not found. Install it to see the graph. Install: 'pip install matplotlib'")
            return

        plt.rcParams.update({'font.size': 6})
        plt.plot([i+1 for i in range(self.generations)], self.fittiests_history)
        plt.ylabel('Error')
        plt.xlabel('Generations')
        plt.show()

    
    def fit(self):
        '''
            Main method of the algorithm. This method will coordinate each step to make the ordinary genetic algorithm work
        '''
        # Initialize aptituds list. This will store all the values of the aptitud function
        self.aptituds = [0] * self.population_size
        self.fittiests_history = []

        # Get initial population randomly and store it in the global variable
        self.population = self.generate_population()

        # Calculate the aptitud function for each cromosome
        self.calculate_aptitud_function()

        # Get fittiest from the initial generation to make the graph
        # fittiest_aptitud, fittiest = self.get_fittiest()
        # self.fittiests_history["0"] = fittiest_aptitud
        
        for i in range(self.generations):    
            # Get a new list of cromosomes with a tournament
            fathers = self.tournament()
            mothers = self.tournament()
            self.log('----fathers----', 1)
            self.print_population(fathers, 1)
            self.log('----mothers----', 1)
            self.print_population(mothers, 1)

            # from the winners of the tournament, get new cromosomes by a 'breeding' process and overwrite the actual population with the new population originated from the 'winners'
            self.population = self.breed(fathers, mothers)

            # Calculate the aptitud function for each new cromosome
            self.calculate_aptitud_function()

            # Get fittiest to make the graph
            generation_fittiest = self.get_fittiest()
            self.fittiests_history.append(generation_fittiest[-1])

            # self.graph_history()
            self.log("---------Generation {:d}".format(i+1), 2)
            self.print_population(debuglevel=2)
            self.log("----fittiest = {}".format(str(generation_fittiest)), 2)
            self.plot_functions(generation_fittiest, i)

        self.log(self.fittiests_history, 2)
        self.graph_history()
        self.fittiest = fittiest

        return fittiest


    def print_population(self, population=None, debuglevel=0):
        '''
            Used for debug
        '''
        if population == None:
            population = self.population

        for i in range(len(population)):
            self.log(str(population[i]), debuglevel)


    def tester(self, args1, args2):
        pass