'''
    Owner: Luis Eduardo Hernandez Ayala
    Email: luis.hdez97@hotmail.com
    Python Version: 3.7.x, but shouldn't have problems with 2.7.x
'''

from math import sqrt
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    plt_found = True
except:
    plt_found = False

logger = open('logs.log', 'w')

    
class PSmodel():
    def __init__(self, population_size, generations=100, competidors_percentage=0.05, gen_bit_length=8, x_value=2, target_value=13, debuglevel=0):
        # Validate population size to be an even number
        if population_size % 2 == 1:
            raise Exception("Population size must be an even number")
        self.population_size = population_size

        # Validate the competidors percentage to be between 1 and 100
        if competidors_percentage < 0.1 and competidors_percentage > 0.99:
            raise Exception("Competidor percentage must be a number between 0.1 and 0.99")
        self.competidors_percentage = competidors_percentage

        self.population = None
        self.generations = generations
        self.aptituds = []    # Stores the aptitud values from each cromosome
        self.fittiests_history = None
        self.fittiest = None
        self.gen_bit_length = gen_bit_length
        self.x = x_value
        self.target = target_value
        self.debuglevel = debuglevel
    

    @classmethod
    def evaluate(cls, cromosome, x, target, resolution=50):
        '''
            Calculate the resulting absolute value for the equation Ax^2 + Bx + C - 13, where A,B and C are each value from the population's array. 
            Each cromosome consistes of an array in the form of: Cromosome = [A, B, C]
            The result will be the error from the result to the target
        '''
        A_result = (cromosome[0] / resolution) * (x**2)
        B_result = (cromosome[1] / resolution) * x
        C_result = (cromosome[2] / resolution)
        return abs(A_result + B_result + C_result - 13)


    def log(self, text, debuglevel=0, logtype="INFO"):
        if self.debuglevel <= debuglevel:
            msg = "{} - {}".format(logtype, text)
            print(msg)
            logger.write(msg + '\n')
        

    def generate_population(self):  
        '''
            Generate a population where each individual consists of an array of 3 numbers which can go from 0 to 2^gen_bit_length-1 (gen_bit_length is received as a parameter, default=8)
        '''
        max_value = 2 ** self.gen_bit_length
        temp_population = [0] * self.population_size
        for i in range(self.population_size):
            # Use random library to generate a shuffled copy of the samples and add at the end the element '0' which will contain the 'aptitud function' value
            temp_population[i] = [
                random.randrange(0, max_value),
                random.randrange(0, max_value),
                random.randrange(0, max_value)
            ]

        return temp_population


    def get_fittiest(self):
        fittiest = None
        fittiest_aptitud = 0
        for i in range(self.population_size):
            if fittiest == None: 
                fittiest = self.population[i]
                fittiest_aptitud = self.aptituds[i]
                continue
            
            if self.aptituds[i] < fittiest_aptitud:
                fittiest = self.population[i]
                fittiest_aptitud = self.aptituds[i]

        return fittiest_aptitud, fittiest


    def calculate_aptitud_function(self):
        '''
            Calculate the aptitud function for each cromosome and store the result in the last position of the list
        '''
        for i in range(self.population_size):
            self.aptituds[i] = PSmodel.evaluate(self.population[i], self.x, self.target)


    def tournament_compete(self, total_competidors):
        '''
            Get an 'n' number of 'randomly' selected cromosomes and get the one with the lowest value of the aptitud function
            'n' is given by the parameter 'total_competidors'
        '''
        self.log("-----start of tournament with {:d} competidors-----".format(total_competidors))
        winner = None
        winner_aptitud = None
        for i in range(total_competidors):
            rndindex = random.randrange(self.population_size)
            self.log("{} = {}".format(str(self.population[rndindex]), self.aptituds[rndindex]))

            # If there's still no winner (first item in the loop) just assign it to be the temporal winner
            if winner == None:
                winner = self.population[rndindex]
                winner_aptitud = self.aptituds[rndindex]
                continue

            if self.aptituds[rndindex] < winner_aptitud:
                winner = self.population[rndindex]
                winner_aptitud = self.aptituds[rndindex]

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


    def breeding_operator1(self, father, mother):
        '''
            This breeding method will get a pivot randomely and will use it to 'break' each cromosome (father's and mother's cromosomes)
            Child1 will consist on father's binary value from position 0 to pivot, and mother's binary value from pivot to last position
            Child2, on the other hand, will consist on mothers's binary value from position 0 to pivot, and fathers's binary value from pivot to last position
        '''
        def cromosome_to_binary(cromosome):
            result = ""
            for gen in cromosome:
                result += "{0:08b}".format(gen)
            return result

        def binary_to_cromosome(binary, gen_bit_length):
            result = []
            for i in range(0, len(binary), gen_bit_length):
                result.append(int(binary[i:i+gen_bit_length], 2))

            return result


        # Get random pivot which will divide the cromosome
        pivot = random.randrange(1, self.gen_bit_length*3)

        # Convert each cromsome to it's binary equivalent. This get the binary value of each gen and will merge them into one
        father_binary = cromosome_to_binary(father)
        mother_binary = cromosome_to_binary(mother)

        # Do the breeding
        child1_binary = father_binary[:pivot] + mother_binary[pivot:]
        child2_binary = mother_binary[:pivot] + father_binary[pivot:]

        # Split the final binary value into each gen
        child1 = binary_to_cromosome(child1_binary, self.gen_bit_length)
        child2 = binary_to_cromosome(child2_binary, self.gen_bit_length)

        self.log("{} & {} ({:d})= {} & {}".format(str(father), str(mother), pivot, str(child1), str(child2)), 2)
        
        return child1, child2


    def breeding_operator2(self, father, mother):
        '''
            This breeding method will get a pivot randomely and will use it to 'break' each cromosome (father's and mother's cromosomes)
            Child1 will consist on father's binary value from position 0 to pivot, and mother's binary value from pivot to last position
            Child2, on the other hand, will consist on mothers's binary value from position 0 to pivot, and fathers's binary value from pivot to last position
        '''
        def cromosome_to_binary(cromosome):
            result = ""
            for gen in cromosome:
                result += "{0:08b}".format(gen)
            return result

        def binary_to_cromosome(binary, gen_bit_length):
            result = []
            for i in range(0, len(binary), gen_bit_length):
                result.append(int(binary[i:i+gen_bit_length], 2))

            return result


        # Get random pivot which will divide the cromosome
        pivot1 = random.randrange(1, self.gen_bit_length*3)
        pivot2 = random.randrange(1, self.gen_bit_length*3)
        if pivot1 > pivot2:
            pivot1, pivot2 = pivot2, pivot1
            

        # Convert each cromsome to it's binary equivalent. This get the binary value of each gen and will merge them into one
        father_binary = cromosome_to_binary(father)
        mother_binary = cromosome_to_binary(mother)

        # Do the breeding
        child1_binary = father_binary[:pivot1] + mother_binary[pivot1:pivot2] + father_binary[pivot2:]
        child2_binary = mother_binary[:pivot1] + father_binary[pivot1:pivot2] + mother_binary[pivot2:]

        # Split the final binary value into each gen
        child1 = binary_to_cromosome(child1_binary, self.gen_bit_length)
        child2 = binary_to_cromosome(child2_binary, self.gen_bit_length)

        self.log("{} & {} ({:d}, {:d})= {} & {}".format(str(father), str(mother), pivot1, pivot2, str(child1), str(child2)))
        
        return child1, child2


    def breed(self, fathers, mothers):
        '''
            select the breeding method
        '''
        self.log('------breeding-------')
        newpopulation = [0] * self.population_size

        for i in range(int(self.population_size/2)):
            newpopulation[i*2], newpopulation[i*2+1] = self.breeding_operator1(fathers[i], mothers[i])

        self.log('------end of breeding-------')
        return newpopulation

    #TODO
    def graph_history(self):
        if not plt_found:
            print("Matplotlib libary not found. Install it to see the graph. Install: 'pip install matplotlib'")
            return

        plt.rcParams.update({'font.size': 6})
        plt.plot(list(self.fittiests_history.keys()), list(self.fittiests_history.values()))
        plt.ylabel('Error')
        plt.xlabel('Generations')
        plt.show()

    
    #TODO
    def fit(self):
        '''
            Main method of the algorithm. This method will coordinate each step to make the ordinary genetic algorithm work
        '''
        # Initialize aptituds list. This will store all the values of the aptitud function
        self.aptituds = [0] * self.population_size
        self.fittiests_history = {}

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
            self.aptituds = [0] * self.population_size  # Reset aptitud values

            # Calculate the aptitud function for each new cromosome
            self.calculate_aptitud_function()

            # Get fittiest to make the graph
            fittiest_aptitud, fittiest = self.get_fittiest()
            self.fittiests_history[str(i+1)] = fittiest_aptitud

            # self.graph_history()
            self.log("---------Generation {:d}".format(i+1), 2)
            self.print_population(debuglevel=2)
            self.log("----fittiest = {} : {}".format(str(fittiest), fittiest_aptitud), 2)

        print(self.fittiests_history)
        self.graph_history()
        self.fittiest = fittiest
        return fittiest


    #TODO
    def print_population(self, population=None, debuglevel=0):
        '''
            Used for debug
        '''
        if population == None:
            population = self.population

        for i in range(len(population)):
            self.log(str(population[i]) + " = " + str(self.aptituds[i]), debuglevel)


    def tester(self, args1, args2):
        self.breeding_operator1(args1, args2)