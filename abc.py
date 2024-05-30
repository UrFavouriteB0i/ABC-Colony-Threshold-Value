import random
import sys
import copy
import numpy as np
import cv2


class Bee(object):
    """object bee(individu) secara umum"""

    def __init__(self, lower, upper, fun, funcon=None):
        """
        Parameter:
            Parameter list bawah : batas bawah vektor solusi
            Parameter list atas  : batas atas vektor solusi
            Parameter def fungsi : fungsi evaluasi
            Parameter def func konstrain : fungsi konstrain, harus return boolean
        """

        self._random(lower, upper)
        if not funcon:
            self.valid = True
        else:
            self.valid = funcon(self.vector)


        if (fun != None):
            self.value = fun(self.vector)
        else:
            self.value = sys.float_info.max
        self._fitness()


        self.counter = 0

    def _random(self, lower, upper):
        self.vector = []
        for i in range(len(lower)):
            self.vector.append( lower[i] + random.random() * (upper[i] - lower[i]) )

    def _fitness(self):
        if (self.value >= 0):
            self.fitness = 1 / (1 + self.value)
        else:
            self.fitness = 1 + abs(self.value)

class BeeHive(object):
    """
    Membuat algoritma Artificial Bee Colony (ABC).

    kelompok yang di inisiasi pada kelas ini meliputi:
        1. "employees",
        2. "onlookers",
        3. "scouts".
    """

    def run(self):
        cost = {}; cost["best"] = []; cost["mean"] = []
        for itr in range(self.max_itrs):
            for index in range(self.size):
                self.send_employee(index)

            self.send_onlookers()
            self.send_scout()
            self.find_best()

            for bee in self.population:
                bee._fitness()

            cost["best"].append( self.best )
            cost["mean"].append( sum( [ bee.value for bee in self.population ] ) / self.size )

            if self.verbose:
                self._verbose(itr, cost)
        return cost

    def __init__(self, lower, upper, fun = None, numb_bees =  30, max_itrs = 100, 
                 max_trials = None, selfun = None, seed = None, verbose = False, extra_params = None):

        assert (len(upper) == len(lower))

        if (seed == None):
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
        random.seed(self.seed)

        self.size = int((numb_bees + numb_bees % 2))
        self.dim = len(lower)
        self.max_itrs = max_itrs
        if (max_trials == None):
            self.max_trials = 0.6 * self.size * self.dim
        else:
            self.max_trials = max_trials
        
        self.selfun = selfun
        self.extra_params = extra_params
        self.evaluate = fun
        self.lower    = lower
        self.upper    = upper

        self.best = sys.float_info.max
        self.solution = None
        self.population = [ Bee(lower, upper, fun) for i in range(self.size) ]


        self.find_best()
        self.compute_probability()
        self.verbose = verbose

    def find_best(self):
        values = [ bee.value for bee in self.population ]
        index  = values.index(min(values))
        if (values[index] < self.best):
            self.best     = values[index]
            self.solution = self.population[index].vector

    def compute_probability(self):
        values = [bee.fitness for bee in self.population]
        max_values = max(values)

        if (self.selfun == None):
            self.probas = [0.9 * v / max_values + 0.1 for v in values]
        else:
            if (self.extra_params != None):
                self.probas = self.selfun(list(values), **self.extra_params)
            else:
                self.probas = self.selfun(values)

        return [sum(self.probas[:i+1]) for i in range(self.size)]

    def send_employee(self, index):
        zombee = copy.deepcopy(self.population[index])
        d = random.randint(0, self.dim-1)
        bee_ix = index;

        while (bee_ix == index): bee_ix = random.randint(0, self.size-1)
        zombee.vector[d] = self._mutate(d, index, bee_ix)
        zombee.vector = self._check(zombee.vector, dim=d)
        zombee.value = self.evaluate(zombee.vector)
        zombee._fitness()

        if (zombee.fitness > self.population[index].fitness):
            self.population[index] = copy.deepcopy(zombee)
            self.population[index].counter = 0
        else:
            self.population[index].counter += 1

    def send_onlookers(self):
        numb_onlookers = 0; beta = 0

        while (numb_onlookers < self.size):
            phi = random.random()
            beta += phi * max(self.probas)
            beta %= max(self.probas)
            index = self.select(beta)
            self.send_employee(index)
            numb_onlookers += 1

    def select(self, beta):
        probas = self.compute_probability()

        for index in range(self.size):
            if (beta < probas[index]):
                return index

    def send_scout(self):
        trials = [ self.population[i].counter for i in range(self.size) ]
        index = trials.index(max(trials))

        if (trials[index] > self.max_trials):
            self.population[index] = Bee(self.lower, self.upper, self.evaluate)
            self.send_employee(index)

    def _mutate(self, dim, current_bee, other_bee):

        return self.population[current_bee].vector[dim]    + \
               (random.random() - 0.5) * 2                 * \
               (self.population[current_bee].vector[dim] - self.population[other_bee].vector[dim])

    def _check(self, vector, dim=None):

        if (dim == None):
            range_ = range(self.dim)
        else:
            range_ = [dim]

        for i in range_:
            if  (vector[i] < self.lower[i]):
                vector[i] = self.lower[i]

            elif (vector[i] > self.upper[i]):
                vector[i] = self.upper[i]
        return vector

    def _verbose(self, itr, cost):
        #Digunakan untuk menampilkan hasil perhitungan nilai fitness pada proses pencarian nilai, sumber: Github
        fitness_values = [bee.value for bee in self.population]
        msg = "# Iter = {} | Best Evaluation Value = {} | Mean Evaluation Value = {} | Fitness Values: {}"
        print(msg.format(int(itr), cost["best"][itr], cost["mean"][itr], fitness_values))


# Otsu Thresholding secara dasar, sumber: medium.com
def threshold_image(im, th):
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1
    return thresholded_im

def compute_otsu_criteria(im, th):
    thresholded_im = threshold_image(im, th)
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1
    if weight1 == 0 or weight0 == 0:
        return np.inf
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    return weight0 * var0 + weight1 * var1


#Menjalankan fungsi keseluruhan
if __name__ == "__main__":
    image = cv2.imread('bnw.jpg', cv2.IMREAD_GRAYSCALE) #Saat ini dimasukan image yang langsung berupa grayscale format

    def evaluation_function(vector):
        threshold = int(vector[0])
        return compute_otsu_criteria(image, threshold)

    lower_bound = [0]
    upper_bound = [np.max(image)]
    hive = BeeHive(lower_bound, upper_bound, fun=evaluation_function, numb_bees=30, max_itrs=100, verbose=True)

    result = hive.run()
    best_threshold = hive.solution[0]
    print(f"Optimized Threshold: {best_threshold}")

    thresholded_image = threshold_image(image, best_threshold)

#Membandingkan hasil gambar masukan asli, dengan hasil yang telah di threshold
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Thresholded Image')
    plt.imshow(thresholded_image, cmap='gray')
    plt.show()
