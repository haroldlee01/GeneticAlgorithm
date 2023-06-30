import numpy as np
from numpy import array
import time

# define the number of uav, usv,task, the minimum number of uav and usv in a single task
uav_num = 60
usv_num = 30
task_num = 8
uav_min = 3
usv_min = 1

platform_cap = array([[0.16, 0.03], [0.03, 0.38], [0.10, 0.02]])

# define the travel cost per kilometer for a single platform, [uav, usv]
cost_per_km = array([[0.01], [0.03]])

# the cost of resource for detect, attack, evaluate
cost_rsrc = array([[1, 1, 1]]).T

# the value and resource requirement of each part of tasks: detect, attack, evaluate
task_val = array([[8, 9, 6, 8, 9, 9, 9, 8],
                    [18, 16, 8, 15, 12, 16, 12, 14],
                    [7, 4, 5, 7, 9, 6, 7, 5]])

task_rqt = array([[0.8, 0.9, 0.8, 0.8, 0.8, 0.8, 0.7, 0.6],
                      [0.7, 0.9, 0.3, 0.5, 0.6, 0.6, 0.8, 0.7],
                      [0.6, 0.6, 0.5, 0.7, 0.7, 0.6, 0.7, 0.4]])

task_distance = array([2, 3, 2, 3, 4, 2, 3, 3]).reshape(task_num,1)


# parameter for sigmoid function, f=1/(1+ exp(c(x-d)) )
C = -10
D = -0.8


# 种群规模,种群，下一代种群,是内容为 nd.array的列表,三维
pop_num = 10   # 数量为偶数
iter_num = 100   # 迭代次数
prob = 0.1   # 变异概率
best_fit = None
best_code = None
fits = []


# Generate a large number of random numbers between 0 and 1 at once for subsequent calls
cnt_rand_decimal = 0
rand_decimal = np.random.random(5000)
def get_rand_decimal(num):
    """ generate 5000 rand number between 0 and 1 to avoid pseudo-random numbers

    :param num: number of random number wanted
    :return: ndarray
    """
    global cnt_rand_decimal, rand_decimal

    if cnt_rand_decimal + num >= 5000:
        rand_decimal = np.random.random(5000)
        cnt_rand_decimal = 0

    tmp = cnt_rand_decimal
    cnt_rand_decimal += num

    return rand_decimal[tmp:cnt_rand_decimal]


# input the code of assignment scheme to determine whether it meets all requirements
def is_valid_code(code, utype):
    if utype  not in ['uav', 'usv', 'whole']:
        raise ValueError

    def is_right_code(code, num_sum, num_min):
        if sum(code) == num_sum and np.all(code>=num_min):
            return True
        return False

    flag = None
    if utype == 'uav':
        flag = is_right_code(code, uav_num, uav_min)
    elif utype == 'usv':
        flag = is_right_code(code, usv_num, usv_min)
    elif utype == 'whole':
        flag = is_right_code(code[0], uav_num, uav_min) and is_right_code(code[1], usv_num, usv_min)

    return flag


# Adjustment of income proportion
def sigmoid(pct):
    if np.isscalar(pct):
        return 1/(1 + np.power(np.e, C * (pct + D)))
    elif isinstance(pct,np.ndarray):
        if pct.ndim == 1:
            ratio = np.zeros((4,))
            for i in range(len(pct)):
                ratio[i] = 1/(1 + np.power(np.e, C * (pct[i] + D)))
            return ratio

        if pct.ndim == 2:
            m, n = np.shape(pct)
            ratio = np.zeros((m,n))
            for i in range(m):
                for j in range(n):
                    ratio[i][j] = 1/(1 + np.power(np.e, C * (pct[i][j] + D)))
            return ratio
    raise ValueError("Please check your data type")


# compute the fitness of the assignment scheme
def fitness_code(code):
    cost_distance_all = np.multiply(cost_per_km @ task_distance.T, code).sum()
    cost_rsrc_all = np.multiply(platform_cap @ code, cost_rsrc).sum()

    pct = platform_cap @ code / task_rqt
    ratio = sigmoid(pct)
    reward = (task_val * ratio).sum()
    return reward - cost_distance_all - cost_rsrc_all


# compute the fitness of the whole generation
def fitness(generation):
    n = len(generation)
    fits = np.zeros((n,))
    for i in range(n):
        fits[i] = fitness_code(generation[i])

    return fits


def get_valid_dist(code, utype):
    """Input unmanned platform allocation code to make it meet the requirements
    """
    u_all, u_min = None, None
    if utype == 'uav':
        u_all = uav_num
        u_min = uav_min
    elif utype == 'usv':
        u_all = usv_num
        u_min = usv_min
    elif utype == 'whole':
        uav_code = get_valid_dist(code[0], 'uav')
        usv_code = get_valid_dist(code[1], 'usv')
        return array([uav_code,usv_code])
    else:
        raise ValueError('utype is not under consideration')

    if is_valid_code(code, utype):
        return code

    # satisfy the minimum constraint
    for i in range(task_num):
        if code[i] < u_min:
            code[i] = u_min

    residual = int(u_all - code.sum())
    if residual == 0:
        return code
    elif residual > 0:
        randint = np.floor(get_rand_decimal(residual)*task_num).astype(int)
        for i in randint:
            code[i] += 1
        return code
    else:
        randint = np.floor(get_rand_decimal(-residual)*task_num).astype(int)

        subable = []  # The position that can be subtracted
        for i in range(task_num):
            if code[i] > u_min:
                subable.append(i)

        for i in randint:
            length = len(subable)
            if length == 0:
                break
            idx = i % length
            code[subable[idx]] -= 1
            if code[subable[idx]] == u_min:
                subable.pop(idx)

        if is_valid_code(code, utype):
            return code

    raise ValueError('the value if code should meet all requirements, but it does not'
                     'OR the input he input data is not captured by any branch')


# initial the first generation with  initial solution and the fraction of the solution
def pop_init(init_sol, ratio=0.3):
    global fits
    num_neighbor = int(ratio * pop_num)
    num_rand = pop_num - num_neighbor  # 全解空间的随机解数量

    usv_code = []
    uav_code = []
    pop = []

    # Generate solutions in the neighborhood of the initial solution
    randint = np.random.randint(-3, 4, size=(num_neighbor, task_num))
    cnt = 0
    while cnt < num_neighbor:
        perturbed_solution = init_sol[0] + randint[cnt]
        uav_code.append(get_valid_dist(perturbed_solution, 'uav'))
        cnt += 1

    randint = np.random.randint(-3, 4, size=(num_neighbor, task_num))
    cnt = 0
    while cnt < num_neighbor:
        perturbed_solution = init_sol[1] + randint[cnt]
        usv_code.append(get_valid_dist(perturbed_solution, 'usv'))
        cnt += 1

    for i in range(num_neighbor):
        pop.append(array([uav_code[i], usv_code[i]]))

    # Randomly generate solutions in the global solution space
    usv_code, uav_code = [], []
    top = uav_num - uav_min * task_num
    summ, mini = uav_num, uav_min
    randint = np.random.randint(0, top, size=(num_rand, task_num))
    cnt = 0
    while cnt < num_rand:
        rand_sort = np.sort(randint[cnt])
        code_tmp = []
        for i in range(task_num - 1):
            code_tmp.append(rand_sort[i + 1] - rand_sort[i] + mini)
        code_tmp.append(summ - sum(code_tmp))
        uav_code.append(code_tmp)
        cnt += 1

    # Randomly generate solutions in the global solution space
    top = usv_num - usv_min * task_num
    summ, mini = usv_num, usv_min
    randint = np.random.randint(0, top, size=(num_rand, task_num))
    cnt = 0
    while cnt < num_rand:
        rand_sort = np.sort(randint[cnt])
        code_tmp = []
        for i in range(task_num - 1):
            code_tmp.append(rand_sort[i + 1] - rand_sort[i] + mini)
        code_tmp.append(summ - sum(code_tmp))
        usv_code.append(code_tmp)
        cnt += 1

    for i in range(num_rand):
        pop.append(array([uav_code[i], usv_code[i]]))
    fits = fitness(pop)

    return array(pop)


# Input generation, act as father parent in turn, cross to form the next generation
def cross(pop):
    next_pop = []
    length = len(pop)
    randint = np.floor(get_rand_decimal(length)*pop_num).astype(int)
    for i in range(length):
        mom = pop[randint[i]]
        son = np.floor((mom + pop[i])/2).astype(int)
        son = get_valid_dist(son, 'whole')
        next_pop.append(son)

    return next_pop


def variation(pop):
    length = len(pop)
    rand = get_rand_decimal(length)
    for i in range(length):
        if rand[i] < prob:  # variation
            rand4 = get_rand_decimal(4)
            loc_rand0, loc_rand1 = int(rand4[0]*8), int(rand4[1]*8)
            num_rand0, num_rand1 = int(rand4[2]*6 - 3), int(rand4[3]*6 -3)
            pop[i][0][loc_rand0] += num_rand0
            pop[i][1][loc_rand1] += num_rand1

            pop[i] = get_valid_dist(pop[i], 'whole')

    return pop


def select(pop_parent, pop_son):
    global best_fit, best_code, fits
    length = len(pop_son)
    next_pop = []
    fits_son = fitness(pop_son)
    for i in range(length):
        if fits_son[i] > fits[i]:
            next_pop.append(pop_son[i])
            fits[i] = fits_son[i]
        else:
            next_pop.append(pop_parent[i])

    idx = np.argmax(fits)
    best_fit = fits[idx]
    best_code = next_pop[idx]

    return next_pop
