#IMPORTING LIBRARIES WHICH ARE REQUIRED
import random
import matplotlib.pyplot as plt

#INITIALISING NUMBER OF GENERATIONS,POPULATION SIZE,LENGTH OF STRING
random.seed(26799)
obj_func=input('enter the objective function : ')
gen=100
N = 8
L = 50
SOL = [[],[],[],[],[],[],[],[]]
#INITIALISING CROSSOVER AND MUTATION PROBABILITY
pc = float(input('enter crossover probability : '))
pm = float(input('enter mutation probability : '))
#DECLARING LISTS WHICH ARE REQUIRED
avgfit_list=[]
maxfit_list=[]
minfit_list=[]
obj_list=[1]

#RANGE OF VARIABLES
x1min = 0
x2min = 0
x1max = 0.5
x2max = 0.5

#OBJECTIVE FUNCTION
def func(x1, x2):
    global obj_func
    f=eval(obj_func)
    return 1 / (1+ f)
#BINARY TO REAL CONVERTING FUNCTION
def bintoreal(k):
    n = (len(k) / 2) - 1
    x1 = 0
    x2 = 0
    for i in range(int(len(k) / 2)):
        x1 = x1 + k[i] * pow(2, n - i)
        x2 = x2 + k[int(len(k) / 2) + i] * pow(2, n - i)
    x1 = x1min + ((x1max - x1min) * (x1) / (pow(2, L / 2) - 1))
    x2 = x2min + ((x2max - x2min) * (x2) / (pow(2, L / 2) - 1))
    return x1,x2
#BINARY TO FITNESS VALUE EVALUATION FUNCTION
def BTOR(S):
    n = (len(S) / 2) - 1
    x1 = 0
    x2 = 0
    for i in range(int(len(S) / 2)):
        x1 = x1 + S[i] * pow(2, n - i)
        x2 = x2 + S[int(len(S) / 2) + i] * pow(2, n - i)

    y = RV(x1, x2)
    return y

#REAL VALUE WITHIN THE RANGE OF VARIABLES CONVERTING FUNCTION
def RV(a, b):
    x1 = x1min + ((x1max - x1min) * (a) / (pow(2, L / 2) - 1))
    x2 = x2min + ((x2max - x2min) * (b) / (pow(2, L / 2) - 1))
    y = func(x1, x2)
    return y
#INITIAL POPULATION
for i in range(N):
    for j in range(L):

        SOL[i].append(random.randint(0,1))

l=0
avgfit=0
#LOOP FOR CALCULATING AVERAGE,MINIMUM AND MAXIMUM FITNESS VALUE FOR DIFFERENT GENERATIONS
while (l<gen):

    Y = []
    MP=[]
    l = l + 1
    #FITNESS VALUE EVALUATION STAGE
    for i in range(N):
        
        Y.append(BTOR(SOL[i]))


    #CALCULATING AVERAGE,MINIMUM AND MAXIMUM FITNESS VALUE
    avgfit = sum(Y) / N
    avgfit_list.append(avgfit)
    maxfit_list.append(max(Y))
    minfit_list.append(min(Y))
    if (min(obj_list) > (1/max(Y))-1):
        obj_list.append((1/max(Y))-1)
    else:
        obj_list.append(obj_list[-1])
    if (l==1):
        maxfit=max(Y)
        solution = bintoreal(SOL[Y.index(max(Y))])

    #CHECKING FOR OPTIMUM SOLUTION IN THIS GENERATION WITH PREVIOUS GENERATIONS
    if (maxfit < max(Y)):
        solution = bintoreal(SOL[Y.index(max(Y))])
    #MATING POOL STAGE
    for i in range(N):

        rr = random.randint(0, int(sum(Y)))
        P = 0
        j = 0
        while (P < rr):
            P = P + Y[j]
            j = j + 1
        if(j==N):
            MP.append(SOL[j-1])
        else:
            MP.append(SOL[j])

    #SHUFFLING TO MAKE CROSS OVER PAIRS
    random.shuffle(MP)


    i = 0

    #CROSSOVER STAGE
    while (i + 1 < len(MP)):

        c1 = random.randint(0, len(MP) - 2)

        c2 = random.randint(c1 + 1, len(MP) - 1)

        r = random.uniform(0, 1)

        if (r <= pc):
            temp = MP[i][c1:c2 + 1]
            MP[i][c1:c2 + 1] = MP[i + 1][c1:c2 + 1]
            MP[i + 1][c1:c2 + 1] = temp
        i = i + 2

#MUTATION STAGE
    for k in range(N):

        for j in range(L):

            rm = random.uniform(0, 1)
            if (rm <= pm):

                if (MP[k][j]) == 0:
                    MP[k][j] = 1

                elif (MP[k][j]) == 1:
                    MP[k][j] = 0
    #EQUATING MATING POOL TO INITIAL POPUATION, TO PROCEED FURTHER
    SOL=MP.copy()

#PRINTING FINAL OPTIMUM SOLUTION AND PLOTTINF THE GRAPHS

print('Final optimum solution (X1,X2) is : ',solution)

plt.plot([a for a in range(gen)],maxfit_list,'r')
plt.xlabel('Generations')
plt.ylabel('maximum fitness values')
plt.ylim(-1,2)
plt.title('Maximum fitness vs Generations')
plt.show()

plt.plot([a for a in range(gen)],minfit_list)
plt.xlabel('Generations')
plt.ylabel('minimum fitness values')
plt.ylim(0,2)
plt.title('Minimum fitness vs Generations')
plt.show()

plt.plot([a for a in range(gen)],avgfit_list,'m')
plt.xlabel('Generations')
plt.ylabel('average fitness values')
plt.ylim(0,2)
plt.title('Average fitness vs Generations')
plt.show()
