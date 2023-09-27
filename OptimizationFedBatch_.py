import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds, check_grad, NonlinearConstraint
import numpy as np
# Load data
"""data = pd.read_excel(r'Projeto\FedBatchOptimization\data_optimization.xlsx')
treal = data['column1']
Sreal = data['column2']
Nreal = data['column3']
Xreal = data['column4']
Preal = data['column5']"""

# Fed-Batch Parameters
Param = [4.3276103255301335, 0.10215543561231799, 12.346130949956718, 0.04565511507929096, 0.0074744247798137625, 0.0022142903501092513, 1.6957452439712215, 0.008166212885461337, 0.028435528710812962, 6.156650857937292, 2.0566837008694536, 0.9088238699999999, 0.8669780915448081, 8.072715437027492, 0.010453310496367572, 5.260491048, 5.325728554979126, 6.837140753063702, 0.03947867100809781]
Optim=[1.4989542634814161,1.4335418277418217,1.5744495703184183,0.2559519340674219,0.7975221812123433,1.3319125502728477,1.1153538963112686]
#Optim=[1.4989542634814161,	1.4335418277418217,	1.5744495703184183,	0.2559519340674219,	0.7975221812123433,	1.3319125502728477,	1.1153538963112686]

# Parameters
um = 0.54 * Param[0] * 0.6 * 0.6 * 0.5 * Optim[0]  # h^-1
Ksr = 0.15 * Param[1]  # adimensional
Sm = 0.3 * Param[2] * Optim[1]  # adimensional
Csx = 1.7640 * Param[3] * 0.59 * 25 * Optim[2]  # gS/gX
Rcsx = 0.022 * Param[4] * Optim[3]  # gX/gS
Csp = 1.7460 * Param[5]  # gS/gP
k1 = 0.14 * Param[6] * 1.5 * Optim[4]  # gP/gX
k2 = 0.74 * Param[7]  # gP/gX h
Cnx = 0.336 * Param[8]  # gN/gX h
KL = 0.1 * Param[9]  # 1/h
k3 = 93 * Param[10]  # gO2/gX
k4 = 85 * Param[11]  # gO2/gP
O2Leq = 7.6 * 10 ** -3 * Param[12]  # gO2/gX
nk = 1.22 * Param[13] * Optim[5]  # adimensional
Rcnx = 0.16 * Param[14] * Optim[6]  # gX/gS
alfa1 = 0.143 * Param[15]  # mmolCO2/g biomass
alfa2 = 0.0000004 * Param[16]  # mmolCO2/g biomass h
alfa3 = 0.0001 * Param[17]  # mmolCO2/lh
Kox = 0.02 * Param[18]  # gO2/gX

# Parâmetros de alimentação
V1 = [0]
V2 = [0]
Sin = 70    # g/L Fructose Concentration in the feeding
Nin = 37      # g/L Nitrogen concentration in the feeding
O2in = 2.432  # g/L
tsim = 20
t = 0
dt = .1
legend_labels_rel = []
productions_rel = {}
def feed_type_fun(l):
    if l == 0:
        feed_type = 'Step'
    elif l == 1:
        feed_type = 'Constant'
    elif l == 2:
        feed_type = 'Sinusoidal'
    elif l == 3:
        feed_type = 'Exponential'
    elif l == 4:
        feed_type = 'Pulse'
    return feed_type

def step_function(F1, F2, t):
    ts = 2
    td = 10
    u = 0
    if t > ts:
        u = 1
    if t > td:
        u = 0
    F1 = F1*0.1 + .5*u
    F2 = F2*0.1 + .5*u
    return F1, F2

def constant_feed(F1, F2,t):
    F1_ = F1
    F2_ = F2
    return F1_, F2_

def sinusoidal_feed(F1, F2, t):
    F1 = F1*5*np.sin(2 * np.pi * t / 10)
    F2 = F2*5*np.sin(2 * np.pi * t / 10)
    return F1, F2

def exponential_feed(F1, F2, t):
    taxa = 0.2
    vazao_F1 = F1 * np.exp(-taxa * t)
    vazao_F2 = F2 * (1 - np.exp(-taxa * t))
    return vazao_F1, vazao_F2

def pulse_feed(F1, F2, t):
    if F1 < 0:
        F1 = 0
    if F2 < 0:
        F2 = 0
    tc = 7.0         # Center time of the pulse
    tau = 5.0        # Pulse duration
    F1 = np.where((t >= tc - tau / 2) & (t <= tc + tau / 2), F1, 0.1)
    F2 = np.where((t >= tc - tau / 2) & (t <= tc + tau / 2), F2, 0.1)
    return F1, F2

def feed_function(F1, F2, t, l):
    if l == 0:
        F1, F2 = step_function(F1,F2,t)
    elif l == 1:
        F1, F2 = constant_feed(F1,F2,t)
    elif l == 2:
        F1, F2 = sinusoidal_feed(F1,F2,t)
    elif l == 3:
        F1, F2 = exponential_feed(F1,F2,t)
    elif l == 4:
        F1, F2 = pulse_feed(F1,F2,t)

    return F1, F2

def FedBatch(F1,F2,F3,t,l,S, X, P, N, O2L, CO2, Vol, F1_,F2_):
    #print(t)
    F1, F2 = feed_function(F1,F2,t,l)
    u = [um * ((N[-1] / S[-1]) / ((N[-1] / S[-1]) + Ksr)) * (1 - ((N[-1] / S[-1]) / Sm) ** nk) * ((O2L[-1] / (Kox * X[-1] + O2L[-1])))]
    Vol.append(Vol[-1] + (F1 + F2) * dt)
    V1.append(F1)
    V2.append(F2)
    X.append(X[-1] + (u[-1] * X[-1] * dt - ((F1 + F2) / Vol[-1]) * X[-1] * dt))
    S.append(S[-1] - ((Csx * u[-1] * X[-1]) + (Rcsx * X[-1]) + Csp * ((k1 * u[-1] * X[-1]) + (k2 * X[-1]))) * dt + (F1 / Vol[-1]) * Sin * dt - ((F1 + F2) / Vol[-1]) * S[-1] * dt)
    if S[-1] < 0:
        S[-1] = 0.001
    P.append(P[-1] + ((k1 * u[-1] * X[-1]) + (k2 * X[-1])) * dt - ((F1 + F2) / Vol[-1]) * P[-1] * dt)
    N.append(N[-1] - ((Cnx * u[-1] * X[-1]) + (Rcnx * X[-1])) * dt + (F2 / Vol[-1]) * Nin * dt - ((F1 + F2) / Vol[-1]) * N[-1] * dt)
    if N[-1] < 0.15:
        N[-1] = 0.15
    O2L.append(O2L[-1] + ((KL * (O2Leq - O2L[-1])) - (k3 * u[-1] * X[-1]) - ((k4 * k1 * u[-1] * X[-1]) + (k4 * k2 * X[-1]))) * dt + (F3 / Vol[-1]) * O2in * dt - ((F1 + F2) / Vol[-1]) * O2L[-1] * dt)
    if O2L[-1] < 0.002432:
        O2L[-1] = 0.002432
    if O2L[-1] > 2.432:
       O2L[-1] = 2.432
    CO2.append(CO2[-1] + ((alfa1 * u[-1] + alfa2) * X[-1] * dt) + (alfa3 * dt) - ((F1 + F2) / Vol[-1]) * CO2[-1] * dt)  # CO2
    u.append(um * ((N[-1] / S[-1]) / ((N[-1] / S[-1]) + Ksr)) * (1 - ((N[-1] / S[-1]) / Sm) ** nk) * ((O2L[-1] / (Kox * X[-1] + O2L[-1]))))
    if u[-1] < 0:
        u[-1] = 0
    #print('%.2f' % S[-1],'%.2f' % X[-1], '%.2f' % P[-1], '%.2f' % N[-1],'%.2f' % O2L[-1], '%.2f' %CO2[-1],'%.2f' %Vol[-1])
    #print(Vol[-1])
    F1_.append(F1)
    F2_.append(F2)
    return S, X, P, N, O2L, CO2, Vol, F1_, F2_

def Model(F1, F2, l):
    t = 0
    aux = 0
    V = [4]
    X = [0.43]  # active biomass  g/L
    S = [.45]    # fructose g/L
    N = [0.058]   # Nitrogen g/L
    P = [0.14]   # PHA g/L
    #P = [0.136603333333333]   # PHA g/L
    O2L = [2.43]
    CO2 = [0.01]
    F1_ = [0]
    F2_ = [0]
    tempo = [0]
    results = {}
    F3 = 200 # Oxygen feeding  
    #F3 = 200 # l = 3
    # Parâmetros de alimentação
    while aux == 0:
        # Variables atualization
        # Specific growth rate
        S, X, P, N, O2L, CO2, V, F1_, F2_  = FedBatch(F1,F2,F3,t, l, S, X, P, N, O2L, CO2, V,F1_,F2_)
        if l == 0:
            feed_type = 'Step'
        elif l == 1:
            feed_type = 'Constant'
        elif l == 2:
            feed_type = 'Sinusoidal'
        elif l == 3:
            feed_type = 'Exponential'
        elif l == 4:
            feed_type = 'Pulse'

        results[feed_type] = {
            'S': S.copy(),
            'X': X.copy(),
            'P': P.copy(),
            'N': N.copy(),
            'O2L': O2L.copy(),
            'CO2': CO2.copy(),
            'V': V.copy(), 
            'F1':F1_.copy(),
            'F2':F2_.copy(),
        }
        t += dt
        tempo.append(t)
        #print(tempo[-1], V[-1])
        if V[-1] > 10 or V[-1] < 4:
            aux = 1
    return results, tempo

def objective_function(feed_values):
# Simulated fermentation process (simplified model)
    if l == 0:
        feed_type = 'Step'
    elif l == 1:
        feed_type = 'Constant'
    elif l == 2:
        feed_type = 'Sinusoidal'
    elif l == 3:
        feed_type = 'Exponential'
    elif l == 4:
        feed_type = 'Pulse'

    results, tempo =  Model(feed_values[0],feed_values[1],l)
    Func_obj = results[feed_type]['P'][-1]*results[feed_type]['V'][-1]
    thal = 3.0
    lambda1 = 10.0
    lambda2 = 10.0
    lambda3 = 1.0
    g1 = np.log(np.absolute(90.11 - results[feed_type]['S'][-1])+ 0.000001) 
    g2 = np.log(np.absolute(10.11 - results[feed_type]['N'][-1]) + 0.000001)
    g3 = np.log(np.absolute(280 - results[feed_type]['X'][-1]) + 0.000001)
    P1 = lambda1*g1 + np.sqrt((lambda1**2)*(g1**2) + thal**2)
    P2 = lambda2*g2 + np.sqrt((lambda2**2)*(g2**2) + thal**2)
    P3 = lambda3*g3 + np.sqrt((lambda3**2)*(g3**2) + thal**2)
    Func_obj -= np.sum([P1,P2,P3])
    return -Func_obj

def constraint1(feed_values):
    if t ==0:
        results, tempo = Model(feed_values[0],feed_values[1],l)
    if l == 0:
        feed_type = 'Step'
    elif l == 1:
        feed_type = 'Constant'
    elif l == 2:
        feed_type = 'Sinusoidal'
    elif l == 3:
        feed_type = 'Exponential'
    elif l == 4:
        feed_type = 'Pulse'
    return 90.11 - results[feed_type]['S'][-1]  # Constraint: Max concentration of A should be <= 0.5

def constraint2(feed_values):
    if t ==0:
        results, tempo = Model(feed_values[0],feed_values[1],l)
    if l == 0:
        feed_type = 'Step'
    elif l == 1:
        feed_type = 'Constant'
    elif l == 2:
        feed_type = 'Sinusoidal'
    elif l == 3:
        feed_type = 'Exponential'
    elif l == 4:
        feed_type = 'Pulse'
    return results[feed_type]['N'][-1] - 10.11

def constraint3(feed_values):
    if t ==0:
        results, tempo = Model(feed_values[0],feed_values[1],l)
    if l == 0:
        feed_type = 'Step'
    elif l == 1:
        feed_type = 'Constant'
    elif l == 2:
        feed_type = 'Sinusoidal'
    elif l == 3:
        feed_type = 'Exponential'
    elif l == 4:
        feed_type = 'Pulse'
    return results[feed_type]['X'][-1] - 280

def constraint4(feed_values):
    if t == 0:
        results,tempo = Model(feed_values[0],feed_values[1],l)
    if l == 0:
        feed_type = 'Step'
    elif l == 1:
        feed_type = 'Constant'
    elif l == 2:
        feed_type = 'Sinusoidal'
    elif l == 3:
        feed_type = 'Exponential'
    elif l == 4:
        feed_type = 'Pulse'
    return results[feed_type]['V'][-1] - 10

def constraint5(feed_values):
    return - feed_values[0] 

def constraint6(feed_values):
    return - feed_values[1] 

# Constraints (e.g., total concentration of nutrients should not exceed a certain limit)

constraints_probs_dict = ({'type': 'ineq', 'fun': constraint1},
                          {'type': 'ineq', 'fun': constraint2},
                          {'type': 'ineq', 'fun': constraint3},
                          {'type': 'ineq', 'fun': constraint4},
                          {'type': 'ineq', 'fun': constraint5},
                          {'type': 'ineq', 'fun': constraint6})

constr1 = NonlinearConstraint(constraint1, 0, np.inf)
constr2 = NonlinearConstraint(constraint2, 0, np.inf)
constr3 = NonlinearConstraint(constraint3, 0, np.inf)
constr4 = NonlinearConstraint(constraint4, 0, np.inf)
constr5 = NonlinearConstraint(constraint5, 0, np.inf)
constr6 = NonlinearConstraint(constraint6, 0, np.inf)
constraints_probs_list = [constr1,constr2,constr3,constr4, constr5, constr6]

# tipo de alimentação = l ['Step','Constant','Sinusoidal','Exponential', 'Pulse']  
for l in range(5):
    if l == 0:
        initial_guess_1 = np.array([1.69,0.7])  # Initial guess for step feed
        initial_guess_2 = np.array([1.4,0.54])  # Initial guess for step feed
        initial_guess_3 = np.array([1,1.54])  # Initial guess for step feed
    elif l == 1:
        initial_guess_1 = np.array([.1,.2])  # Initial guess for Constant feed
        initial_guess_2 = np.array([.2,.2])  # Initial guess for Constant feed
        initial_guess_3 = np.array([.52,.5])  # Initial guess for Constant feed
    elif l == 2:
        initial_guess_1 = np.array([.6,.5])  # Initial guess for Sinusoidal feed
        initial_guess_2 = np.array([.2,.5])  # Initial guess for Sinusoidal feed
        initial_guess_3 = np.array([.2,.5])  # Initial guess for Sinusoidal feed
    elif l == 3:
        initial_guess_1 = np.array([.5,.3])  # Initial guess for Exponential feed        
        initial_guess_2 = np.array([.45,.5])  # Initial guess for Exponential feed     
        initial_guess_3 = np.array([.2,.5])  # Initial guess for Exponential feed     
    elif l == 4:
        initial_guess_1 = np.array([.2,.1])  # Initial guess for Pulse feed
        initial_guess_2 = np.array([.26,.1])  # Initial guess for Pulse feed
        initial_guess_3 = np.array([.1,.1])  # Initial guess for Pulse feed
    
    bnds = ((0, 5), (0,2))
    feed_type = feed_type_fun(l)  
    # Optimization with SLSQP
    result_1 = minimize(objective_function, initial_guess_1, method= 'Powell', hessp = None, bounds=bnds, constraints=constraints_probs_dict,tol=1e-8,options={'maxiter': 1000})
    optimized_feed_1 = result_1.x
    print(result_1.message)
    PHA_result_1, tempo_1 = Model(optimized_feed_1[0], optimized_feed_1[1],l)
    optimal_production_1 = round(PHA_result_1[feed_type]['P'][-1]*PHA_result_1[feed_type]['V'][-1],1)
    
    result_2 = minimize(objective_function, initial_guess_2, method= 'Nelder-Mead', hessp = None, bounds=bnds, constraints=constraints_probs_dict,tol=1e-8,options={'maxiter': 1000})
    optimized_feed_2 = result_2.x
    print(result_2.message)
    PHA_result_2, tempo_2 = Model(optimized_feed_2[0], optimized_feed_2[1],l)
    optimal_production_2 = round(PHA_result_2[feed_type]['P'][-1]*PHA_result_2[feed_type]['V'][-1],1)
    
    result_3 = minimize(objective_function, initial_guess_3, method= 'trust-constr', hessp = None, bounds=bnds, constraints=constraints_probs_list,tol=1e-8,options={'maxiter': 1000})
    optimized_feed_3 = result_3.x
    print(result_3.message)
    PHA_result_3, tempo_3 = Model(optimized_feed_3[0], optimized_feed_3[1],l)
    optimal_production_3 = round(PHA_result_3[feed_type]['P'][-1]*PHA_result_3[feed_type]['V'][-1],1)
    # Valores de alimentação otimizados pelos modelos
    optimal_feed_1 = result_1.x
    optimal_feed_2 = result_2.x
    optimal_feed_3 = result_3.x
    print(result_1.nfev,result_2.nfev,result_3.nfev)
    print(result_1.nit,result_2.nit,result_3.nit)
    # Métodos de otimização
    methods = ['Powell', 'Nelder-Mead','trust-constr']

    # Rótulos para a legenda
    legend_labels = [f'{methods[0]}\nF1: {optimal_feed_1[0]:.2f}, F2: {optimal_feed_1[1]:.2f}', f'{methods[1]}\nF1: {optimal_feed_2[0]:.2f}, F2: {optimal_feed_2[1]:.2f}',f'{methods[2]}\nF1: {optimal_feed_3[0]:.2f}, F2: {optimal_feed_3[1]:.2f}' ]

    # Valores de produção
    productions = [optimal_production_1, optimal_production_2,optimal_production_3]

    # Criação do gráfico de barras
    fig, ax = plt.subplots(figsize=(4, 12))
    bars = ax.bar(methods, productions, color=['b', 'g','r'])

    # Adiciona os valores de produção acima das barras
    for bar, label in zip(bars, productions):
        height = bar.get_height()
        ax.annotate(
            f'{label:.2f}',  # Formata os valores de produção com duas casas decimais
            xy=(bar.get_x() + bar.get_width() / 2, height),
            ha='center',  # Alinhamento horizontal
        )

    ax.set_xlabel('Método de Otimização')
    ax.set_ylabel('Produção Total')
    ax.set_title('Comparação da Produção Total')

    # Adiciona uma legenda personalizada com os valores de alimentação
    legend = plt.legend(bars, legend_labels, loc ='lower left')
    plt.setp(legend.get_texts(), fontsize='10')
    # Exibe o gráfico de barras
    #plt.show()

    results_1 = PHA_result_1
    results_2 = PHA_result_2
    results_3 = PHA_result_3

    S_1 = results_1[feed_type]['S']
    X_1 = results_1[feed_type]['X']
    P_1 = results_1[feed_type]['P']
    N_1 = results_1[feed_type]['N']
    O2L_1 = results_1[feed_type]['O2L']
    CO2_1 = results_1[feed_type]['CO2']
    vol1 = results_1[feed_type]['V']

    S_2 = results_2[feed_type]['S']
    X_2 = results_2[feed_type]['X']
    P_2 = results_2[feed_type]['P']
    N_2 = results_2[feed_type]['N']
    O2L_2 = results_2[feed_type]['O2L']
    CO2_2 = results_2[feed_type]['CO2']
    vol2 = results_2[feed_type]['V']
    
    S_3 = results_3[feed_type]['S']
    X_3 = results_3[feed_type]['X']
    P_3 = results_3[feed_type]['P']
    N_3 = results_3[feed_type]['N']
    O2L_3 = results_3[feed_type]['O2L']
    CO2_3 = results_3[feed_type]['CO2']
    vol3 = results_3[feed_type]['V']

    # Create subplots for each variable
    plt.figure(figsize=(4, 12))
    plt.subplot(3,1,1)
   
    
    plt.plot(tempo_3, X_3, label='trust-constr', color='r')
    plt.plot(tempo_2, X_2, label='Nelder-Mead', color='g')
    plt.plot(tempo_1, X_1, label='Powell', color='b')
    plt.xlabel('Reactor Volume (L)')
    plt.ylabel('Biomass conc. (g/L)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.title('Biomass Concentration Over Time')
    plt.grid(True)
    #plt.legend()

    # Subplot 2: Substrate Concentration (S)
    plt.subplot(3,1,2)
    plt.plot(tempo_3, S_3, label='trust-constr', color='r')
    plt.plot(tempo_2, S_2, label='Nelder-Mead', color='g')
    plt.plot(tempo_1, S_1, label='Powell', color='b')
    #plt.xlabel('Reaction time (h)')
    plt.ylabel('Substrate conc. (g/L)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.title('Substrate Concentration Over Time')
    plt.grid(True)
    #plt.legend()

    # Subplot 3: Product Concentration (P)
    plt.subplot(3, 1,3)
    plt.plot(tempo_3, P_3, label='trust-constr', color='r')
    plt.plot(tempo_2, P_2, label='Nelder-Mead', color='g')
    plt.plot(tempo_1, P_1, label='Powell', color='b')
    plt.xlabel('Reaction time (h)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('Product conc. (g/L)', fontsize=13)
    #plt.title('Product Concentration Over Time')
    plt.grid(True)
    plt.legend(fontsize=12)

    # Subplot 4: Nitrogen Concentration (N)
    """plt.subplot(3, 2, 4)
    plt.plot(vol1, N_1, label='Powell', color='b')
    plt.plot(vol2, N_2, label='Nelder-Mead', color='g')
    plt.plot(vol3, N_3, label='trust-constr', color='r')
    plt.xlabel('Reactor Volume (L)')
    plt.ylabel('Concentration (g/L)')
    plt.title('Nitrogen Concentration Over Time')
    plt.grid(True)
    plt.legend()

    # Subplot 5: Oxygen Concentration (O2L)
    plt.subplot(3, 2, 5)
    plt.plot(vol1, O2L_1, label='Powell', color='b')
    plt.plot(vol2, O2L_2, label='Nelder-Mead', color='g')
    plt.plot(vol3, O2L_3, label='trust-constr', color='r')
    plt.xlabel('Reactor Volume (L)')
    plt.ylabel('Concentration (g/L)')
    plt.title('Oxygen Concentration Over Time')
    plt.grid(True)
    plt.legend()

    # Subplot 6: Carbon Dioxide Concentration (CO2)
    plt.subplot(3, 2, 6)
    plt.plot(vol1, CO2_1, label='Powell', color='b')
    plt.plot(vol2, CO2_2, label='Nelder-Mead', color='g')
    plt.plot(vol3, CO2_3, label='trust-constr', color='r')
    plt.xlabel('Reactor Volume (L)')
    plt.ylabel('Concentration (mmol/L)')
    plt.title('Carbon Dioxide Concentration Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()"""
    #plt.show()

    F1_1 = results_1[feed_type]['F1']
    F2_1 = results_1[feed_type]['F2']

    F1_2 = results_2[feed_type]['F1']
    F2_2 = results_2[feed_type]['F2']

    F1_3 = results_3[feed_type]['F1']
    F2_3 = results_3[feed_type]['F2']
    # Cria um gráfico para F1 e F2 no tempo
    plt.figure(figsize=(12, 12))
    plt.plot(tempo_1, F1_1, 'b--', label= 'F1 (Powell) ' + feed_type)
    plt.plot(tempo_1, F2_1, 'g--',label='F2 (Powell) '  + feed_type)
    plt.plot(tempo_2, F1_2, 'r.',label='F1 (Nelder-Mead) ' + feed_type)
    plt.plot(tempo_2, F2_2, 'c.',label='F2 (Nelder-Mead) ' + feed_type)
    plt.plot(tempo_3, F1_3, 'm-.',label='F1 (Trust-constr) ' + feed_type)
    plt.plot(tempo_3, F2_3, 'y-.',label='F2 (Trust-constr) ' + feed_type)
    plt.xlabel('Time (h)', fontsize=14)
    plt.ylabel('Flow value (L/h)', fontsize=14)
    plt.title('Feed profile of F1 and F2', fontsize=14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend( fontsize=14)
    plt.grid(True)

    # Exibe o gráfico
    #plt.show()

    
    methods = ['Powell', 'Nelder-Mead', 'trust-constr']
    # Assuming feed_type is defined somewhere before this code snippet
    legend_labels_rel.append(feed_type)  # Custom labels for the legend
    productions_rel[feed_type] = productions

width = 0.2
multiplier = 0
x = np.arange(len(methods))
fig, ax = plt.subplots()

for attribute, measurement in productions_rel.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, fontsize=13)
    multiplier += 1

ax.set_ylabel('Production by feed type (g)', fontsize=13)
ax.set_title('Penguin attributes by species', fontsize=13)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(methods, fontsize=13)  # Use set_xticklabels to set custom labels
ax.set_xlabel('Optimization Methods', fontsize=13)  # Uncomment this line if you want an x-axis label
ax.legend(loc='upper left', ncols=3, fontsize=13)
ax.set_ylim(0, 100)
ax.set_title('Optimized production comparison among the methods', fontsize=13)
plt.show()