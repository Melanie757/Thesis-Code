import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


# First single-cell model

# s=0 ##################################################################
def sc1_0(t, y, params):
    a, m = y
    g, M, b, d = params
    dydt = [1-g*a,
            M+b*m*(a-d/b)]
    return dydt

# s=1 ##################################################################
def sc1_1(t, y, params):
    a, m = y
    a1, a2, b, d = params
    dydt = [a1*a*(1-a)-a2*a*(1-m),
            b*m*(a-d/b)]
    return dydt


# Second single-cell model

# s=0 ##################################################################
def sc2_0(t, y, params):
    a, m, c = y
    g, M, b1, b2, d1, C, g1, g2, d2 = params
    derivs = [1-g*a,
              M+b1*c*(1-m)-b2*a*m-d1*m,
              C+g1*a*m-g2*c*(1-m)-d2*c]
    return derivs

# s=1 ##################################################################
def sc2_1(t, y, params):
    a, m, c = y
    a1, a2, b1, b2, d1, g1, g2, d2 = params
    derivs = [a1*a*(1-a)-a2*a*(1-m),
              b1*c*(1-m)-b2*a*m-d1*m,
              g1*a*m-g2*c*(1-m)-d2*c]
    return derivs


# Derived multi-cell models

# s=0 ##################################################################

# logistic growth
def m_0l(t, x, params):
    y, a, m = x
    r, K, g, M, b, d = params
    derivs = [r*y*(1-y/K),
              y-g*a,
              M*y+b*m*(a-d/b*y)]
    return derivs

# generalised logistic growth
def m_0g(t, x, params):
    y, a, m = x
    r, n, K, g, M, b, d = params
    derivs = [r/n*y*(1-(y/K)**n),
              y-g*a,
              M*y+b*m*(a-d/b*y)]
    return derivs

# s=1 ##################################################################

# first version with missing y in the equation for m
def m_1e(t, x, params):
    y, a, m = x
    r1, K1, Tm, a1, a2, b1, d1 = params
    derivs = [r1*y*(1-y/K1)*(m-Tm*y),
              a1*a*(y-a)-a2*a*(y-m),
              b1*m*(a-d1/b1)]
    return derivs

# logistc growth
def m_1l(t, x, params):
    y, a, m = x
    r1, K1, Tm, a1, a2, b1, d1 = params
    derivs = [r1*y*(1-y/K1)*(m-Tm*y),
              a1*a*(y-a)-a2*a*(y-m),
              b1*m*(a-d1/b1*y)]
    return derivs

# generalised logistic growth
def m_1g(t, x, params):
    y, a, m = x
    r1, n1, K1, Tm, a1, a2, b1, d1 = params
    derivs = [r1/n1*y*(1-(y/K1)**n1)*(m-Tm*y),
              a1*a*(y-a)-a2*a*(y-m),
              b1*m*(a-d1/b1*y)]
    return derivs

# first modification (with -d*m)
def m_1d(t, x, params):
    y, a, m = x
    r1, K1, Tm, a1, a2, b1, d1 = params
    derivs = [r1*y*(1-y/K1)*(m-Tm*y),
              a1*a*(y-a)-a2*a*(y-m),
              b1*(y-m)*a-d1*m]
    return derivs

# first modification, adapted (with Ty)
def m_1d2(t, x, params):
    y, a, m = x
    r1, K1, Tm, a1, a2, b1, d1, Ty = params
    derivs = [r1*y*(1-y/K1)*(m-Tm*y),
              a1*a*(y-a)-a2*a*(y-m),
              b1*(Ty*y-m)*a-d1*m]
    return derivs

# second modification (with y-m)
def m_1y(t, x, params):
    y, a, m = x
    r1, K1, Tm, a1, a2, b1, d1 = params
    derivs = [r1*y*(1-y/K1)*(m-Tm*y),
              a1*a*(y-a)-a2*a*(y-m),
              b1*(y-m)*(a-d1/b1*y)]
    return derivs

# second modification with 'typo' (changed sign to minus)
def m_1m(t, x, params):
    y, a, m = x
    r1, K1, Tm, a1, a2, b1, d1 = params
    derivs = [r1*y*(1-y/K1)*(m-Tm*y),
              a1*a*(y-a)-a2*a*(y-m),
              b1*m*(d1/b1*y-a)]
    return derivs

# third modification (changed 'position' of the variables)
def m_1c(t, x, params):
    y, a, m = x
    r1, K1, Tm, a1, a2, b1, d1 = params
    derivs = [r1*y*(1-y/K1)*(m-Tm*y),
              a1*a*(y-a)-a2*a*(y-m),
              b1*a*(y-d1/b1*m)]
    return derivs


# Further functions with the ode models

# adapted residual for the gen. log. growth to minimize with scipy
def residual_m(params, t, data, x0):
    model = solve_ivp(m_1g, [t[0], t[-1]], x0, t_eval=t, args=(params,), method='LSODA')
    model_data = model.y[0:3]
    
    p = 0 # penalty if solutions 'explode'
    data_t = data # needed if model_data is too short
       
    if model_data.shape != data.shape:
        l = np.min(np.array([len(model_data[0]), len(model_data[1]), len(model_data[2])]))
        data_t = data[:, 0:l]
        p = (data.shape[1]-l)*100
    # different weight for the residuals of each solution curve
    wy = np.sum(100*(model_data[0]-data_t[0])**2)
    wa = np.sum((model_data[1]-data_t[1])**2)
    wm = np.sum((model_data[2]-data_t[2])**2)
    return wy + wa + wm + p

# compute the rss for a certain model
def comp_rss(func, t, x0, params, data):
    x = solve_ivp(func, [t[0], t[-1]], x0, t_eval=t, args=(params,), method='LSODA')
    return np.sum((x.y-data)**2, axis=1)

# Jacobian matrix of m_1l
def jac_s(p, params):
    y, a, m = p
    r1, K1, Tm, a1, a2, b1, d1 = params 
    return np.array([[r1*(3*Tm/K1*y**2-2*(m/K1+Tm)*y+m), 0, r1*y*(1-y/K1)],
                     [(a1-a2)*a, a1*(y-2*a)-a2*(y-m), a2*a],
                     [-d1*m, b1*m, b1*(a-d1/b1*y)]])


# Functions for plotting results

def plot_solutions(sol, data, s=1, save_plot=False, save_name='test.pdf'):
    
    t, y, a, m = data
    
    if s==0: 
        ylim_t = 1.55
        ylim_b = -.05
    elif s==1:
        ylim_t = 5.9
        ylim_b = 0
    
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9,4))

    # first subplot
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('$OD_{600}$')
    ax1.plot(sol.t, sol.y[0], label='y', color=colors[0])
    ax1.scatter(t, y, label='$OD_{600}$', color=colors[0], marker='o')
    ax1.set_ylim(top=ylim_t, bottom=ylim_b)
    ax2 = ax1.twinx()
    ax2.set_ylabel('level of a, m, scaled by y')
    ax2.plot(sol.t, sol.y[1], label='a', color=colors[1])
    ax2.plot(sol.t, sol.y[2], label='m', color=colors[2])
    ax2.scatter(t, a, label='a (data)', color=colors[1], marker='>')
    ax2.scatter(t, m, label='m (data)', color=colors[2], marker='d')
    ax2.set_ylim(top=ylim_t, bottom=ylim_b)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend([h1[0]]+h2[0:2]+[h1[1]]+h2[2:4], [l1[0]]+l2[0:2]+[l1[1]]+l2[2:4], shadow=True)

    #second subplot
    ax3.set_xlabel('Time (h)')
    ax3.set_ylabel('level of a, m (per cell)')
    ax3.plot(sol.t, sol.y[1]/sol.y[0], label='a', color=colors[1])
    ax3.plot(sol.t, sol.y[2]/sol.y[0], label='m', color=colors[2])
    ax3.legend(shadow=True)

    fig.tight_layout()
    if save_plot: plt.savefig(save_name)
    plt.show()
    
def plot_fits(result, data):
    a_fit, m_fit, y_fit = result
    t, y, a, m = data

    plt.plot(t, y_fit, label='y')
    plt.plot(t, a_fit, label='a')
    plt.plot(t, m_fit, label='m')
    plt.scatter(t, y, label='$OD_{600}$')
    plt.scatter(t, a, label='a (data)')
    plt.scatter(t, m, label='m (data)')
    plt.xlabel('t')
    plt.legend(shadow=True)
    plt.show()


# Functions for plotting sensitivities

def sobol_plots(Siy, Sia, Sim, labels, save_plot=False, save_name='test.pdf'):
    x = np.arange(len(labels))
    width = 0.2
    caps = 3
    
    fig = plt.figure(figsize=(9,6))
    gs = gridspec.GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1], sharey = ax1)
    ax3 = fig.add_subplot(gs[1,:])

    ax1.errorbar(x-width, Siy['S1'], yerr=Siy['S1_conf'], capsize=caps, 
                 label='$\dot Y$', linestyle="None", marker='o')
    ax1.errorbar(x, Sia['S1'], yerr=Sia['S1_conf'], capsize=caps,
                 label='$\dot a$', linestyle="None", marker='>')
    ax1.errorbar(x+width, Sim['S1'], yerr=Sim['S1_conf'], capsize=caps,
                 label='$ \dot m$', linestyle="None", marker='d')
    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('S1')
    ax1.set_title('S1 Sensitivities')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2.errorbar(x-width, Siy['ST'], yerr=Siy['ST_conf'], capsize=caps, 
                 label='$\dot Y$', linestyle="None", marker='o')
    ax2.errorbar(x, Sia['ST'], yerr=Sia['ST_conf'], capsize=caps,
                 label='$\dot a$', linestyle="None", marker='>')
    ax2.errorbar(x+width, Sim['ST'], yerr=Sim['ST_conf'], capsize=caps,
                 label='$ \dot m$', linestyle="None", marker='d')
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('ST')
    ax2.set_title('ST Sensitivities')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    ax3.errorbar(x-width, Siy['ST']-Siy['S1'], yerr=Siy['ST_conf']+Siy['S1_conf'],
                capsize=caps, label='$\dot Y$', linestyle="None", marker='o', )
    ax3.errorbar(x, Sia['ST']-Sia['S1'], yerr=Sia['ST_conf']+Sia['S1_conf'],
                capsize=caps, label='$\dot a$', linestyle="None", marker='>', )
    ax3.errorbar(x+width, Sim['ST']-Sim['S1'], yerr=Sim['ST_conf']+Sim['S1_conf'],
                capsize=caps, label='$ \dot m$', linestyle="None", marker='d', )

    ax3.set_xlabel('Parameter')
    ax3.set_ylabel('ST-S1')
    ax3.set_title('ST-S1 Sensitivities')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    
    plt.subplots_adjust(bottom=.8)
    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(h1, l1, loc=8, ncol=3, bbox_to_anchor=(.523, -.001), shadow=True)
    
    fig.tight_layout()
    if save_plot: plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def morris_plot(Siy_m, Sia_m, Sim_m, labels, save_plot=False, save_name='test.pdf'):
    caps = 3
    mu = np.linspace(-15, 15)
    #mu = np.linspace(-1000, 1000)
    sem1 = mu*np.sqrt(1000)/2
    sem2 = -mu*np.sqrt(1000)/2
    
    max_y = np.max(Siy_m['sigma'])+5
    max_x = np.max(Siy_m['mu_star'])+12
    #max_y = np.max(Sia_m['sigma'])+350
    #max_x = np.max(Sia_m['mu_star'])+900
    
    fig, axs = plt.subplots(2, 3, figsize=(12,8), sharex=True, sharey=True)

    axs[0,0].set_title('Results for $\dot Y$')
    axs[0,0].scatter(Siy_m['mu_star'], Siy_m['sigma'], color=colors[0])
    axs[0,0].set_xlim(-max_x, max_x)
    axs[0,0].set_ylim(-.2, max_y)
    
    axs[0,1].set_title('Results for $\dot a$')
    axs[0,1].scatter(Sia_m['mu_star'], Sia_m['sigma'], color=colors[1])

    axs[0,2].set_title('Results for $\dot m$')
    axs[0,2].scatter(Sim_m['mu_star'], Sim_m['sigma'], color=colors[2])

    for i, lab in enumerate(labels):
        axs[0,0].annotate(lab, (Siy_m['mu_star'][i], Siy_m['sigma'][i]),
                          (Siy_m['mu_star'][i]+.6, Siy_m['sigma'][i]+.1))
        axs[0,1].annotate(lab, (Sia_m['mu_star'][i], Sia_m['sigma'][i]),
                          (Sia_m['mu_star'][i]+.6, Sia_m['sigma'][i]+.1))
        axs[0,2].annotate(lab, (Sim_m['mu_star'][i], Sim_m['sigma'][i]),
                          (Sim_m['mu_star'][i]+.6, Sim_m['sigma'][i]+.1))

    axs[1,0].set_title('Results for $\dot Y$')
    axs[1,0].scatter(Siy_m['mu'], Siy_m['sigma'], color=colors[0])
    axs[1,0].plot(mu, sem1, color='black', linestyle='dashed', linewidth=1)
    axs[1,0].plot(mu, sem2, color='black', linestyle='dashed', linewidth=1)

    axs[1,1].set_title('Results for $\dot a$')
    axs[1,1].scatter(Sia_m['mu'], Sia_m['sigma'], color=colors[1])
    axs[1,1].plot(mu, sem1, color='black', linestyle='dashed', linewidth=1)
    axs[1,1].plot(mu, sem2, color='black', linestyle='dashed', linewidth=1)

    axs[1,2].set_title('Results for $\dot m$')
    axs[1,2].scatter(Sim_m['mu'], Sim_m['sigma'], color=colors[2])
    axs[1,2].plot(mu, sem1, color='black', linestyle='dashed', linewidth=1)
    axs[1,2].plot(mu, sem2, color='black', linestyle='dashed', linewidth=1)

    for i, lab in enumerate(labels):
        axs[1,0].annotate(lab, (Siy_m['mu'][i], Siy_m['sigma'][i]),
                          (Siy_m['mu'][i]+.6, Siy_m['sigma'][i]+.1))
        axs[1,1].annotate(lab, (Sia_m['mu'][i], Sia_m['sigma'][i]),
                          (Sia_m['mu'][i]+.6, Sia_m['sigma'][i]+.1))
        axs[1,2].annotate(lab, (Sim_m['mu'][i], Sim_m['sigma'][i]),
                          (Sim_m['mu'][i]+.6, Sim_m['sigma'][i]+.1))    

    plt.setp(axs[0, :], xlabel='$\mu^*$')
    plt.setp(axs[1, :], xlabel='$\mu$')
    plt.setp(axs[:, 0], ylabel='$\sigma$')
    
    fig.tight_layout()
    if save_plot: plt.savefig(save_name)
    plt.show() 