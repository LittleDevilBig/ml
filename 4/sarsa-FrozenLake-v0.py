import gym
import numpy as np
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':
    env=gym.make('FrozenLake-v0')
    env.render()
    Q=np.zeros([env.observation_space.n,env.action_space.n])
    Alpha=np.arange(0.75, 1, 0.02)
    Gamma=np.arange(0.1, 1, 0.05)
    num_episodes=2000

    train=np.zeros([len(Alpha),len(Gamma)])
    test=np.zeros([len(Alpha),len(Gamma)])

    for i in range(len(Alpha)):
        for j in range(len(Gamma)):
            alpha=Alpha[i]
            gamma=Gamma[j]

            rlist=[]
            for k in range(num_episodes):
                s=env.reset()
                reward=0
                terminate=False
                step=0
                while step<99:
                    step+=1
                    a=np.argmax(Q[s,:]+np.random.randn(1,env.action_space.n)*(1./(i+1)))#贪婪加噪声
                    s1,r,terminate,_=env.step(a)
                    #Q学习
                    #Q[s,a]=Q[s,a]+alpha*(r+gamma*np.max(Q[s1,:])-Q[s,a])
                    #sarsa
                    a1=np.argmax(Q[s1,:]+np.random.randn(1,env.action_space.n)*(1./(i+1)))
                    Q[s,a]=Q[s,a]+alpha*(r+gamma*Q[s1,a1]-Q[s,a])
                    reward+=r
                    s=s1
                    if terminate:
                        break
                rlist.append(reward)
            train[i,j]=(sum(rlist)/num_episodes)
            rlist=[]
            for k in range(num_episodes):
                s=env.reset()
                reward=0
                terminate=False
                step=0
                while step<99:
                    step+=1
                    a=np.argmax(Q[s,:])
                    s1,r,terminate,_=env.step(a)
                    reward+=r
                    s=s1
                    if terminate:
                        break
                rlist.append(reward)
            test[i,j]=(sum(rlist)/num_episodes)
            print("Score over time：" + str(sum(rlist) / num_episodes))
            print("打印Q表：", Q)
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(train, interpolation='nearest', cmap='rainbow',
               extent=[0.75, 1, 0, 1],
               origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.savefig('./sarsa.jpg')
    plt.show()



