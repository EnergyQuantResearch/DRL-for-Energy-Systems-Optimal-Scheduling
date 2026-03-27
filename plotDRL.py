import pickle
from re import escape
import shutil
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
import pandas as pd 
import os
from tools import Arguments
import matplotlib
# Use LaTeX when available, but do not require it for basic plotting.
matplotlib.rc('text', usetex=shutil.which('latex') is not None)
pd.options.display.notebook_repr_html=False  

def plot_evaluation_information(datasource,directory):
    sns.set_theme(style='whitegrid')

    with open(datasource,'rb') as tf:
        test_data=pickle.load(tf)
    # I need to plot unbalance, and reward of each step by bar figures
    plt.rcParams["figure.figsize"] = (16,9)
    fig,axs=plt.subplots(2,2)

    plt.subplots_adjust(wspace=0.7,hspace=0.3)
    plt.autoscale(tight=True)
    # fig.tight_layout()
    #prepare data for evaluation the environment here 
    eval_data=pd.DataFrame(test_data['information'])
    eval_data.columns=['time_step','price','netload','action','real_action','soc','battery','gen1','gen2','gen3','unbalance','operation_cost']

    #plot unbalance in axs[0]   
    axs[0,0].cla()
    axs[0,0].set_ylabel('Unbalance of Generation and Load')
    axs[0,0].bar(eval_data['time_step'],eval_data['unbalance'],label='Exchange with Grid',width=0.4)
    axs[0,0].bar(eval_data['time_step']+0.4,eval_data['netload'],label='Netload',width=0.4)
    axs[0,0].legend(loc='upper left',fontsize=12, frameon=False,labelspacing=0.5)
    # axs[0,0].set_xticks([i for i in range(24)],[i for i in range(1,25)])
    # plot reward in axs[1]
    axs[1,1].cla()
    axs[1,1].set_ylabel('Costs')
    axs[1,1].bar(eval_data['time_step'],eval_data['operation_cost'])

    #plot generation and netload in ax[2]
    axs[1,0].cla()
    axs[1,0].set_ylabel('Outputs of Units and Netload (kWh)')
    # axs[1,0].set_xticks([i for i in range(24)], [i for i in range(1, 25)])
    # x=eval_data['time_step']
    battery_positive=np.array(eval_data['battery'])
    battery_negative=np.array(eval_data['battery'])

    battery_negative=np.minimum(battery_negative,0)#  discharge 
    battery_positive=np.maximum(battery_positive,0)# charge 
    # deal with power exchange within the figure 
    imported_from_grid=np.minimum(np.array(eval_data['unbalance']),0)
    exported_2_grid=np.maximum(np.array(eval_data['unbalance']),0)
    # print(battery_negative)
    # print(battery_positive)
    axs[1,0].bar(eval_data['time_step'],eval_data['gen1'],label='gen1')
    axs[1,0].bar(eval_data['time_step'],eval_data['gen2'],label='gen2',bottom=eval_data['gen1'])
    axs[1,0].bar(eval_data['time_step'],eval_data['gen3'],label='gen3',bottom=eval_data['gen1']+eval_data['gen2'])
    axs[1,0].bar(eval_data['time_step'],-battery_positive,color='blue',hatch='/',label='battery charge')
    axs[1,0].bar(eval_data['time_step'],-battery_negative,label='battery discharge',hatch='/',bottom=eval_data['gen1']+eval_data['gen2']+eval_data['gen3'])
    axs[1,0].bar(eval_data['time_step'],-imported_from_grid,label='imported from grid',bottom=eval_data['gen1']+eval_data['gen2']+eval_data['gen3']-battery_negative)
    axs[1,0].bar(eval_data['time_step'],-exported_2_grid,label='exported to grid',bottom=-battery_positive)
    
    axs[1,0].plot(eval_data['time_step'],eval_data['netload'],drawstyle='steps-mid',label='netload')
    axs[1,0].legend(fontsize=12, frameon=False,labelspacing=0.3,loc=(1.05, 0.5))#loc='upper left',
    #plot energy charge/discharge with price in ax[3].
    
    axs[0,1].cla()
    axs[0,1].set_ylabel('Price')
    axs[0,1].set_xlabel('Time Steps')
    
    axs[0,1].plot(eval_data['time_step'],eval_data['price'],drawstyle='steps-mid',label='Price',color='pink')
    axs[0,1].legend(fontsize=12, frameon=False,labelspacing=0.3,loc=(0.2, 1.05))
    axs[0,1]=axs[0,1].twinx()
    axs[0,1].set_ylabel('SOC')
    # axs[0,1].set_xticks([i for i in range(24)], [i for i in range(1, 25)])
    axs[0,1].plot(eval_data['time_step'],eval_data['soc'],drawstyle='steps-mid',label='SOC',color='grey')
    
    # axs[0,1].legend(fontsize=12, frameon=False,labelspacing=0.3,loc=(0.5, 1.05))
    # plt.show()
    # plt.close()
    fig.savefig(f"{directory}/Evoluation Information.svg", format='svg', dpi=600, bbox_inches='tight')
    print('the evaluation figure have been plot and saved')
def make_dir(directory,feature_change):
    cwd=f'{directory}/DRL_{feature_change}_plots'
    os.makedirs(cwd,exist_ok=True)
    return cwd
class PlotArgs():
    def __init__(self) -> None:
        self.cwd=None
        self.feature_change=None 
        self.plot_on=None
def plot_optimization_result(datasource, directory):  # data source is dataframe
    sns.set_theme(style='whitegrid')
    plt.rcParams["figure.figsize"] = (16,9)
    fig, axs = plt.subplots(2, 2)
    # fig.tight_layout()
    plt.subplots_adjust(wspace=0.7,hspace=0.3)
    plt.autoscale(tight=True)
    # plt.subplots_adjust(wspace=0.7, hspace=0.3)
    T = np.array([i for i in range(24)])
    # plot step cost
    axs[0, 0].cla()
    axs[0, 0].set_ylabel('Costs')
    axs[0, 0].set_xlabel('Time(h)')

    axs[0, 0].bar(T, datasource['step_cost'])
    # axs[0,0].set_xticks([i for i in range(24)],[i for i in range(1,25)])

    # plot soc and price at first
    axs[0, 1].cla()
    axs[0, 1].set_ylabel('Price')
    axs[0, 1].set_xlabel('Time(h)')

    axs[0, 1].plot(T, datasource['price'], drawstyle='steps-mid', label='Price',  color='pink')
    axs[0, 1].legend(fontsize=12, frameon=False, labelspacing=0.3, loc=(0.2, 1.05))
    axs[0, 1] = axs[0, 1].twinx()

    axs[0, 1].set_ylabel('SOC')
    axs[0, 1].plot(T, datasource['soc'], drawstyle='steps-mid',  label='SOC', color='grey')
    # axs[0,1].set_xticks([i for i in range(24)],[i for i in range(1,25)])

    axs[0, 1].legend(fontsize=12, frameon=False, labelspacing=0.3, loc=(0.5, 1.05))
    # plot accumulated generation and consumption here
    axs[1, 0].cla()
    axs[1, 0].set_ylabel('Outputs of DGs and Battery')
    axs[1,0].set_xlabel('Time(h)')
    battery_positive = np.array(datasource['battery_energy_change'])
    battery_negative = np.array(datasource['battery_energy_change'])
    battery_negative = np.minimum(battery_negative, 0)  # discharge
    battery_positive = np.maximum(battery_positive, 0)  # charge
    # deal with power exchange within the figure
    # imported_from_grid = np.maximum(np.array(datasource['grid_import']), 0)
    # exported_2_grid = np.minimum(np.array(datasource['grid_import']), 0)
    imported_from_grid=np.array(datasource['grid_import'])
    exported_2_grid=np.array(datasource['grid_export'])
    axs[1, 0].bar(T, datasource['gen1'], label='gen1')
    axs[1, 0].bar(T, datasource['gen2'], label='gen2', bottom=datasource['gen1'])
    axs[1, 0].bar(T, datasource['gen3'], label='gen3', bottom=datasource['gen2'] + datasource['gen1'])
    axs[1, 0].bar(T, -battery_positive, color='blue', hatch='/', label='battery charge')
    axs[1, 0].bar(T, -battery_negative, hatch='/', label='battery discharge',
                  bottom=datasource['gen3'] + datasource['gen2'] + datasource['gen1'])
    # import as generate
    axs[1, 0].bar(T, imported_from_grid, label='import from grid',
                  bottom=-battery_negative + datasource['gen3'] + datasource['gen2'] + datasource['gen1'])
    # export as load
    axs[1, 0].bar(T, -exported_2_grid, label='export to grid', bottom=-battery_positive)
    axs[1, 0].plot(T, datasource['netload'], label='netload',  drawstyle='steps-mid', alpha=0.7)
    axs[1, 0].legend(fontsize=12, frameon=False, labelspacing=0.3, loc=(1.05, 0.5))
    # axs[1,0].set_xticks([i for i in range(24)],[i for i in range(1,25)])
    # plt.show()
    # plt.close()
    fig.savefig(f"{directory}/optimization_information.svg", format='svg', dpi=600, bbox_inches='tight')
    print('optimization result has been ploted')
def smooth(data, sm=5):
        if sm > 1:
            smooth_data = []
            for n, d in enumerate(data):
                if n - sm + 1 >= 0:
                    y = np.mean(data[n - sm + 1: n])
                else:
                    y = np.mean(data[: n])
                smooth_data.append(y)
        return smooth_data
if __name__=='__main__':
    print('test dir and plot shadow loss with sub figure ')




