import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pickle
from scipy.signal import find_peaks_cwt
from scipy.ndimage import gaussian_filter
from base_model import BaseModel


scale = 0.5


class LandmarkLearning(BaseModel):

    def time_steps(self, input_stream):
        
        return input_stream.shape[0] 


    def input(self, input_stream):
        
        time_steps = self.time_steps(input_stream) 
        
        for i in range(time_steps):
            
            input = input_stream[i]
            
            p_rand_prox = (
                (
                    self.sim_params["input_rate"] * input 
                )
                * self.sim_params["dt"]
                * 1e-3
            )

            yield (i, p_rand_prox, None)
    
    def PP_generation(self, i, context_window):

        if context_window[0] <= i < context_window[1]:
            self.dSpikes = (
                self.rng.random(self.num_neurons)
                < self.accelerometer
                * self.neuron_params["r_1"]
                * 1e-3
                * self.sim_params["dt"]
            )
        else:
            self.dSpikes = (
                self.rng.random(self.num_neurons)
                < self.accelerometer
                * self.neuron_params["r_0"]
                * 1e-3
                * self.sim_params["dt"]
            )

    
def full_activity(rep_id, repititions):
    t = repititions[rep_id]

    fig, axes = plt.subplots(10, 8, figsize=(20, 20))
    for i in range(10):
        for j in range(8):
            axes[i, j].set_ylim([0,50])
            axes[i, j].plot(50 * t['f'][j * 10 + i])
            axes[i, j].plot(t['accelerometer'][j * 10 + i])
    

def fr_analysis(traces):
    f_list = traces["f"]
    pat_idx = traces["pat_idx"]
    pat_start_idx = traces["pat_start_idx"]
    width = traces["width"]

    reps = []
    for id, pat_start in zip(pat_idx, pat_start_idx):
        rep = np.mean(
            f_list[:, pat_start : pat_start + width],
            axis=1,
        )
        reps.append(rep)

    fr_mat = np.asarray(reps)
    sorting = np.argsort(np.argmax(fr_mat, axis=0))

    return fr_mat, sorting


if __name__ == "__main__":

    rng = np.random.default_rng(123)
    
    neuron_params = {
        "g_d_prox": 1.0,
        "g_d_dist": 1.0,
        "g_L": 1 / 15.0,
        "tau_m": 15.0,  # in ms
        "tau_syn": 5.0,  # in ms
        "tau_pp": 100.0,  # in ms
        "tau_rate": 200.0,  # in ms
        "tau_plast_short": 500,
        "tau_plast_long": 750,
        "lr": 1e-2,
        "tau_velo": 20.0,
        "alpha": 0.5,
        "beta": 5,
        "pp_refac": 3000,  # ms
        "up_cross_dur": 500,  # ms
        "w_prox_max": 0.2,
        "w_prox_min": 0.0,
        "theta": 1,
        "r_0": 0.1,
        "r_1": 1e1,
        'w_sum': 2,
        'delta_w': 0.005,
    }

    sim_params = {
        "dt": 1,  # in ms
        "mean_rate_prox": 2,  # in Hz
        "mean_rate_dist": 2,  # in Hz
        "stim_dur": 10,  # in ms
        "num_pats": 20,  # number of input patterns
        "frac_input": 0.5,
        "input_rate": 1,
    }

    num_neurons = 80

    num_inputs = 784

    neuron = LandmarkLearning(
        num_neurons, num_inputs, neuron_params, sim_params, rng
    )
    
    add = '_alt'


    input_stream = np.load(f'vis_exp_1/input_stream{add}.npy').astype(np.float32)
    input_stream = input_stream.reshape((input_stream.shape[0], input_stream.shape[1] * input_stream.shape[2]))
    imgs = np.load(f'vis_exp_1/mnist_digit{add}.npy')
    order = [1, 3, 0, 2, 4]

    input_stream /= 255
    input_stream *= 30

    state = {}
    input_params = {
            'input_stream': input_stream
    }

    state_0  = neuron.run(input_params=input_params, state=state, context_args={'context_window': [0, 0]}, learning=False, recording=False)
    state_1, container_train_1 = neuron.run(input_params=input_params, state=state_0, context_args={'context_window': [4000, 4200]}, recording=True)
    state_2, container_train_2 = neuron.run(input_params=input_params, state=state_1, context_args={'context_window': [4000, 4200]}, recording=True)
    state_3, container_test = neuron.run(input_params=input_params, state=state_2, context_args={'context_window': [0, 0]}, recording=True, learning=False)

    container = [container_train_1, container_train_2, container_test]


    fig, ax = plt.subplots(ncols=3, figsize=(9, 3))

    for i in range(3):

        ax[i].plot(50 * container[i]['f'].mean(axis=0), color='indianred')
        ax[i].set_ylim([0,10])
        ax[i].axvspan(0, 500, alpha=0.3, color='yellow')
        ax[i].axvspan(2000, 2500, alpha=0.3, color='yellow')
        ax[i].axvspan(4000, 4500, alpha=0.3, color='yellow')
        ax[i].axvspan(6000, 6500, alpha=0.3, color='yellow')
        ax[i].axvspan(8000, 8500, alpha=0.3, color='yellow')

    plt.show()

    with open(f'vis_exp_1/container_train_1{add}.pkl', 'wb') as f:
        pickle.dump(container_train_1, f)
    
    with open(f'vis_exp_1/container_train_2{add}.pkl', 'wb') as f:
        pickle.dump(container_train_2, f)
    
    with open(f'vis_exp_1/container_test{add}.pkl', 'wb') as f:
        pickle.dump(container_test, f)
