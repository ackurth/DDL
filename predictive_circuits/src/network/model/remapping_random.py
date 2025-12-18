import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from base_model import BaseModel


class PlaceFieldRemapping(BaseModel):

    def process_misc(self, misc, i):
            if self.recording and misc is not None:
                self.container["pat_idx"].append(misc)
                self.container["pat_start_idx"].append(i)

    def PP_generation(self, i, context):
        if context:
            self.dSpikes = self.rng.random(self.num_neurons) < self.neuron_params['p_1']

        else:
            self.dSpikes = self.rng.random(self.num_neurons) < self.neuron_params['p_0']

    def time_steps(self):
        
        stim_width = self.sim_params['stim_dur'] / self.sim_params['dt']
        return int(stim_width * self.sim_params["num_pats"])

    def input(self):

        # Generate input stream
        stim_width = self.sim_params['stim_dur'] / self.sim_params['dt']
        time_steps = int(stim_width * self.sim_params["num_pats"])
        
        pat_id = -1
        
        i = 0
        for i in range(time_steps):
            
            if i % stim_width == 0:
                pat_id += 1
                pat = pat_id
            else:
                pat = None
            
            p_rand_prox = (
                (
                    self.sim_params["mean_rate_prox"]
                    * np.ones_like(self.num_inp)
                    + self.sim_params["input_rate"] * self.input_neurons[pat_id]
                )
                * self.sim_params["dt"]
                * 1e-3
            )

            yield (i, p_rand_prox, pat)
    
    def generate_container(self, time_steps):
        self.container = {
            "f": np.zeros((self.num_neurons, time_steps)),
            "spks": np.zeros((self.num_neurons, time_steps)),
            "PSP_prox": np.zeros((self.num_inp, time_steps)),
            "accelerometer": np.zeros((self.num_neurons, time_steps)),
            "V_prox": np.zeros((self.num_neurons, time_steps)),
            "pat_start_idx": [],
            "pat_idx": [],
        }
    
    def record(self, i):
        self.container["f"][:, i] = self.f
        self.container["spks"][self.spikes_rec, i] += 1
        self.container["accelerometer"][:, i] = self.accelerometer
        self.container["V_prox"][:, i] = self.V_prox
        self.container["PSP_prox"][:, i] = self.PSP_prox
    
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
    rng = np.random.default_rng()

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
        "p_0": 0.0004,
        "p_1": 0.004,
        'w_sum': 2,
        'delta_w': 0.05,
    }
    sim_params = {
        "dt": 1,  # in ms
        "mean_rate_prox": 2,  # in Hz
        "mean_rate_dist": 2,  # in Hz
        "stim_dur": 200,  # in ms
        "num_pats": 20,  # number of input patterns
        "frac_input": 0.5,
        "input_rate": 50,
    }

    num_neurons = 80

    num_inputs = 500

    neuron = PlaceFieldRemapping(
        num_neurons, num_inputs, neuron_params, sim_params, rng
    )
    state = {}

    repititions = 1

    representations = []
    f = lambda x, mu, sigma: np.exp(-((x - mu) ** 2) / 2 / sigma**2)

    len_stim = 50
    min_shift = 5

    steps = num_inputs // min_shift

    inp_space = np.linspace(0, len_stim - 1, len_stim)
    inp = f(inp_space, len_stim // 2, len_stim // 4) * 1.5

    sim_params.update({"num_pats": steps})

    neuron.sim_params["num_pats"] = steps
    neuron.input_neurons = np.zeros(
        (neuron.sim_params["num_pats"], neuron.num_inp)
    )

    for i in range(steps):

        if len_stim + min_shift * i < num_inputs:
            neuron.input_neurons[i][
                min_shift * i : len_stim + min_shift * i
            ] = inp
        else:
            wrapped_idx = (len_stim + min_shift * i) % num_inputs
            neuron.input_neurons[i][min_shift * i : -1] = inp[
                : num_inputs - min_shift * i - 1
            ]
            neuron.input_neurons[i][:wrapped_idx] = inp[
                num_inputs - min_shift * i - 1 : -1
            ]
    
    #neuron.input_neurons = np.roll(neuron.input_neurons, axis=1, shift=-len_stim // 2)

    vmax = 50

    repititions = []
    for i in range(3):
        if i == 0:
            state, traces = neuron.run(
                learning=False, state=state, recording=True, context_args={'context': False}
            )

        else:

            state, t1 = neuron.run(
                learning=True, state=state, recording=True, context_args={'context': True}
            )
            state, t1 = neuron.run(
                learning=True, state=state, recording=True, context_args={'context': True}
            )
            for i in range(4):
                state, t1 = neuron.run(
                    learning=True, state=state, recording=True, context_args={'context': False}
                )
            state, traces = neuron.run(
                learning=False, state=state, recording=True, context_args={'context': False}
            )

        traces['w_prox'] = state['w_prox'].copy()

        repititions.append(traces)
    
    with open('remapping/traces_random.pkl', 'wb') as f:
        pickle.dump(repititions, f)

    results = []
    sorting = []
    for i in range(3):

        traces = repititions[i]

        f_list = traces["f"]
        pat_idx = traces["pat_idx"]
        pat_start_idx = traces["pat_start_idx"]
        width = int(sim_params['stim_dur'] / sim_params['dt'])

        reps = []
        for id, pat_start in zip(pat_idx, pat_start_idx):
            rep = (
                np.mean(
                    f_list[:, pat_start : pat_start + width],
                    axis=1,
                )
                * 50
            )
            reps.append(rep)

        res = np.asarray(reps)
        results.append(res)

        m = np.argsort(np.argmax(res, axis=0))
        sorting.append(m)


    np.save('remapping/results_random.npy', np.array(results))
    np.save('remapping/sorting_random.npy', np.array(sorting))

    fig, axes = plt.subplots(ncols=4, figsize=(12, 3))

    axes[0].imshow(results[0].T, vmin=0, vmax=vmax, cmap='jet')
    axes[1].imshow(results[1].T[sorting[1]], vmin=0, vmax=vmax, cmap='jet')
    axes[2].imshow(results[2].T[sorting[1]], vmin=0, vmax=vmax, cmap='jet')
    axes[3].imshow(results[2].T[sorting[2]], vmin=0, vmax=vmax, cmap='jet')

    for i in range(4):
        axes[i].set_xlabel('Location')

    axes[0].set_ylabel('Neurons')
    for i in range(1,4):
        axes[i].set_yticks([],[])

    plt.savefig("remapping/repeat_random.png")
