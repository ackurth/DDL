import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from base_model import BaseModel

class PlaceFieldFormation(BaseModel):

    def process_misc(self, misc, i):

        if self.recording and misc is not None:
            self.container["pat_idx"].append(misc)
            self.container["pat_start_idx"].append(i)
    
    def create_inputs(self, len_stim=200, min_shift=5):
    
        receptive_field = lambda x, mu, sigma: np.exp(-((x - mu) ** 2) / 2 / sigma**2)

        steps = num_inputs // min_shift

        inp_space = np.linspace(0, len_stim - 1, len_stim)
        inp = receptive_field(inp_space, len_stim // 2, len_stim // 10)

        self.sim_params["num_pats"] = steps

        self.input_neurons = np.zeros(
            (self.sim_params["num_pats"], self.num_inp)
        )

        for i in range(steps):

            if len_stim + min_shift * i < num_inputs:
                self.input_neurons[i][
                    min_shift * i : len_stim + min_shift * i
                ] = inp
            else:
                wrapped_idx = (len_stim + min_shift * i) % num_inputs
                self.input_neurons[i][min_shift * i : -1] = inp[
                    : num_inputs - min_shift * i - 1
                ]
                self.input_neurons[i][:wrapped_idx] = inp[
                    num_inputs - min_shift * i - 1 : -1
                ]

        self.input_neurons = np.roll(self.input_neurons, axis=1, shift=-len_stim // 2)

        return steps

    def run_passive(self, state, timesteps=0):

        w_prox = state['w_prox'].copy()

        for i in range(timesteps):
            tau = 40000
            w_prox = ((1 - self.sim_params['dt'] / tau) * w_prox
                           + self.neuron_params['w_sum'] / self.num_inp / tau
                           + self.rng.normal(0, self.neuron_params['w_sum'] / self.num_inp / 2) / tau)
            w_prox[self.w_prox < 0] = 0

        state['w_prox'] = w_prox

        return state
    
    def PP_generation(self, i, self_gen=False, PPs=None):

        if PPs is None:
            PPs = -1 * np.ones(self.num_neurons)

        timer_mask = self.cross_low < self.cross_high
        self.mask_cross_low[timer_mask] = False
        self.mask_cross_high[timer_mask] = False

        delta_t = self.cross_high[timer_mask] - self.cross_low[timer_mask]
        self.cross_low[timer_mask] = 0
        self.cross_high[timer_mask] = 0

        dSpike_prob = np.zeros(self.num_neurons)
        dSpike_prob[timer_mask] = np.exp(- delta_t / 1000)
        #dSpike_prob[timer_mask] = 0

        #random_ind_prob = (
        #    self.neuron_params['p_dSpike']
        #    * 1e-3
        #    * self.sim_params["dt"]
        #)

        random_ind_prob = (PPs == i)

        self.PPs =(
            self.rng.random(self.num_neurons)
            < int(self_gen) * dSpike_prob +  random_ind_prob
            )

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

    
def full_activity(rep_id, repititions):
    t = repititions[rep_id]

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i in range(10):
        for j in range(10):
            axes[i, j].set_ylim([0,50])
            axes[i, j].plot(50 * t['f'][j * 10 + i])
            PP_loc = t['PP_onset'][j * 10 + i]
            for time in np.where(PP_loc == 1)[0]:
                axes[i, j].axvline(x=time, c='red')

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
        "pp_refac": 30000,  # ms
        "up_cross_dur": 0,  # ms
        "w_prox_max": 1.,
        "w_prox_min": 0.0,
        "theta": 1,
        "r_PP": 3,
        "p_dSpike": 1 / 25,
        'w_sum': 4,
        'delta_w': 0.005,
        'dSpike_thres_high': 0.4,
        'dSpike_thres_low': 0.2,
        'norm': True 
    }
    sim_params = {
        "dt": 1,  # in ms
        "mean_rate_prox": 1,  # in Hz
        "mean_rate_dist": 1,  # in Hz
        "stim_dur": 200,  # in ms
        "num_pats": 20,  # number of input patterns
        "frac_input": 0.5,
        "input_rate": 20,
    }
    
    num_neurons = 100

    num_inputs = 500

    neuron = PlaceFieldFormation(
        num_neurons, num_inputs, neuron_params, sim_params, rng
    )
    state = {}

    steps = neuron.create_inputs()
    nums = 2

    time_steps = neuron.sim_params['stim_dur'] * steps

    PPs = np.random.choice(time_steps, num_neurons)

    repititions = []

    for i in range(nums):

        if i  == 0:
            state, t1 = neuron.run(
                learning=False, state=state, recording=True
            )
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'PPs': PPs}
            )
        else:
            #state = neuron.run_passive(state, 3 * int(1e4))
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'self_gen': True}
            )
        state, traces = neuron.run(
            learning=False, state=state, recording=True
        )


        traces['w_prox'] = state['w_prox'].copy()

        repititions.append(traces)

    results = []
    sorting = []
    for i in range(nums):

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
        
        repititions[i]['rate_avg'] = res
        repititions[i]['sorting'] = m

    rep_id = 0

    for rep_id in range(nums):
        full_activity(rep_id, repititions)

        plt.savefig(f'fig/full_{rep_id}.png')
        plt.close()
        
    vmax = 50

    fig, axes = plt.subplots(nums, nums, figsize=(20, 20))

    for i in range(0, nums):
        for j in range(0, nums):
            if j > i:
                axes[i, j].remove()
            else:
                axes[i, j].imshow(results[i].T[sorting[j]], vmin=0, vmax=vmax)


    plt.savefig("fig/consolitdation.png")
    
    with open('data/consolitdation.pkl', 'wb') as f:
        pickle.dump(repititions, f)
    
    neuron = PlaceFieldFormation(
        num_neurons, num_inputs, neuron_params, sim_params, rng
    )
    state = {}

    steps = neuron.create_inputs()
    nums = 2

    time_steps = neuron.sim_params['stim_dur'] * steps

    PP_1 = np.random.choice(time_steps, num_neurons)
    PP_2 = np.random.choice(time_steps, num_neurons)

    repititions = []

    for i in range(nums):

        if i  == 0:
            state, t1 = neuron.run(
                learning=False, state=state, recording=True
            )
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'PPs': PP_1}
            )
        else:
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'PPs': PP_2}
            )
        state, traces = neuron.run(
            learning=False, state=state, recording=True
        )


        traces['w_prox'] = state['w_prox'].copy()

        repititions.append(traces)

    results = []
    sorting = []
    for i in range(nums):

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

        repititions[i]['rate_avg'] = res
        repititions[i]['sorting'] = m

    rep_id = 0

    for rep_id in range(nums):
        full_activity(rep_id, repititions)

        plt.savefig(f'fig/full_{rep_id}.png')
        plt.close()
        
    vmax = 50

    fig, axes = plt.subplots(nums, nums, figsize=(20, 20))

    for i in range(0, nums):
        for j in range(0, nums):
            if j > i:
                axes[i, j].remove()
            else:
                axes[i, j].imshow(results[i].T[sorting[j]], vmin=0, vmax=vmax)


    plt.savefig("fig/remapping.png")

    with open('data/remapping.pkl', 'wb') as f:
        pickle.dump(repititions, f)
