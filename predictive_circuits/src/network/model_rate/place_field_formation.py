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
    
    def PP_generation(self, i, context):

        timer_mask = self.cross_low < self.cross_high
        self.mask_cross_low[timer_mask] = False
        self.mask_cross_high[timer_mask] = False

        delta_t = self.cross_high[timer_mask] - self.cross_low[timer_mask]
        self.cross_low[timer_mask] = 0
        self.cross_high[timer_mask] = 0

        dSpike_prob = np.zeros(self.num_neurons)
        dSpike_prob[timer_mask] = np.exp(- delta_t / 1000)

        random_ind_prob = (
            self.neuron_params['p_dSpike']
            * 1e-3
            * self.sim_params["dt"]
        )

        self.PPs =(
            self.rng.random(self.num_neurons)
            < dSpike_prob + int(context) * random_ind_prob
            )


    def time_steps(self, input_stream):
        
        return input_stream.shape[0]

    #def input(self):

    #    # Generate input stream
    #    stim_width = self.sim_params['stim_dur'] / self.sim_params['dt']
    #    time_steps = int(stim_width * self.sim_params["num_pats"])
    #    
    #    pat_id = -1
    #    
    #    i = 0
    #    for i in range(time_steps):
    #        
    #        if i % stim_width == 0:
    #            pat_id += 1
    #            pat = pat_id
    #        else:
    #            pat = None
    #        
    #        p_rand_prox = (
    #            (
    #                self.sim_params["mean_rate_prox"]
    #                * np.ones_like(self.num_inp)
    #                + self.sim_params["input_rate"] * self.input_neurons[pat_id]
    #            )
    #            * self.sim_params["dt"]
    #            * 1e-3
    #        )

    #        yield (i, p_rand_prox, pat)
    
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
        "tau_m": 15.0,  # in ms
        "tau_syn": 5.0,  # in ms
        "tau_rate": 200.0,  # in ms
        "tau_plast_short": 500,
        "tau_plast_long": 750,
        "alpha": 0.5,
        "beta": 5,
        "pp_refac": 3000,  # ms
        "up_cross_dur": 500,  # ms
        "w_prox_max": 0.5,
        "w_prox_min": 0.0,
        "theta": 1,
        "r_PP": 1000,
        "p_dSpike": 0.1,
        'w_sum': 2,
        'delta_w': 0.5,
        'dSpike_thres_high': 0.2,
        'dSpike_thres_low': 0.1
    }
    sim_params = {
        "dt": 1,  # in ms
        "mean_rate_prox": 2,  # in Hz
        "mean_rate_dist": 2,  # in Hz
        "stim_dur": 200,  # in ms
        "frac_input": 0.5,
        "input_rate": 50,
    }

    num_neurons = 300
    num_inputs = 1200

    neuron = PlaceFieldFormation(
        num_neurons, num_inputs, neuron_params, sim_params, rng
    )
    
    input_stream = np.load('fig_2/input_stream.npy')
    input_stream = input_stream.reshape((input_stream.shape[0], input_stream.shape[1] * input_stream.shape[2] * input_stream.shape[3]))
    
    input_params = {
            'input_stream': input_stream
    }

    state = {}
            
    state, _ = neuron.run(
            learning=False, state=state, recording=True, context_args={'context': False}, input_params=input_params
        )
    state, _ = neuron.run(
        learning=True, state=state, recording=True, context_args={'context': True}, input_params=input_params
    )
    state, traces = neuron.run(
        learning=False, state=state, recording=True, context_args={'context': False}, input_params=input_params
    )
    traces['w_prox'] = state['w_prox'].copy()
    
    results = []
    sorting = []

    f_list = traces["f"]
    pat_idx = traces["pat_idx"]
    pat_start_idx = traces["pat_start_idx"]
    width = int(sim_params['stim_dur'] / sim_params['dt'])

    dur = 200
    rate_bins = input_stream.shape[0] // dur

    reps = []
    for i in range(rate_bins): 
        rep = (
            np.mean(
                f_list[:, i * dur : (i + 1) *  dur],
                axis=1,
            )
            * 50
        )
        reps.append(rep)

    results = np.asarray(reps)

    traces['results'] = results

    with open('fig_2/place_field_data.pkl', 'wb') as f:
        pickle.dump(traces, f)
