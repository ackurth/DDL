import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from base_model import BaseModel


class Attention(BaseModel):
    
    def generate_container(self, time_steps):
        self.container = {
            "f": np.zeros((self.num_neurons, time_steps)),
            "cross_low": np.zeros((self.num_neurons, time_steps)),
            "cross_high": np.zeros((self.num_neurons, time_steps)),
            "spks": np.zeros((self.num_neurons, time_steps)),
            "V_prox": np.zeros((self.num_neurons, time_steps)),
            "PP_onset": np.zeros((self.num_neurons, time_steps)),
            "behaviour": [],
            "lick": [],
        }

    def process_misc(self, misc, i):

        if self.behaviour:

            behaviour = np.dot(self.w_readout, self.f)
            if behaviour > 1:
                behaviour = 1
            elif behaviour < 0:
                behaviour = 0


            lick = self.rng.random(1) < self.baseline_lick + behaviour 

            if lick:
                reward = misc 

                error = (reward - behaviour)

                delta_w_readout = self.lr_behaviour * error * self.f

                self.w_readout += delta_w_readout

            if self.recording:
                self.container["behaviour"].append(behaviour)
                self.container["lick"].append(lick)
        else:
            pass
    

    def run_passive(self, state, timesteps=0, traces=None):
        PP_mask = traces['PP_onset'].sum(axis=1) >= 1

        tau = np.ones(self.num_neurons)
        tau[PP_mask] *= 1.6

        w_prox = state['w_prox'].copy()
        #w_prox = (1  - np.exp(- timesteps / tau)) * self.neuron_params['w_sum'] / self.num_inp + np.exp(- timesteps / tau) * w_prox
        w_prox = (1  - np.exp(- timesteps / tau))[:, np.newaxis] * self.neuron_params['w_sum'] / self.num_inp + np.exp(- timesteps / tau)[:, np.newaxis] * w_prox
        w_prox += self.rng.normal(0, self.neuron_params['w_sum'] / self.num_inp / 2, size=(self.num_neurons, self.num_inp))
        w_prox[self.w_prox < 0] = 0

        if self.neuron_params['norm']:
            w_prox /= w_prox.sum(axis=1)[:, np.newaxis] / self.neuron_params['w_sum'] 

        state['w_prox'] = w_prox

        return state
    
    def PP_generation(self, i, self_gen=False, PPs=None):

        if PPs is None:
            PPs = -1 * np.ones(self.num_neurons)

        timer_mask = (np.zeros(self.num_neurons) < self.cross_low) * (self.cross_low < self.cross_high)
        self.mask_cross_low[timer_mask] = False
        self.mask_cross_high[timer_mask] = False

        delta_t_mask = (self.cross_high - self.cross_low) < 300
        self.PP_attempt = timer_mask * delta_t_mask * (self.pp_refac_counter == 0)

        dSpike_prob = np.zeros(self.num_neurons)
        dSpike_prob[self.PP_attempt] = self.neuron_params['dSpike_prob']

        self.cross_low[timer_mask] = 0
        self.cross_high[timer_mask] = 0

        random_ind_prob = (PPs == i)

        self.PPs =(
            self.rng.random(self.num_neurons)
            < int(self_gen) * dSpike_prob +  random_ind_prob
            )

    def time_steps(self, input_stream, reward_stream):
        
        return input_stream.shape[0] 

    def input(self, input_stream, reward_stream):
        
        time_steps = self.time_steps(input_stream, reward_stream) 
        
        for i in range(time_steps):
            
            input = input_stream[i]
            reward = reward_stream[i]
            
            p_rand_prox = (
                (
                    self.sim_params["input_rate"] * input 
                    + self.sim_params["mean_rate_prox"]
                )
                * self.sim_params["dt"]
                * 1e-3
            )

            yield (i, p_rand_prox, reward)
    
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


    num_neurons = 200

    num_pcs = 100

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
        "pp_refac": 5000,  # ms
        "up_cross_dur": 0,  # ms
        "w_prox_max": 2.,
        "w_prox_min": 0.0,
        "theta": 1,
        "r_PP": 3,
        'w_sum': 4,
        'delta_w': 0.005,
        'dSpike_thres_high': 0.2,
        'dSpike_thres_low': 0.1,
        'norm': True,
        'dSpike_prob': 0.8
    }
    sim_params = {
        "dt": 1,  # in ms
        "mean_rate_prox": 1,  # in Hz
        "mean_rate_dist": 1,  # in Hz
        "stim_dur": 100,  # in ms
        "num_pats": 20,  # number of input patterns
        "frac_input": 0.5,
        "input_rate": 20,
    }
    
    num_inputs = 961
    
    input_stream = np.load(f'input_generation/input_stream.npy').astype(np.float32)
    input_stream = input_stream.reshape((input_stream.shape[0], input_stream.shape[1] * input_stream.shape[2]))
    reward_1_stream = np.load(f'input_generation/reward_1_stream.npy').astype(int)
    reward_2_stream = np.load(f'input_generation/reward_2_stream.npy').astype(int)

    time_steps = input_stream.shape[0]

    state = {}
    input_params = {
            'input_stream': input_stream,
            'reward_stream': reward_2_stream 
    }

    neuron = Attention(
        num_neurons, num_inputs, neuron_params, sim_params, rng
    )

    neuron.baseline_lick = 10. / 1e3
    neuron.w_readout = np.zeros(num_neurons) 
    neuron.lr_behaviour =  1e-3
    neuron.behaviour = False


    PPs = -1 * np.ones(num_neurons)
    rnd_neurons = np.random.choice(num_neurons, num_pcs, replace=False)
    PP_times = np.random.choice(time_steps, num_pcs)
    PPs[rnd_neurons] = PP_times

    neurons_with_PC = []
    neurons_with_random_PC = []
    neurons_with_PC.append(rnd_neurons)
    neurons_with_random_PC.append(rnd_neurons)

    nums = 15

    day_1 = []
    neurons_with_PC = []
    for i in range(nums):

        tau = 1.

        if i  == 0:
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'PPs': PPs}, input_params=input_params
            )
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'self_gen': True}, input_params=input_params
            )
            
            neuron.behaviour = True
        else:
            state, traces = neuron.run(
                learning=False, state=state, recording=True, context_args={}, input_params=input_params
            )

        day_1.append(traces)

    errs_1 = []

    for i in range(1, nums):
        errs_1.append(((reward_2_stream - day_1[i]['behaviour']) ** 2).mean())
    
    
    day_2 = []
    neurons_with_PC = []
    for i in range(nums):

        if i  == 0:
            neuron.behaviour = False
            neuron.neuron_params['dSpike_prob'] = 0.1
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'self_gen': True}, input_params=input_params
            )
            PP_mask = traces['PP_onset'].sum(axis=1) >= 1
            num_neurons_with_surviving_RF = PP_mask.sum()
            print(num_neurons_with_surviving_RF)

            state["w_prox"][~PP_mask] = rng.normal(neuron_params['w_sum'] / num_inputs,
                                neuron_params['w_sum'] / num_inputs / 2,
                                size=((~PP_mask).sum(),num_inputs)
                                )
            state["w_prox"][state["w_prox"] < 0] = 0
            if neuron_params['norm']:
                state["w_prox"] /= state["w_prox"].sum(axis=1)[:, np.newaxis] / neuron_params['w_sum']



            PPs = -1 * np.ones(num_neurons)
            
            elig_mask = np.ones(num_neurons, dtype=bool)
            elig_mask[PP_mask] = False
            rnd_neurons = np.random.choice(np.arange(0, num_neurons, dtype=int)[elig_mask], num_pcs - num_neurons_with_surviving_RF, replace=False)
            
            PP_times = np.random.choice(time_steps, num_pcs - num_neurons_with_surviving_RF)
            PPs[rnd_neurons] = PP_times
            
            neuron.neuron_params['dSpike_prob'] = 0.8
            
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'PPs': PPs}, input_params=input_params
            )
            state, traces = neuron.run(
                learning=True, state=state, recording=True, context_args={'self_gen': True}, input_params=input_params
            )
            neuron.behaviour = True
        else:
            state, traces = neuron.run(
                learning=False, state=state, recording=True, context_args={}, input_params=input_params
            )

        day_2.append(traces)

    errs_2 = []

    for i in range(1, nums):
        errs_2.append(((reward_2_stream - day_2[i]['behaviour']) ** 2).mean())

    plt.close()
    plt.figure()
    plt.plot(errs_1)
    plt.plot(errs_2)
    plt.show()
    plt.close()

    import IPython
    IPython.embed()
    
    with open('data/attention_random_day_1.pkl', 'wb') as f:
        pickle.dump(day_1, f)
    with open('data/attention_random_day_2.pkl', 'wb') as f:
        pickle.dump(day_2, f)
