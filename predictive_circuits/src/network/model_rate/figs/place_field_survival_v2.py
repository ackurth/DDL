import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from base_model import BaseModel

max_val = 1 
vals = np.arange(0.01, 5 + 0.01, 0.01)
gamma = 2
density = 1 /(0.01 + vals ** gamma)
density /= density.sum()

gamma = 5
density = 1 /(0.015 + vals ** gamma)
density /= density.sum()

gamma = 1.
shift = 0.001
fac = 1
density = 1 /(shift + (fac * vals)) ** gamma
density /= density.sum()

p = [0.9, 0.1]
a = [0.0, 0.2]



class PlaceFieldFormation(BaseModel):
    
    def _firing_rate(self, V, theta=1):
        return 1.0 / (
            1.0
            + self.neuron_params["alpha"]
            * np.exp(-self.neuron_params["beta"] * (V - theta + self.neuron_params['theta_offset']))
        )
    
    def _weigth_norm(self, w_prox):
        return w_prox / (w_prox.sum(axis=1) / self.neuron_params['w_sum'])[:, np.newaxis]
    
    def process_misc_after(self, misc_after):

        if misc_after.get('decay', False):
            PP_mask = self.container['PP_onset'].sum(axis=1) >= 1

            mask_1 = PP_mask * (self.neuron_params['decay_fac'] == 0)
            mask_2 = PP_mask * (self.neuron_params['decay_fac'] > 0)
            self.neuron_params['decay_fac'][~PP_mask] = 0
            self.neuron_params['decay_fac'][mask_1] = 1
            self.neuron_params['decay_fac'][mask_2] = np.exp(-0.1) 
            #self.neuron_params['theta_offset'][~PP_mask] = np.minimum(rng.exponential(scale=0.15, size=int((~PP_mask).sum())), 0.5)
            self.neuron_params['theta_offset'][~PP_mask] = self.rng.choice(a=a, p=p, size=int((~PP_mask).sum())) 

            self.neuron_params['theta_offset'][PP_mask] *= self.neuron_params['decay_fac'][PP_mask]


    def process_misc(self, misc, i):

        if self.recording and misc is not None:
            self.container["pat_idx"].append(misc)
            self.container["pat_start_idx"].append(i)
    
    def create_inputs(self, len_stim=200 , min_shift=1):
    
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

    def run_passive(self, state, timesteps=0, traces=None):
        #PP_mask = traces['PP_onset'].sum(axis=1) >= 1
        #print(PP_mask.sum())

        tau = np.ones(self.num_neurons)
        tau = self.rng.uniform(low=2, high=3, size=self.num_neurons)
        #tau = self.rng.uniform(low=2, high=3, size=self.num_neurons)
        #tau[PP_mask] *= 1.6

        w_prox = state['w_prox'].copy()
        a = w_prox.copy()
        w_prox = (1  - np.exp(- timesteps / tau))[:, np.newaxis] * self.neuron_params['w_sum'][:, np.newaxis] / self.num_inp + np.exp(- timesteps / tau)[:, np.newaxis] * w_prox

        if self.neuron_params['norm']:
            w_prox = self._weigth_norm(w_prox) 


        state['w_prox'] = w_prox


        return state
    
    def PP_generation(self, i, self_gen=False, PPs=None):

        if PPs is None:
            PPs = -1 * np.ones(self.num_neurons)

        timer_mask = (np.zeros(self.num_neurons) < self.cross_low) * (self.cross_low < self.cross_high)
        self.mask_cross_low[timer_mask] = False
        self.mask_cross_high[timer_mask] = False
        

        #delta_t = self.cross_high[timer_mask] - self.cross_low[timer_mask]
        #dSpike_prob[timer_mask] = np.exp(- delta_t / self.neuron_params['tau_acc'][timer_mask])

        #if len(dSpike_prob[timer_mask]):
        #    print(delta_t)
        #    print(self.neuron_params['tau_acc'][timer_mask])
        #    print(dSpike_prob[timer_mask])

        delta_t_mask = (self.cross_high - self.cross_low) < 300
        self.PP_attempt = timer_mask * delta_t_mask * (self.pp_refac_counter == 0)

        dSpike_prob = np.zeros(self.num_neurons)
        dSpike_prob[self.PP_attempt] = self.neuron_params['tau_acc'][self.PP_attempt]

        if self.PP_attempt.sum() > 0:
            self.stuff.append(np.where(self.PP_attempt == True)[0])
        
        
        self.cross_low[timer_mask] = 0
        self.cross_high[timer_mask] = 0

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


    num_neurons = 250

    num_pcs = 100
    
    num_neurons = 600

    num_pcs = 200

    tau_acc = np.ones(num_neurons)

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
        "theta": 1.1,
        "theta_offset": np.zeros(num_neurons), 
        "r_PP": 3,
        'w_sum': 3 * np.ones(num_neurons),
        'delta_w': 0.005,
        'dSpike_thres_high': 0.5,
        'dSpike_thres_low': 0.2,
        'norm': True,
        'tau_acc': tau_acc,
        'decay_fac': np.zeros(tau_acc.shape)
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
    
    num_inputs = 500
    
    rng = np.random.default_rng()

    seeds = [1133, 83245, 13845, 39, 234, 5561]

    for seed_number, seed in enumerate(seeds):

        print(f'#################################')
        print(f'{seed_number + 1} of {len(seeds)}')
        print(f'#################################')
        rng = np.random.default_rng(seed)

        neuron_params["theta_offset"] = rng.choice(a=a, p=p, size=num_neurons)


        neuron = PlaceFieldFormation(
            num_neurons, num_inputs, neuron_params, sim_params, rng
        )
        state = {}

        min_shift = 1

        steps = neuron.create_inputs(min_shift=min_shift)
        nums = 7

        time_steps = neuron.sim_params['stim_dur'] * steps

        num_neurons_set = set([i for i in range(num_neurons)])

        PPs = -1 * np.ones(num_neurons)
        rnd_neurons = rng.choice(num_neurons, num_pcs, replace=False)
        PP_times = rng.choice(time_steps, num_pcs)
        PPs[rnd_neurons] = PP_times

        results = []
        during = []
        after_run = []
        w_pre = None

        neurons_with_PC = []
        neurons_with_random_PC = []
        neurons_with_PC.append(rnd_neurons)
        neurons_with_random_PC.append(rnd_neurons)

        for i in range(nums):
            
            print(f'{i+1} of {nums}')

            if i  == 0:
                w_pre = None
                state, t1 = neuron.run(
                    learning=False, state=state, recording=True
                )
                state, traces = neuron.run(
                    learning=True, state=state, recording=True, context_args={'PPs': PPs}
                )
                state, traces_ = neuron.run(
                    learning=True, state=state, recording=True, context_args={'self_gen': True}, misc_after={'decay': False}
                )
            else:
                state = neuron.run_passive(state, 1, traces_)
                _, traces = neuron.run(
                    learning=False, state=state, recording=True, context_args={}
                    )


                after_run.append(traces)
                after_run[-1]['w_prox'] = state['w_prox'].copy()

                state, traces = neuron.run(
                            learning=True, state=state, recording=True, context_args={'self_gen': True}, misc_after={'decay': True}
                )
                during.append(traces)
                during[-1]['w_prox'] = state['w_prox'].copy()
                print(traces['PP_onset'].sum())
                print(during[-1]['PP_onset'].sum())

                neurons_with_PC_mask = traces['f'].max(axis=1) >= neuron_params['dSpike_thres_high']

                neurons_with_PC.append(np.where(neurons_with_PC_mask == True)[0])
                    
                num_neurons_with_PC = np.minimum(len(neurons_with_PC[-1]), num_pcs)
                print(num_neurons_with_PC)

                elig_mask = np.ones(num_neurons, dtype=bool)
                elig_mask[neurons_with_PC[-1]] = False
                if i > 1:
                    elig_mask[neurons_with_PC[-2]] = False

                PPs = -1 * np.ones(num_neurons)
                rnd_neurons = rng.choice(np.arange(0, num_neurons, dtype=int)[elig_mask], num_pcs - num_neurons_with_PC, replace=False)
                neurons_with_random_PC.append(rnd_neurons)
                PP_times = rng.choice(time_steps, num_pcs - num_neurons_with_PC)
                PPs[rnd_neurons] = PP_times
                
                state, traces = neuron.run(
                    learning=True, state=state, recording=True, context_args={'self_gen': False, 'PPs': PPs}
                )
                
                state, traces_ = neuron.run(
                    learning=True, state=state, recording=True, context_args={'self_gen': True}, misc_after={'decay': False}
                )


            state, traces = neuron.run(
                learning=False, state=state, recording=True, context_args={}
                )
            traces['w_prox'] = state['w_prox'].copy()
                
            results.append(traces)

            if i == 0:
                after_run.append(traces)
                during.append(traces)

        for i in range(nums):


            traces_results = results[i]
            traces_after = after_run[i]
            traces_during = during[i]

            for j, traces in enumerate([traces_results, traces_after, traces_during]):

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

                thres = 10

                fr_max = res.max(axis=0)
                fr_max_mask = fr_max < thres
                silent_neuron_ids = np.where(fr_max_mask == True)[0]
                PC_neuron_ids = np.argsort(np.argmax(res[:, fr_max_mask], axis=0))

                m = res.max(axis=0) >= thres 
                silent = np.where(m == False)[0]
                non_silent = np.where(m == True)[0]

                non_silent_fr = res[:, m]
                pcs = np.argsort(np.argmax(non_silent_fr, axis=0))
                joint_sorting = np.hstack([silent, non_silent[pcs]])

                if j == 0:
                    results[i]['rate_avg'] = res
                    results[i]['sorting'] = joint_sorting
                elif j == 1:
                    after_run[i]['rate_avg'] = res
                    after_run[i]['sorting'] = joint_sorting
                else:
                    during[i]['rate_avg'] = res
                    during[i]['sorting'] = joint_sorting

        
        with open(f'data/survival_results_{seed_number}.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(f'data/survival_after_run_{seed_number}.pkl', 'wb') as f:
            pickle.dump(after_run, f)
        with open(f'data/survival_during_{seed_number}.pkl', 'wb') as f:
            pickle.dump(during, f)
