import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from base_model import BaseModel
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from remapping import PlaceFieldRemapping

class PlaceFieldFormation(BaseModel):

    def process_misc(self, misc, i):
            if self.recording and misc is not None:
                self.container["pat_idx"].append(misc)
                self.container["pat_start_idx"].append(i)

    def PP_generation(self, i, context):
        if context:
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

    stim_durs = [100, 150, 200, 250, 400, 700, 1000]
    sigma_mean = []
    sigma_std = []
    for stim_dur in stim_durs:
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
            "r_0": 0.1,
            "r_1": 1e1,
            'w_sum': 2,
            'delta_w': 0.05,
        }
        sim_params = {
            "dt": 1,  # in ms
            "mean_rate_prox": 2,  # in Hz
            "mean_rate_dist": 2,  # in Hz
            "stim_dur": stim_dur,  # in ms
            "num_pats": 20,  # number of input patterns
            "frac_input": 0.5,
            "input_rate": 50,
        }

        num_neurons = 80

        num_inputs = 500

        neuron = PlaceFieldFormation(
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
        
        neuron.input_neurons = np.roll(neuron.input_neurons, axis=1, shift=-len_stim // 2)

        repititions = []

        nums = 4

        state, t1 = neuron.run(
            learning=False, state=state, recording=True, context_args={'context': False}
        )
        state, t1 = neuron.run(
            learning=True, state=state, recording=True, context_args={'context': True}
        )
        state, t1 = neuron.run(
            learning=True, state=state, recording=True, context_args={'context': True}
        )
        state, t1 = neuron.run(
            learning=True, state=state, recording=True, context_args={'context': False}
        )
        state, traces = neuron.run(
            learning=False, state=state, recording=True, context_args={'context': False}
        )




        frs = traces['f']

        sigma_local = []

        num_time_points = len(frs[0])
        space_location = np.arange(0, num_time_points, 1) / stim_dur
        n_bins = 500

        locs = np.linspace(0, 1, n_bins)

        f = lambda x, a, b, m, sigma: a * np.exp(-(x - m) **2 / sigma **2) + b
        ps = []

        for fr in frs:

            fr = fr.reshape(n_bins, len(fr) // n_bins).mean(axis=1)
            fr = np.roll(fr, shift=n_bins// 2 - np.argmax(fr))
            if np.max(fr) < 0.4:
                ps.append(None)
                continue

            filtered = gaussian_filter(fr, len(fr) * 0.05)
            m = filtered > 0.25
            if np.sum(m * ~np.roll(m, 1)) > 1:
                ps.append(None)

            p, c = curve_fit(f, locs, fr, p0=[1, 0.1, 0.5 , 0.1], bounds=(0, [1, 0.5, 1, 1]))

            ps.append(p)

            sigma_local.append(p[-1])


        sigma_mean.append(np.mean(sigma_local))
        sigma_std.append(np.std(sigma_local))

        fig, axes = plt.subplots(10, 8, figsize=(20, 20))
        for i in range(10):
            for j in range(8):
                p = ps[j * 10 + i]
                fr = frs[j * 10 + i]
                fr = fr.reshape(n_bins, len(fr) // n_bins).mean(axis=1)
                axes[i, j].plot(
                        locs,
                        np.roll(fr, shift=n_bins// 2 - np.argmax(fr)))
                if p is None:
                    continue
                axes[i,j].set_title(f'{p[-1]}')
                axes[i, j].plot(locs, f(locs, *p))


        plt.savefig(f'velocity_dependence/{stim_dur}.png')

    np.save('velocity_dependence/sigma_mean', sigma_mean)
    np.save('velocity_dependence/sigma_std', sigma_std)

