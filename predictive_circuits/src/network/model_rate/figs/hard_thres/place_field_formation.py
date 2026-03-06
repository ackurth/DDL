import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from base_model import BaseModel

class PlaceFieldFormation(BaseModel):
    def _weight_norm(self, w_prox):
        return w_prox / (self.w_prox.sum(axis=1)[:, np.newaxis] / self.neuron_params["w_sum"])

    def _tuning_curve(self, location, mu, sigma=15):
        return np.exp(-((location - mu) ** 2) / 2 / sigma**2)


    def create_place_field_centers(self):
        self.place_filed_centers = np.linspace(0, self.sim_params["len_track"], self.num_inp)
        

    def _PP_generation(self, i, self_gen=False, PPs=None):

        if PPs is None:
            PPs = -1 * np.ones(self.num_neurons)

        timer_mask = (np.zeros(self.num_neurons) < self.cross_low) * (self.cross_low < self.cross_high)
        delta_t_mask = (self.cross_high - self.cross_low) < self.neuron_params["time_diff_self_mediated"]

        self.PP_attempt = timer_mask * delta_t_mask

        dSpike_prob = np.zeros(self.num_neurons)
        dSpike_prob[self.PP_attempt] = self.neuron_params["prob_self_mediated"]
        random_ind_prob = (PPs == i)

        self.PPs =(
            self.rng.random(self.num_neurons)
            < int(self_gen) * dSpike_prob +  random_ind_prob
            )
        
        self.cross_low[timer_mask] = 0
        self.cross_high[timer_mask] = 0

    def _time_steps(self):

        time_per_lap = self.sim_params['len_track'] / self.sim_params['velocity'] * 1e3 # in ms
        time_steps = time_per_lap / self.sim_params['dt']

        
        return int(time_steps)

    def _input(self):

        # Generate input stream
        time_steps = self._time_steps() 
        
        for i in range(time_steps):
            location = i * self.sim_params['dt'] * 1e-3 * self.sim_params['velocity']

            p_rand_prox = (
                (
                    self.sim_params["mean_rate_background"]
                    * np.ones_like(self.num_inp)
                    + self.sim_params["input_rate"] * self._tuning_curve(location, self.place_filed_centers)
                )
                * self.sim_params["dt"]
                * 1e-3
            )

            yield (i, p_rand_prox, None)

    
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
        "tau_plast_short": 1000,
        "tau_plast_long": 3000,
        "alpha": 0.5,
        "beta": 5,
        "theta": 1,
        "pp_refac": 1000,  # ms
        "w_prox_max": 0.15,
        "w_prox_min": 0.0,
        'w_sum': 2,
        'delta_w': 0.0025,
        'shift': 300,
        'norm': False, 
        'dSpike_thres_high': 0.35,
        'dSpike_thres_low': 0.2,
        'prob_self_mediated': 0.75,
        'time_diff_self_mediated': 300
    }
    sim_params = {
        "dt": 1,  # in ms
        "mean_rate_background": 0,  # in Hz
        "input_rate": 30,
        "len_track": 300, # in cm
        "velocity": 20, # in cm / s
    }

    max_rate = 50

    num_neurons = 500
    num_inputs = 200

    velocities = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    num_bins = sim_params['len_track'] // 2

    runs = []

    for i, velocity in enumerate(velocities):
        print(f"{i+1} of {len(velocities)}")

        sim_params['velocity'] = velocity

        neuron = PlaceFieldFormation(
            num_neurons, num_inputs, neuron_params, sim_params, rng
        )

        time_steps = neuron._time_steps()
        PPs = np.random.choice(time_steps, num_neurons)

        state = {}
        neuron.create_place_field_centers()

        state, t1 = neuron.run(
            learning=False, state=state, recording=True
        )
        state, traces_during = neuron.run(
            learning=True, state=state, recording=True, context_args={'PPs': PPs}
        )
        state, traces = neuron.run(
            learning=False, state=state, recording=True
        )

        f_list = traces["f"]
        time_window = time_steps // num_bins 
        rate_avg = []

        for step in range(num_bins):
            rate = (
                np.mean(
                    f_list[:, step * time_window : (step + 1) * time_window],
                    axis=1,
                )
                * max_rate
                
            )
            rate_avg.append(rate)

        rate_avg = np.asarray(rate_avg)
        traces["rate_avg"] = rate_avg


        runs.append(traces)
    with open('data/supp_1/width.pkl', 'wb') as f:
        pickle.dump(runs, f)


