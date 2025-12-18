import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle


class Network:

    def __init__(self, num_exc, num_inh, num_inp, neuron_params, sim_params, rng):

        self.num_exc = num_exc
        self.num_inh = num_inh

        self.num_neurons = self.num_exc + self.num_inh

        self.num_inp = num_inp
        
        self.neuron_params = neuron_params
        self.sim_params = sim_params
        self.aux_params = {
            "stim_width": int(
                self.sim_params["stim_dur"] / self.sim_params["dt"]
            ),
            "pp_refac_width": int(
                self.neuron_params["pp_refac"] / self.sim_params["dt"]
            ),
            "pp_dur_width": int(
                self.neuron_params["pp_dur"] / self.sim_params["dt"]
            ),
        }

        self.rng = rng
        
        self.input_neurons = np.asarray(
            self.rng.random((self.sim_params["num_pats"], self.num_inp))
            <= self.sim_params["frac_input"]
            ,
            dtype=int,
        )
        

    def _firing_rate(self, V, theta=1, omega=0):
        return 1.0 / (
            1.0
            + self.neuron_params["alpha"]
            * np.exp(-self.neuron_params["beta"] * (V - theta + omega))
        )

    def init_state(self, state):
        self.I_syn_prox = state.get("I_syn_prox", np.zeros(self.num_inp))
        self.I_syn_rec = state.get("I_syn_rec", np.zeros(self.num_neurons))

        self.PSP_prox = state.get("PSP_prox", np.zeros(self.num_inp))
        self.PSP_rec = state.get("PSP_rec", np.zeros(self.num_neurons))
        self.PP = state.get("PP", np.zeros(self.num_neurons))  # plateau potential

        self.V_soma = state.get("V_soma", np.zeros(self.num_neurons))
        self.V_prox = state.get("V_prox", np.zeros(self.num_neurons))
        self.V_rec = state.get("V_rec", np.zeros(self.num_neurons))

        self.rate_approx = state.get("rate_approx", np.zeros(self.num_neurons))
        self.rate_envelope = state.get("rate_envelope", np.zeros(self.num_neurons))
        self.pp_refac_counter = state.get(
            "pp_refac_counter", np.zeros(self.num_neurons)
        )
        self.pp_dur_counter = state.get("pp_dur_counter", np.zeros(self.num_neurons))
        self.theta = state.get(
            "theta", self.neuron_params["theta"] * np.ones(self.num_neurons)
        )
        self.omega = state.get("omega", np.zeros(self.num_neurons))
        
        self.ET_1 = state.get("ET_1", np.zeros(self.num_inp))
        self.ET_2 = state.get("ET_2", np.zeros(self.num_inp))
        self.IS_1 = state.get("IS_1", np.zeros(self.num_neurons))
        self.IS_2 = state.get("IS_2", np.zeros(self.num_neurons))

        # Disable input for inhibitory neurons
        self.g_D = np.ones(self.num_neurons) * self.neuron_params["g_d_prox"]
        self.g_D[self.num_exc:] = 0

        # Fetch of alternative initialize proximal feedforward and recurrent weights
        self.w_prox = state.get(
            "w_prox",
            (
                np.abs(
                    self.rng.standard_normal(
                        size=(self.num_neurons, self.num_inp)
                    )
                )
                / self.num_inp
            ),
        )

        try:
            self.w_rec = state["w_rec"]

        except:
            self.w_rec = (
                #np.abs(
                #    self.rng.standard_normal(
                #        (self.num_neurons, self.num_neurons)
                #    )
                #)
                #/ self.num_neurons
                #* np.asarray(
                np.asarray(
                    self.rng.random((self.num_neurons, self.num_neurons))
                    <= self.sim_params["rec_conn_prob"],
                    dtype=float,
                )
            )

            # Only connection E -> I and I ->E
            self.w_rec[self.num_exc:, self.num_exc:] = 0.0
            self.w_rec[: self.num_exc, : self.num_exc] = 0.0
            self.w_rec[self.num_exc:, : self.num_exc] *= 0.1
            self.w_rec[: self.num_exc, self.num_exc:] = 0.5 #1/20

            # Remove recurrent self connections
            self.w_rec[np.diag_indices_from(self.w_rec)] = 0
            self.w_rec *= 0

    def generate_container(self, time_steps):
        container = {
                'f': np.zeros((self.num_neurons, time_steps)),
                'f_pred': np.zeros((self.num_neurons, time_steps)),
                'spks': np.zeros((self.num_neurons, time_steps)),
                'rate_approx': np.zeros((self.num_neurons, time_steps)),
                'rate_envelope': np.zeros((self.num_neurons, time_steps)),
                'accelerometer': np.zeros((self.num_neurons, time_steps)),
                'V_soma': np.zeros((self.num_neurons, time_steps)),
                'V_prox': np.zeros((self.num_neurons, time_steps)),
                'PP': np.zeros((self.num_neurons, time_steps)),
                'theta': np.zeros((self.num_neurons, time_steps)),
                'omega': np.zeros((self.num_neurons, time_steps)),
                'PE_prox_': np.zeros((self.num_neurons, time_steps)),
                'pat_start_idx': [],
                'pat_idx': [],
                'width': self.aux_params['stim_width']
            }

        return container

    def run_track(self, state={}, learning=True, recording=False, context_on=True):
        
        time_steps = self.aux_params['stim_width'] * self.sim_params['num_pats']
        self.init_state(state)

        if recording:
            container = self.generate_container(time_steps)

        # Initially no recurrent spikes
        spikes_rec = np.zeros(self.num_neurons, dtype=bool)

        pat_id = -1


        for i in tqdm(range(time_steps), desc="[running]"):

            # Generate input spikes
            if i % self.aux_params['stim_width'] == 0:
                pat_id += 1
                if recording:
                    container['pat_idx'].append(pat_id)
                    container['pat_start_idx'].append(i)
                

            p_rand_prox = (
                    self.sim_params["mean_rate_prox"]
                    * np.ones_like(self.num_inp)
                    + self.sim_params["input_rate"]
                    * self.input_neurons[pat_id]
                ) * self.sim_params["dt"] * 1e-3
            
            spike_vec_prox = self.rng.random(self.num_inp) < p_rand_prox
            
            self.I_syn_prox = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_syn"]
            ) * self.I_syn_prox
            self.I_syn_prox[spike_vec_prox] += 1.0 / (
                self.neuron_params["tau_m"] * self.neuron_params["tau_syn"]
            )

            self.ET_1 = (
                1.0 - self.sim_params["dt"] / self.neuron_params['tau_plast_short'] 
            ) * self.ET_1
            self.ET_1[spike_vec_prox] += 1.0
            self.ET_2 = (
                1.0 - self.sim_params["dt"] / self.neuron_params['tau_plast_long'] 
            ) * self.ET_2
            self.ET_2[spike_vec_prox] += 1.0

            self.I_syn_rec = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_syn"]
            ) * self.I_syn_rec
            self.I_syn_rec[spikes_rec] += 1.0 / (
                self.neuron_params["tau_m"] * self.neuron_params["tau_syn"]
            )
            
            # Neuron potentials
            self.PSP_prox = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_m"]
            ) * self.PSP_prox + self.I_syn_prox * self.sim_params['dt']
            self.PSP_unit_prox = self.PSP_prox * 25.0

            # Plateau potential
            self.PP = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_pp"]
            ) * self.PP

            # Membrane potential cause by proximal inputs
            self.V_prox = self.w_prox @ self.PSP_unit_prox
            
            # Recurrent conductances
            self.g_rec_exc = (
                self.neuron_params["tau_m"]
                * self.w_rec[:, : self.num_exc]
                @ self.I_syn_rec[: self.num_exc]
            ) 
            self.g_rec_inh = (
                self.neuron_params["tau_m"]
                * self.w_rec[:, self.num_exc:]
                @ self.I_syn_rec[self.num_exc:]
            )
            
            d_V_soma = (
                - self.neuron_params["g_L"] * self.V_soma
                + self.g_D * (self.V_prox - self.V_soma)
                # + (1 - f) * V_rec * (-0.3 - V_soma)
                + self.g_rec_exc * (5 - self.V_soma)
                + self.g_rec_inh * (-0.3 - self.V_soma)
            )
            self.V_soma = self.V_soma + self.sim_params["dt"] * d_V_soma
            self.V_soma[self.V_soma < -0.3] = -0.3
           
           # Adapt pp dependent part of threshold
            self.omega = self.neuron_params["g_d_dist"] * self.PP

            # Somatic firing and dendritic prediction
            self.f = self._firing_rate(self.V_soma, theta=self.theta, omega=self.omega)

            # dSpike
            self.rate_approx = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_rate"]
            ) * self.rate_approx
            self.rate_approx[spikes_rec] += 1
            self.rate_envelope = (
                1.0 - self.sim_params["dt"] / self.neuron_params["tau_envelope"]
            ) * self.rate_envelope + self.rate_approx * self.sim_params["dt"] / self.neuron_params["tau_envelope"]

            self.accelerometer = self.rate_approx - self.rate_envelope
            

            if context_on:
                self.dSpikes = self.rng.random(self.num_neurons) < np.heaviside(
                    self.accelerometer - 1.5, 0
                ) * self.neuron_params['r_1'] * 1e-3 * self.sim_params['dt'] 
            else:
                self.dSpikes = self.rng.random(self.num_neurons) < np.heaviside(
                    self.accelerometer - 1.5, 0
                ) * self.neuron_params['r_0'] * 1e-3 * self.sim_params['dt']

            mask = self.dSpikes * (self.pp_refac_counter == 0)
            self.pp_refac_counter[mask] += (
                self.aux_params["pp_refac_width"]
                + self.aux_params["pp_dur_width"]
            )
            self.pp_dur_counter[mask] += self.aux_params["pp_dur_width"]

            self.PP[mask] += 1 / self.sim_params["dt"] / 10
            self.PP[self.pp_dur_counter == 0] = 0
            #self.PP[self.active_dendrites == False] = 0
            
            self.IS_1 = (
                1.0 - self.sim_params["dt"] / self.neuron_params['tau_plast_short'] 
            ) * self.IS_1
            self.IS_1[mask] += 1.0
            self.IS_2 = (
                1.0 - self.sim_params["dt"] / self.neuron_params['tau_plast_long']
            ) * self.IS_2
            self.IS_2[mask] += 1.0

            if learning:
                self.w_prox[:, spike_vec_prox] += (
                        0.2 * self.IS_1[:, np.newaxis]
                        - 0.2 * self.w_prox[:, spike_vec_prox] / self.neuron_params["w_prox_max"] * self.IS_2[:, np.newaxis])
                
                self.w_prox[mask, :] += (
                        0.2 * self.ET_1[np.newaxis, :]
                        - 0.2 * self.w_prox[mask, :] / self.neuron_params["w_prox_max"] * self.ET_2[np.newaxis, :])

                self.w_prox[self.w_prox > self.neuron_params["w_prox_max"]] = (
                    self.neuron_params["w_prox_max"]
                )
                self.w_prox[self.w_prox < self.neuron_params["w_prox_min"]] = (
                    self.neuron_params["w_prox_min"]
                )

            if i % 1000 == 0:
                self.w_prox /= self.w_prox.sum(axis=1)[:, np.newaxis]
            
            self.pp_refac_counter[self.pp_refac_counter > 0] -= 1
            self.pp_dur_counter[self.pp_dur_counter > 0] -= 1

            spikes_rec = self.rng.random(self.num_neurons) < self.f * 50 * 1e-3

            self.V_soma[spikes_rec] -= 1
            
            if recording:
                container['f'][:, i] = self.f
                container['spks'][spikes_rec, i] += 1
                container['rate_approx'][:, i] = self.rate_approx
                container['rate_envelope'][:, i] = self.rate_envelope
                container['accelerometer'][:, i] = self.accelerometer
                container['V_soma'][:, i] = self.V_soma
                container['V_prox'][:, i] = self.V_prox
                container['PP'][:, i] = self.PP  # same definition as in the original
                container['theta'][:, i] = self.theta  # same definition as in the original
                container['omega'][:, i] = self.omega  # same definition as in the original
        
        state = {
            "I_syn_prox": self.I_syn_prox,
            "PSP_prox": self.PSP_prox,
            "PSP_rec": self.PSP_rec,
            "PP": self.PP,
            "V_soma": self.V_soma,
            "V_prox": self.V_prox,
            "V_rec": self.V_rec,
            "pp_refac_counter": self.pp_refac_counter,
            "pp_dur_counter": self.pp_dur_counter,
            "w_prox": self.w_prox,
            "w_rec": self.w_rec,
            "theta": self.theta,
            "omega": self.omega,
            "IS_1": self.IS_1,
            "IS_2": self.IS_2,
            "ET_1": self.ET_1,
            "ET_2": self.ET_2,
        }
        
        if recording:
            return state, container 

        else:
            return state

if __name__ == "__main__":
    rng = np.random.default_rng()

    neuron_params = {
        "g_d_prox": 1.0,
        "g_d_dist": 1.0,
        "g_L": 1/15.,
        "tau_m": 15.0,  # in ms
        "tau_syn": 5.0,  # in ms
        "tau_pp": 100.0,  # in ms
        "tau_rate": 500.0,  # in ms
        "tau_envelope": 400.0,  # in ms
        "tau_plast_short": 500,
        "tau_plast_long": 750,
        "lr": 1e-2,
        "tau_velo": 20.0,
        "alpha": 0.5,
        "beta": 5,
        "pp_refac": 3000,  # ms
        "pp_dur": 500,  # ms
        "gamma": 0.0,
        "target_rate": 0.04,  # relative to max rate, value between 0, 1
        "w_prox_max": 0.5,
        "w_prox_min": 0.0,
        "theta": 1,
        "r_0": 0.01,
        "r_1": 1e2
    }
    sim_params = {
        "dt": 1,  # in ms
        "mean_rate_prox": 2, # in Hz
        "mean_rate_dist": 2,  # in Hz
        "stim_dur": 300,  # in ms
        "num_pats": 20,  # number of input patterns
        "frac_input": 0.5,
        "input_rate": 50,
        "rec_conn_prob": 0.1,
        # "rec_conn_prob": 0.5,
    }

    num_exc = 1 
    num_inh = 0

    num_neurons = num_exc + num_inh
    num_inputs = 500

    neuron = Network(num_exc, num_inh, num_inputs, neuron_params, sim_params, rng)
    state = {}

    repititions = 1

    representations = []
    f = lambda x, mu, sigma: np.exp(- (x - mu) ** 2 / 2 / sigma ** 2)
        
    len_stim = 100
    min_shift = 5

    steps = num_inputs // min_shift

    inp_space = np.linspace(0, 99, 100)
    inp = f(inp_space, 50, 20) * 1.5



    sim_params.update({'num_pats': steps})

    neuron.sim_params['num_pats'] = steps
    neuron.input_neurons = np.zeros((neuron.sim_params['num_pats'], neuron.num_inp))

    for i in range(steps):

        if i >45 and i < 55 :
        #if True: 

            if len_stim  + min_shift * i < num_inputs:
                neuron.input_neurons[i][min_shift * i:len_stim + min_shift * i] = inp
            else:
                wrapped_idx = (len_stim + min_shift * i) % num_inputs
                neuron.input_neurons[i][min_shift * i:-1] = inp[:num_inputs - min_shift * i - 1]
                neuron.input_neurons[i][:wrapped_idx] = inp[num_inputs - min_shift * i -1:-1]


    state, trace = neuron.run_track(
                learning=False, state=state, recording=True, context_on=True)

    
    with open ('single_trace.pkl', 'wb') as f:
        pickle.dump(trace, f)
