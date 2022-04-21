import random as rd
import json


# feature ideas:
#   if we are scared, run

class ParameterSetter:
    def get_params(self, gen, defensive=True, parent=None):
        if defensive:
            weight_ranges = {
                'numInvaders': [-50000, -10000], 
                'onDefense': [100, 1000], 
                'invaderDistance': [-2000, -1000], 
                'stop': [-150, -50], 
                'reverse': [-150, -10],
                'distanceToFood': [-1, 1],
                'foodLeft': [-1, 1],
                'distanceFromTeam': [10, 100],
                'distanceToOurFood': [-50, -10],
                'ourFoodLeft': [300, 1100]
            }
        else:
            weight_ranges = {
                'numInvaders': [100, 400], 
                'onDefense': [-200, -50], 
                'invaderDistance': [-300, -10], 
                'stop': [-100, -10], 
                'reverse': [-1000, -10],
                'distanceToFood': [-700, -100],
                'foodLeft': [-2000, -300],
                'distanceFromTeam': [0, 200],
                'distanceToOurFood': [-1, 1],
                'ourFoodLeft': [-1, 1],
                'distanceToStart': [-10000, -3000],
                'distanceToAlly': [50, 300],
                'returnReward': [2000, 4000]
            }

        new_weights = {}
        for param, prange in weight_ranges.items():
            val = rd.randint(prange[0], prange[1])
            val = val / float(gen + 1)
            if parent:
                val += float(parent[param] / 2.0)
            new_weights[param] = val

        return new_weights

    def set_params(self, weights, red, defensive=True):
        if defensive and red:
            parameter_file = 'parameters_dr.json'
        elif defensive and not red:
            parameter_file = 'parameters_db.json'
        elif not defensive and red:
            parameter_file = 'parameters_ar.json'
        else:
            parameter_file = 'parameters_ab.json'
        with open(parameter_file, 'w') as parameters:
            parameters.write(json.dumps(weights))

    def read_params(self, file):
        with open(file) as p:
            params = json.load(p)
        return params