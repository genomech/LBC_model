import numpy as np
import random
from math import exp

BEAD_SIZE_NM = 15.0
NUM_INTEGRATION_STEP = 0.01
EXTRUDER_PRES = 1 # exp(4.67)

def prob_function_polymer(
    thread,
    rnap_n,
    rnap_pos,
    rnap_bristle,
    rnap_occupied,
    rna_bead_params,
    is_topology,
):
    if is_topology == 1.0:
        return 1.0
    else:
        a = rna_bead_params[1]
        pi = rna_bead_params[2]
        d = rna_bead_params[3]
        n_b = rnap_bristle[thread, rnap_n, 1] / rna_bead_params[0]

        if d > n_b * a:
            return 1.0

        l1 = -1.0
        l2 = -1.0
        for i in range(1, min(rnap_occupied.shape[1] - rnap_pos[thread, rnap_n], rnap_pos[thread, rnap_n])):
            if (rnap_occupied[thread, rnap_pos[thread, rnap_n] + i] != 0) and (l1 == -1):
                l1 = i * BEAD_SIZE_NM
            if (rnap_occupied[thread, rnap_pos[thread, rnap_n] - i] != 0) and (l2 == -1):
                l2 = i * BEAD_SIZE_NM
            if (l1 > 0) and (l2 > 0):
                break
    
        prob2 = 2.0
        
        n_b = ((d / a)**5 * 8*pi/9)**(1/3)
        brist_r = (9 * n_b**3 / (8 * pi))**(1/5) * a
        F1 = brist_r**2 / (n_b * a**2) + (n_b**2 * a**3) / ((4/3) * pi * brist_r**3)
        F2 = d**2 / (n_b * a**2) + (4 * n_b**2 * a**3) / (pi * brist_r * d**2)
        A0 = F2 - F1

        n_b = rnap_bristle[thread, rnap_n, 1] / rna_bead_params[0]

        brist_r = (9 * n_b**3 / (8 * pi))**(1/5) * a
        F1 = brist_r**2 / (n_b * a**2) + (n_b**2 * a**3) / ((4/3) * pi * brist_r**3)
        F2 = d**2 / (n_b * a**2) + (4 * n_b**2 * a**3) / (pi * brist_r * d**2)
        
        prob1 = exp(F1 - F2 + A0)
        brist_r1 = brist_r
        
        if (l1 > 0) and (l2 > 0):
            n_b = (d**4 * pi * (l1 + l2) / (2 * a**5))**(1/3)
            brist_r = (2 * n_b**3 * a**5 / (pi * (l1 + l2)))**(1/4)
            F1 = brist_r**2 / (n_b * a**2) + (n_b**2 * a**3) / (pi * (l1+l2)/2 * brist_r**2)
            F2 = d**2 / (n_b * a**2) + (4 * n_b**2 * a**3) / (pi * (l1+l2)/2 * d**2)
            A0 = F2 - F1

            n_b = rnap_bristle[thread, rnap_n, 1] / rna_bead_params[0]

            brist_r = (2 * n_b**3 * a**5 / (pi * (l1 + l2)))**(1/4)
            F1 = brist_r**2 / (n_b * a**2) + (n_b**2 * a**3) / (pi * (l1+l2)/2 * brist_r**2)
            F2 = d**2 / (n_b * a**2) + (4 * n_b**2 * a**3) / (pi * (l1+l2)/2 * d**2)
            
            prob2 = exp(F1 - F2 + A0)
        
        if max([brist_r1, brist_r]) < d:
            return 1.0
        else:
            return min(prob1, prob2, 1.0)


def prob_function_step(
    thread,
    rnap_n,
    rnap_pos,
    rnap_bristle,
    rnap_occupied,
    rna_bead_params,
    is_topology,
):
    if is_topology == 1.0:
        return 1.0
    else:
        a = rna_bead_params[1]
        pi = rna_bead_params[2]
        d = rna_bead_params[3]
        n_b = rnap_bristle[thread, rnap_n, 1] / rna_bead_params[0]

        if d > n_b * a:
            return 1.0

        l1 = -1.0
        l2 = -1.0
        for i in range(1, min(rnap_occupied.shape[1] - rnap_pos[thread, rnap_n], rnap_pos[thread, rnap_n])):
            if (rnap_occupied[thread, rnap_pos[thread, rnap_n] + i] != 0) and (l1 == -1):
                l1 = i * BEAD_SIZE_NM
            if (rnap_occupied[thread, rnap_pos[thread, rnap_n] - i] != 0) and (l2 == -1):
                l2 = i * BEAD_SIZE_NM
            if (l1 > 0) and (l2 > 0):
                break
        
        brist_r = (9 * n_b**3 / (8 * pi))**(1/5) * a
        brist_r1 = brist_r

        if (l1 > 0) and (l2 > 0):
            brist_r = (2 * n_b**3 * a**5 / (pi * (l1 + l2)))**(1/4)

        if max(brist_r1, brist_r) < d:
            return 1.0
        else:
            return 0.0
        # if rnap_bristle[thread, rnap_n, 1] < 1600:
        #     return 1.0
        # else:
        #     return 0.0

def prob_function_passive(
    thread,
    rnap_n,
    rnap_pos,
    rnap_bristle,
    rnap_occupied,
    rna_bead_params,
    is_topology,
):
    if is_topology == 1.0:
        return 1.0
    else:
        a = rna_bead_params[1]
        pi = rna_bead_params[2]
        d = rna_bead_params[3]
        n_b = rnap_bristle[thread, rnap_n, 1] / rna_bead_params[0]

        if d > n_b * a:
            return 1.0

        l1 = -1.0
        l2 = -1.0
        for i in range(1, min(rnap_occupied.shape[1] - rnap_pos[thread, rnap_n], rnap_pos[thread, rnap_n])):
            if (rnap_occupied[thread, rnap_pos[thread, rnap_n] + i] != 0) and (l1 == -1):
                l1 = i * BEAD_SIZE_NM
            if (rnap_occupied[thread, rnap_pos[thread, rnap_n] - i] != 0) and (l2 == -1):
                l2 = i * BEAD_SIZE_NM
            if (l1 > 0) and (l2 > 0):
                break
        
        
        A = 0
        brist_r = a
        while brist_r < n_b * a:
            F1 = brist_r**2 / (n_b * a**2) + (n_b**2 * a**3) / ((4/3) * pi * brist_r**3)
            A += exp(-F1) * NUM_INTEGRATION_STEP
            brist_r += NUM_INTEGRATION_STEP
        A = 1/A
        
        prob = 0
        brist_r = a
        while brist_r < d:
            F1 = brist_r**2 / (n_b * a**2) + (n_b**2 * a**3) / ((4/3) * pi * brist_r**3)
            prob += (A * exp(-F1) * NUM_INTEGRATION_STEP) * EXTRUDER_PRES
            brist_r += NUM_INTEGRATION_STEP

        prob1 = prob
        
        if (l1 > 0) and (l2 > 0):
            A = 0
            brist_r = a
            while brist_r < n_b * a:
                F1 = brist_r**2 / (n_b * a**2) + (n_b**2 * a**3) / (pi * (l1+l2)/2 * brist_r**2)
                A += exp(-F1) * NUM_INTEGRATION_STEP
                brist_r += NUM_INTEGRATION_STEP
            A = 1/A

            prob = 0
            brist_r = a
            while brist_r < d:
                F1 = brist_r**2 / (n_b * a**2) + (n_b**2 * a**3) / (pi * (l1+l2)/2 * brist_r**2)
                prob += (A * exp(-F1) * NUM_INTEGRATION_STEP) * EXTRUDER_PRES
                brist_r += NUM_INTEGRATION_STEP
        
        return min(prob1, prob, 1.0)

def prob_function_one_sided(
    is_topology,
):
    if is_topology == 0:
        return 1.0
    else:
        return 0.0

def prob_function_stable_prob(
    is_topology,
):
    # if is_topology <= 0.0:
    #     return 1.0
    # else:
    #     return is_topology
    return is_topology