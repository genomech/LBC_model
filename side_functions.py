import numpy as np
from chicken_loop_3D import fractal_conf_maker

try:
    import openmm
except Exception:
    import simtk.openmm as openmm

import simtk.unit


class bead_params_updater_from_file(object):
    def __init__(self, gr_size, file_h5_io, sim_i, N_raw, base_bead_nm = 15., a = 0.33*5,
                 # nbCutOffDist = None, Attr_dist_nm = 1.0, min_R_nm = 1.0,
                ):
        self.gr_size = gr_size
        self.rnaps = file_h5_io
        self.N_raw = N_raw
        self.N = int(np.ceil(self.N_raw/2/self.gr_size)*2)
        self.base_bead_nm = base_bead_nm
        self.a = a
        self.sim_i = sim_i
        # self.nbCutOffDist = nbCutOffDist
        # self.Attr_dist_nm = Attr_dist_nm
        # self.min_R_nm = min_R_nm
    
    def _index_calculator(self, pos):
        if pos < self.N_raw/2:
            return pos//self.gr_size  
        else:
            return self.N - (self.N_raw - 1 - pos)//self.gr_size - 1
    
    def _rnap_sep_calculator(self, pos, tmp_positions, brist):
        lam_before = self.N_raw
        lam_after = self.N_raw
        pos_delta = tmp_positions - pos
        mask_before = pos_delta < 0
        mask_after = pos_delta > 0
        if mask_before.sum() > 0:
            lam_before = min(abs(pos_delta[mask_before]).min(), lam_before)
        if mask_after.sum() > 0:
            lam_after = min(abs(pos_delta[mask_after]).min(), lam_after)
        if pos >= self.N_raw/2 and (pos - lam_before) <= self.N_raw/2:
            lam_before = self.N_raw
        if pos <= self.N_raw/2 and (pos + lam_after) >= self.N_raw/2:
            lam_after = self.N_raw
        lam_before = min(lam_before/2 * self.base_bead_nm, self.a/2 * (8/9 * brist**3 / np.pi)**(1/5))
        lam_after = min(lam_after/2 * self.base_bead_nm, self.a/2 * (8/9 * brist**3 / np.pi)**(1/5))
        return lam_before + lam_after
    
    def setup(self, start_block, block_step):
        self.block_step = block_step
        self.curr_block = start_block
        self.non_empty_particles = {}
        n_rnaps_per_particle = {}
        
        curr_brist_N = np.zeros((self.N,))
        curr_lambd = np.ones((self.N,))
        curr_n_rnaps = np.zeros((self.N,))
        
        tmp_positions = self.rnaps['positions'][self.sim_i, start_block, :]
        if (tmp_positions > 0.).sum() > 0.:
            tmp_bristles = self.rnaps['bristles'][self.sim_i, start_block, tmp_positions > 0.]
            tmp_positions = tmp_positions[tmp_positions > 0.]
            for pos, brist in zip(tmp_positions, tmp_bristles):
                index = self._index_calculator(pos)
                assert index >= 0 and index < self.N 
                if index not in self.non_empty_particles.keys():
                    self.non_empty_particles[index] = {'brist_N': 0., 'lambd': 0.}
                    n_rnaps_per_particle[index] = 0

                n_rnaps_per_particle[index] += 1
                n_bristle = brist*0.33/(2*self.a)
                self.non_empty_particles[index]['brist_N'] += n_bristle
                self.non_empty_particles[index]['lambd'] += self._rnap_sep_calculator(pos, tmp_positions, n_bristle)

            for index in self.non_empty_particles.keys():
                self.non_empty_particles[index]['brist_N'] /= n_rnaps_per_particle[index]
                self.non_empty_particles[index]['lambd'] /= n_rnaps_per_particle[index]
                curr_brist_N[index] = self.non_empty_particles[index]['brist_N']
                curr_lambd[index] = self.non_empty_particles[index]['lambd']
                curr_n_rnaps[index] = n_rnaps_per_particle[index]
        
        return curr_brist_N, curr_lambd, curr_n_rnaps
    
    def step(self, sim_obj, force_name):
        prepu = sim_obj.force_dict[force_name] 
        self.curr_block += self.block_step
        particles_to_update = self.non_empty_particles
        n_rnaps_per_particle = {}
        for k in particles_to_update.keys():
            n_rnaps_per_particle[k] = 0
            particles_to_update[k] = {'brist_N': 0., 'lambd': 1.}
        
        tmp_positions = self.rnaps['positions'][self.sim_i, self.curr_block, :]
        
        if (tmp_positions > 0.).sum() > 0.:
            tmp_bristles = self.rnaps['bristles'][self.sim_i, self.curr_block, tmp_positions > 0.]
            tmp_positions = tmp_positions[tmp_positions > 0.]
            for pos, brist in zip(tmp_positions, tmp_bristles):
                index = self._index_calculator(pos)
                assert index >= 0 and index < self.N 
                if index not in particles_to_update.keys():
                    particles_to_update[index] = {'brist_N': 0., 'lambd': 1.}
                    n_rnaps_per_particle[index] = 0
                
                n_bristle = brist*0.33/(2*self.a)
                n_rnaps_per_particle[index] += 1
                particles_to_update[index]['brist_N'] += n_bristle
                
                particles_to_update[index]['lambd'] += self._rnap_sep_calculator(pos, tmp_positions, n_bristle)
            
        # brist_len_beads=np.zeros((self.N,))
        # brist_sep_nm=np.zeros((self.N,))
        for index in particles_to_update.keys():
            if n_rnaps_per_particle[index] > 0:
                particles_to_update[index]['brist_N'] /= n_rnaps_per_particle[index]
                particles_to_update[index]['lambd'] /= n_rnaps_per_particle[index]
                # brist_len_beads[index] = particles_to_update[index]['brist_N']
                # brist_sep_nm[index] = particles_to_update[index]['lambd']
            prepu.setParticleParameters(index, (particles_to_update[index]['brist_N']*1., particles_to_update[index]['lambd']*sim_obj.conlen, n_rnaps_per_particle[index]))
            
        self.non_empty_particles = {}
        for index in particles_to_update.keys():    
            if n_rnaps_per_particle[index] != 0:
                self.non_empty_particles[index] = particles_to_update[index]
        
        # attr_dist = self.Attr_dist_nm * sim_obj.conlen
        # if sum(brist_len_beads)>0:
        #     radiusMult = max((self.a**(5/4) * np.array(brist_len_beads)**(3/4) * (np.pi * np.array(brist_sep_nm))**(-1/4)).max(), self.min_R_nm)
        # else:
        #     radiusMult = self.min_R_nm
        # nbCutOffDist = 2*(sim_obj.conlen*((self.nbCutOffDist or radiusMult)) + attr_dist)
        # prepu.setCutoffDistance(nbCutOffDist)
        
        prepu.updateParametersInContext(sim_obj.context)
        
        
#         if sum(brist_len_beads)>0:
#             return brist_len_beads, brist_sep_nm
#         else:
#             return np.zeros((self.N,),dtype=float,), np.ones((self.N,),dtype=float,)

def bond_calculator(bead_sizes, beads_in_one, bp_nuc = 225., bp_tu = 45., df_nuc = 3., df_tu = 1., base_bead_nm = 15.0, two_chains = True):
    # Для линиаризации Df
    Df = [df_nuc, df_tu]
    beads_bp = [bp_nuc, bp_tu]
    k_df = (Df[0] - Df[1]) / (beads_bp[0] - beads_bp[1])
    b_df = Df[0] - k_df * beads_bp[0]

    # Узнаём число нуклеотидов в каждой бусине
    if two_chains:
        half_bead_sizes = bead_sizes[:len(bead_sizes)//2]
        tmp_bead_sizes = [sum(half_bead_sizes[i:i+beads_in_one]) for i in range(0, len(half_bead_sizes), beads_in_one)]
        
        half_bead_sizes = bead_sizes[len(bead_sizes)//2:][::-1]
        tmp_bead_sizes += [sum(half_bead_sizes[i:i+beads_in_one]) for i in range(0, len(half_bead_sizes), beads_in_one)][::-1]
    else:
        tmp_bead_sizes = [sum(bead_sizes[i:i+beads_in_one]) for i in range(0, len(bead_sizes), beads_in_one)]
    diams = []
    # Теперь получаем диаметр объединённой бусины, опираясь на среднюю фрактальную размерность входящих в её состав бусин
    for b_idx, b_size in enumerate(tmp_bead_sizes):
        dfractal = (b_size/beads_in_one) * k_df + b_df
        diams.append(base_bead_nm * beads_in_one**(1/dfractal))
    diams = np.array(diams)
    # Полученные размеры надо применить к связи между бусинами, для чего используем среднее арифмитическое между соседями
    halves = diams[:len(diams)//2], diams[len(diams)//2:]
    bonds = np.concatenate( ((halves[0][1:]+halves[0][:-1])/2, (halves[1][1:]+halves[1][:-1])/2) )
    return bonds

def bond_robustness_calculator(bead_sizes, beads_in_one, max_robustness = 2e-1, bp_nuc = 225., bp_tu = 45., df_nuc = 3., df_tu = 1., two_chains = True):
    # Для линиаризации Df
    Df = [df_nuc, df_tu]
    beads_bp = [bp_nuc, bp_tu]
    k_df = (Df[0] - Df[1]) / (beads_bp[0] - beads_bp[1])
    b_df = Df[0] - k_df * beads_bp[0]

    # Узнаём число нуклеотидов в каждой бусине
    if two_chains:
        half_bead_sizes = bead_sizes[:len(bead_sizes)//2]
        tmp_bead_sizes = [sum(half_bead_sizes[i:i+beads_in_one]) for i in range(0, len(half_bead_sizes), beads_in_one)]
        
        half_bead_sizes = bead_sizes[len(bead_sizes)//2:][::-1]
        tmp_bead_sizes += [sum(half_bead_sizes[i:i+beads_in_one]) for i in range(0, len(half_bead_sizes), beads_in_one)][::-1]
    else:
        tmp_bead_sizes = [sum(bead_sizes[i:i+beads_in_one]) for i in range(0, len(bead_sizes), beads_in_one)]
    bonds_rob = []
    # Теперь получаем жёсткость связи объединённой "бусины", опираясь на среднюю фрактальную размерность входящих в её состав бусин
    for b_idx, b_size in enumerate(tmp_bead_sizes):
        dfractal = (b_size/beads_in_one) * k_df + b_df
        bonds_rob.append(max_robustness*beads_in_one**(-1/dfractal+1))
    bonds_rob = np.array(bonds_rob)
    # Полученные жёсткости надо применить к связи между бусинами, для чего используем среднее арифмитическое между соседями
    halves = bonds_rob[:len(bonds_rob)//2], bonds_rob[len(bonds_rob)//2:]
    bonds_rob = np.concatenate( ((halves[0][1:]+halves[0][:-1])/2, (halves[1][1:]+halves[1][:-1])/2) )
    return bonds_rob

def ps_smoother(s_arr, P_s, smth_sigma = 1., sigma_adder = 0.):
    tmp_x = s_arr
    tmp_y = P_s
    x_draw = []
    y_draw = [] 
    min_log_y = np.nan_to_num(np.log(tmp_y), neginf=0.0).min()

    t = tmp_x[0]
    tmp_mask = (np.log(tmp_x) >= np.log(t)) & (np.log(tmp_x) < np.log(t) + smth_sigma)
    while True:
        x_draw += [np.exp(np.nanmean(np.log(tmp_x[tmp_mask])))]
        y_draw += [np.exp(np.mean(np.nan_to_num(np.log(tmp_y[tmp_mask]), neginf=min_log_y)))]
        smth_sigma += sigma_adder
        if sum(tmp_mask) > 0:
            if np.arange(len(tmp_x))[tmp_mask][-1]+1 < len(tmp_x):
                t = tmp_x[np.arange(len(tmp_x))[tmp_mask][-1]+1]
                tmp_mask = (np.log(tmp_x) >= np.log(t)) & (np.log(tmp_x) < np.log(t) + smth_sigma)
            else:
                break
        else:
            break

    x_draw = np.array(x_draw)
    y_draw = np.array(y_draw)
    return x_draw, y_draw

def lef_pos_calculator(pos, N_raw, gr_size):
    N = int(np.ceil(N_raw/2/gr_size)*2)
    if pos < N_raw/2:
        return pos//gr_size  
    else:
        return N - (N_raw - 1 - pos)//gr_size - 1


def marko_excluded_volume_with_attraction(sim_object, 
                                    brist_len_beads,
                                    brist_sep_nm,
                                    n_rnaps,
                                    brist_bead_size_nm = 1.0,
                                    Erep = 1.0,
                                    Eattr = 0.0,
                                    min_R_nm = 1.0,
                                    dt_fs = 1.0,
                                    base_mass_amu = 1.0,
                                    n_timesteps_mult = 1.0,
                                    Attr_n_sigmas = None,
                                    Attr_dist_nm = 1.0,
                                    name="marko_excluded_volume_with_attraction", 
                                    nbCutOffDist=None,
                                    ):
    """

    Parameters
    ----------

    trunc : float
        the kT multiplier

    """
    for param_name, param in zip(['brist_len_beads', 'brist_sep_nm', 'n_rnap'], [brist_len_beads, brist_sep_nm, n_rnaps]):
        assert (isinstance(param, list)) or (isinstance(param, tuple)), f'Type of {param_name} is {type(param)} but must be list or tuple'
    assert len(brist_len_beads) == len(brist_sep_nm), f'All params should have the same size'
    assert len(brist_len_beads) == len(n_rnaps), f'All params should have the same size'

    repul_energy = (
        "step(select(N1+N2, 0.0, ATTR_dist) + REPsigma - r) * Er;"
        "Er = select(N1+N2, REPe * (n_rnaps1 + n_rnaps2), ATTRe) * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma - 1.0;"
        ""
        "REPsigma = (R1 + R2);"
        "R1 = max(a^(5/4) * N1^(3/4) * (pi * lambda1)^(-1/4), rmin);"
        "R2 = max(a^(5/4) * N2^(3/4) * (pi * lambda2)^(-1/4), rmin);"
    ) # 
    force = openmm.CustomNonbondedForce(repul_energy)
    force.name = name
    
    
    force.addPerParticleParameter("N")
    force.addPerParticleParameter("lambda")
    force.addPerParticleParameter("n_rnaps")
    for i in range(len(brist_len_beads)):
        force.addParticle([brist_len_beads[i], sim_object.conlen*brist_sep_nm[i], n_rnaps[i]])
    

    alpha = np.exp(-1/n_timesteps_mult)
    if Attr_n_sigmas is not None:
        attr_dist = Attr_n_sigmas * (dt_fs * simtk.unit.femtosecond) * \
            simtk.unit.sqrt(
                sim_object.kT * (1-alpha**2)/ (base_mass_amu * simtk.unit.amu)
                )
    else:
        attr_dist = Attr_dist_nm * sim_object.conlen
    force.addGlobalParameter("ATTR_dist", attr_dist)

    force.addGlobalParameter("REPe", Erep * sim_object.kT)
    force.addGlobalParameter("ATTRe", Eattr * sim_object.kT)
    force.addGlobalParameter("a", brist_bead_size_nm * sim_object.conlen)
    force.addGlobalParameter("rmin", min_R_nm * sim_object.conlen)
    force.addGlobalParameter("pi", np.pi)

    radiusMult = max((brist_bead_size_nm**(5/4) * np.array(brist_len_beads)**(3/4) * (np.pi * np.array(brist_sep_nm))**(-1/4)).max(), min_R_nm)
    nbCutOffDist = 2*(sim_object.conlen*((nbCutOffDist or radiusMult)) + attr_dist)
    force.setCutoffDistance(nbCutOffDist)

    return force

def prepared_object_force(sim_object, force_vecs, bourder,name="Prepared"):
    """
    adds force pulling on each particle
    particles: list of particle indices
    force_vecs: list of forces [[f0x,f0y,f0z],[f1x,f1y,f1z], ...]
    if there are fewer forces than particles forces are padded with forces[-1]
    """
    force = openmm.CustomExternalForce("step(abs(z)-b)*z*z * fz")
    
    force.name = name
    force.addPerParticleParameter("fz")
    force.addGlobalParameter("b", bourder * sim_object.conlen)
    # for num, force_vec in itertools.zip_longest(particles, force_vecs, fillvalue=force_vecs[-1]):
    #     force_vec = [float(f) * (sim_object.kT / sim_object.conlen) for f in force_vec]
    #     force.addParticle(int(num), force_vec)
    for i,f in enumerate(force_vecs):
        force.addParticle(i,[float(f) * (sim_object.kT / sim_object.conlen)])

    return force


def line_force(sim_object, force_vecs, name="line"):
    """
    adds force pulling on each particle
    particles: list of particle indices
    force_vecs: list of forces [[f0x,f0y,f0z],[f1x,f1y,f1z], ...]
    if there are fewer forces than particles forces are padded with forces[-1]
    """
    force = openmm.CustomExternalForce("x * fx")
    
    force.name = name
    force.addPerParticleParameter("fx")
    for i,f in enumerate(force_vecs):
        force.addParticle(i,[float(f) * (sim_object.kT / sim_object.conlen)])

    return force



class bondUpdater(object):

    def __init__(self, LEFpositions, N):
        """
        :param smcTransObject: smc translocator object to work with
        """
        self.LEFpositions = LEFpositions
        self.curtime  = 0
        self.allBonds = []
        self.N = N

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict


    def setup(self, bondForce, nonbondForce,  blocks = 100, smcStepsPerBlock = 1, except_bonds = False, verbose = False):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """
        

        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce
        self.nonbondForce = nonbondForce
        if hasattr(self.nonbondForce, "addException"):
            self.nonbond_base_exepts = nonbondForce.getNumExceptions()
        elif hasattr(self.nonbondForce, "addExclusion"):
            self.nonbond_base_exepts = nonbondForce.getNumExclusions() 

        #precalculating all bonds
        allBonds = []
        
        loaded_positions  = self.LEFpositions[self.curtime : self.curtime+blocks]
        allBonds = [[(int(loaded_positions[i, j, 0]), int(loaded_positions[i, j, 1])) 
                        for j in range(loaded_positions.shape[1])] for i in range(blocks)]
        
        allBonds = [[i for i in s if sum(i)>=0] for s in allBonds]
        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset) # changed from addBond
            self.bondInds.append(ind)
        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}
        
        self.curtime += blocks
        
        if except_bonds:
            sorted_curBonds = np.sort(np.array(self.curBonds), axis=1)
            if hasattr(self.nonbondForce, "setExclusionParticles"):
                tmp_bonds_in_exept = [tuple(self.nonbondForce.getExclusionParticles(i)) for i in range(self.nonbond_base_exepts)]
                bonds_for_add = []
                for bond in sorted_curBonds:
                    if not (tuple(bond) in tmp_bonds_in_exept):
                        bonds_for_add += [tuple(bond)]
                        
                if hasattr(self.nonbondForce, "addException"):
                    exc = list(set([tuple(i) for i in np.sort(np.array(bonds_for_add), axis=1)]))
                    for pair in exc:
                        self.nonbondForce.addException(pair[0], pair[1], 0, 0, 0, True)

                    num_exc = self.nonbondForce.getNumExceptions()

                # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
                elif hasattr(self.nonbondForce, "addExclusion"):
                    self.nonbondForce.createExclusionsFromBonds([(b[0], b[1]) for b in np.sort(np.array(bonds_for_add), axis=1)], int(except_bonds))
                    num_exc = self.nonbondForce.getNumExclusions()

                if verbose:
                    print("Number of exceptions after milker.setup:", num_exc)
            else:
                print('Cannot make exeptions')
        
        return self.curBonds,[]


    def step(self, context, verbose=False, except_bonds = False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation and run setup  again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),
                                                                            len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        
        if except_bonds:
            if hasattr(self.nonbondForce, "addException"):
                exepts = self.nonbondForce.getNumExceptions()
            elif hasattr(self.nonbondForce, "addExclusion"):
                exepts = self.nonbondForce.getNumExclusions()
            
            sorted_curBonds = np.sort(np.array(self.curBonds), axis=1)
            if hasattr(self.nonbondForce, "setExclusionParticles"):
                
                tmp_bonds_in_exept = [self.nonbondForce.getExclusionParticles(i) for i in range(exepts)]
                bonds_for_add = []
                for bond in sorted_curBonds:
                    if not (tuple(bond) in tmp_bonds_in_exept):
                        bonds_for_add += [tuple(bond)]
                        
                breaked = 0
                for i_b, bond in enumerate(bonds_for_add):
                    if self.nonbond_base_exepts + i_b < exepts:
                        self.nonbondForce.setExclusionParticles(self.nonbond_base_exepts + i_b, bond[0], bond[1])
                    else:
                        breaked = 1
                        break
                tmp_left_bonds = bonds_for_add[i_b + 1 - breaked:]
                
                if len(tmp_left_bonds) > 0:
                    if hasattr(self.nonbondForce, "addException"):
                        exc = list(set([tuple(i) for i in np.sort(np.array(tmp_left_bonds), axis=1)]))
                        for pair in exc:
                            self.nonbondForce.addException(pair[0], pair[1], 0, 0, 0, True)

                    # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
                    elif hasattr(self.nonbondForce, "addExclusion"):
                        self.nonbondForce.createExclusionsFromBonds([(b[0], b[1]) for b in np.sort(np.array(tmp_left_bonds), axis=1)], int(except_bonds))

                if verbose:
                    if hasattr(self.nonbondForce, "addException"): 
                        num_exc = self.nonbondForce.getNumExceptions()
                    elif hasattr(self.nonbondForce, "addExclusion"):
                        num_exc = self.nonbondForce.getNumExclusions()
                    print("Number of exceptions in milker.step:", num_exc)
                self.nonbondForce.updateParametersInContext(context)
            else:
                print('Cannot make exeptions')
        
        return self.curBonds, pastBonds
    
def slice_str_parser(chosen_slice_str):
    chosen_slice = chosen_slice_str.split(':')
    chosen_slice = [chosen_slice[0], *chosen_slice[1].split('-')]
    chosen_slice[1] = int(chosen_slice[1].replace(',', ''))
    chosen_slice[2] = int(chosen_slice[2].replace(',', ''))
    return chosen_slice

def gene_intersection_corrector(chosen_genes):
    new_indexes = []
    for i in chosen_genes.index:
        intersected = False
        for k in chosen_genes.index:
            if i != k:
                if (chosen_genes.loc[i,'start'] >= chosen_genes.loc[k,'start']) and (chosen_genes.loc[i,'end'] <= chosen_genes.loc[k,'end']):
                    intersected = True
        if not intersected:
            new_indexes += [i]
    
    chosen_genes = chosen_genes.loc[new_indexes, :]
    for i in new_indexes:
        for k in new_indexes:
            if i < k:
                if chosen_genes.loc[i, 'end'] > chosen_genes.loc[k, 'start']:
                    if chosen_genes.loc[i, 'FPKM'] >= chosen_genes.loc[k, 'FPKM']:
                        chosen_genes.loc[k, 'start'] = chosen_genes.loc[i, 'end']
                    else:
                        chosen_genes.loc[i, 'end'] = chosen_genes.loc[k, 'start']
    
    chosen_genes['len'] = chosen_genes['end'] - chosen_genes['start']
    chosen_genes = chosen_genes.sort_index().reset_index(drop=True)
    return chosen_genes

def make_start_conf(N, BondDist, starting_conformation = None, starting_conf_file = None):
    if starting_conf_file == None:
        if starting_conformation == 'grow_cubic':
            dens = 0.1
            box = (N / dens) ** 0.33 
            data = BondDist.min() * np.vstack(grow_cubic(N, int(box) - 2))
        elif starting_conformation == 'random_walk':
            data = BondDist*np.array(random_walk(N)).T
        elif starting_conformation == 'one_line':
            #data = BondDist*create_line(N)
            x = np.arange(0, N, 1)
            y = np.zeros(N)
            z = np.zeros(N)
            data = BondDist*np.vstack([x, y, z]).T
        elif starting_conformation == 'sis_lines':
            dens = 0.1
            x = np.concatenate([np.arange(0, N//2, 1), np.arange(N//2, 0, -1)])
            y = np.concatenate([np.zeros(N//2),np.zeros(N//2)+1])
            z = np.concatenate([np.zeros(N//2),np.zeros(N//2)+1])
            print(len(x), len(y))
            data = BondDist*np.vstack([x, y, z]).T*dens
        elif starting_conformation == 'spiral':
            rt = 100
            x = rt*np.cos(np.arange(0, N, 1)*0.1)
            y = rt*np.sin(np.arange(0, N, 1)*0.1)
            z = np.arange(0, N, 1)*0.1
            data = BondDist*np.vstack([x, y, z]).T
        elif starting_conformation == 'fractal':
            base_transition = np.array([[1,0,1,0,1], [0,1,0,-1,0], [0]*5])
            conformation = np.hstack([[[0],[0],[0]], fractal_conf_maker(N//2, base_transition).cumsum(axis=1)])[:,:N//2]
            conformation = np.hstack([conformation, conformation[:,::-1] + np.array([[0],[0],[1]])])
            conformation = conformation - conformation.mean(axis=1)[:,np.newaxis]            
            data = BondDist.min() * conformation.T
        else:
            print('Не выбрана ни одна из имеющихся конформаций, по умолчанию установлен grow_cubic')
            dens = 0.1
            box = (N / dens) ** 0.33 
            data = BondDist*grow_cubic(N, int(box) - 2)
    else:
        data = load_URI(starting_conf_file)
        data = data['pos'].tolist()
    return data

def index_calculator(pos):
    if pos < N_raw/2:
        return pos//beads_gr_size  
    else:
        return N - (N_raw - 1 - pos)//beads_gr_size - 1
    
