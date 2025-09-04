import pickle
import os
import time
import h5py 
import polychrom
import numpy as np
import pandas as pd

import polychrom.hdf5_format as hdf5
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
#from polychrom.starting_conformations import create_line
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file

import simtk.openmm 
import os 
import shutil

import warnings
import glob


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
            if hasattr(self.nonbondForce, "setExclusionParticles"):
                
                tmp_bonds_in_exept = [tuple(self.nonbondForce.getExclusionParticles(i)) for i in range(self.nonbond_base_exepts)]
                bonds_for_add = []
                for bond in self.curBonds:
                    if not (bond in tmp_bonds_in_exept):
                        bonds_for_add += [bond]
                        
                if hasattr(self.nonbondForce, "addException"):
                    exc = list(set([tuple(i) for i in np.sort(np.array(bonds_for_add), axis=1)]))
                    for pair in exc:
                        self.nonbondForce.addException(pair[0], pair[1], 0, 0, 0, True)

                    num_exc = self.nonbondForce.getNumExceptions()

                # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
                elif hasattr(self.nonbondForce, "addExclusion"):
                    # for b in bonds_for_add:
                    #     if b[0] < 0 or b[1] < 0:
                    #         print(b)
                    #     if b[0] >= 17620 or b[1] >= 17620:
                    #         print(b)
                    self.nonbondForce.createExclusionsFromBonds([(b[0], b[1]) for b in bonds_for_add], int(except_bonds))
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
                
            if hasattr(self.nonbondForce, "setExclusionParticles"):
                
                tmp_bonds_in_exept = [self.nonbondForce.getExclusionParticles(i) for i in range(exepts)]
                bonds_for_add = []
                for bond in self.curBonds:
                    if not (bond in tmp_bonds_in_exept):
                        bonds_for_add += [bond]
                        
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
                        self.nonbondForce.createExclusionsFromBonds([(b[0], b[1]) for b in tmp_left_bonds], int(except_bonds))

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
    
def boundsfromradiuses(radiuses, mul_rad = 1.0):
    siz=len(radiuses)
    bondlenght=np.zeros(siz-1)
    for i in range(siz-1):
        r = radiuses[i]+radiuses[i+1]
#         if r<bondLength:
#             r=bondLength
        bondlenght[i]=r*mul_rad
    bondlenght = np.concatenate((bondlenght, bondlenght[::-1]))
#     bondlenght[siz]=r
#     for i in range(2*(siz-1)-1,siz,-1):
#         r=radiuses[2*(siz-1)+1-i]+radiuses[2*(siz-1)+1-i+1]
# #         if r<20:
# #             r=20
#         bondlenght[i]=r*mul_rad
    return bondlenght.tolist()

def create_line(N):
    # Просто прямая линия вдоль оси х
    x = np.arange(0, N, 1)
    y = np.zeros(N)
    z = np.zeros(N)

    return np.vstack([x, y, z]).T

def random_walk(N):
    # Случайные блуждания
    r = 1
    x = y = z = np.array([0])
    for i in range(1,N):
        theta = np.random.sample(1)*np.pi
        phi = 2*np.random.sample(1)*np.pi
        #print(i)
        x=np.append(x,r*np.sin(theta)*np.cos(phi)+x[-1])
        y=np.append(y,r*np.sin(theta)*np.sin(phi)+y[-1])
        z=np.append(z,r*np.cos(theta)+z[-1])
        #print((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2)
    return np.vstack([x, y, z])

def fractal_conf_maker(N, base_transition):
    if base_transition.shape[1]>=N:
        return base_transition
    else:
        new_ys = np.array(base_transition[(base_transition[0,:]==0)*1,:] * (base_transition[0,:]+base_transition[1,:])[:,np.newaxis], dtype=int).reshape(-1)
        new_zs = np.array(base_transition[(base_transition[1,:]==0)*1,:] * (base_transition[0,:]+base_transition[1,:])[:,np.newaxis], dtype=int).reshape(-1)
        assert len(new_ys) == len(new_zs)
        new_base_transition = np.array([new_ys.tolist(), new_zs.tolist(), [0]*len(new_ys)])

    return fractal_conf_maker(N, new_base_transition)
    
def make_3D_model(path_to_LEFs = '/storage2/vvkonstantinov/Polychrom_files/polychrom-master/examples/loopExtrusion/trajectory/',
                 path_to_file = '/storage2/vvkonstantinov/Polychrom_files/polychrom-master/examples/loopExtrusion/trajectory/',
                 steps = 100, # Число шагов молекулярной динамики
                 saveEveryBlocks = 100,   # save every 100 blocks (saving every block is now too much almost)
                 restartSimulationEveryBlocks = 2000,# Интервал перезапуска симуляции с сохранением положений всех точек
                 smcBondDist = 20.0, # Размер LEFа в нм
                 bondLength = 20.0, # Расстояние между узлами (нуклеосомами) хроматина в нм
                 robustness = 0.005, # Жёсткость связи, соответвует ширине шага в 1 сигма для гаусова расспределения
                 overwrite=True, # Параметр отвечающий за перезапись файлов в папке при попытке начать новую запись в папку с уже имеющимися данными (например, симуляция оборвалась)
                 collision_rate = 0.01, # Наиболее близко к понятию "вязкости", т.е. частота и сила столкновений с молекулами воды, если параметр слишком велик, то может возникнуть сильное растяжение связей, ибо это будет энергетически выгоднее с точки зрения кинетической либо потенциальной энергии
                  n_timesteps_mult = 1.0,
                 mass = 1e-1,
                  starting_conformation = None, # Возможные начальные конформации: 1 = grow cubic, 2 = random walk, 3 = line
                 starting_conf_file = None,
                 platform = 'CUDA',
                 rna_info = './smallradiuses.csv',
                 rad = 7.0,
                 save_decimals = 5,
                 trunc = 1e1,
                  bond_len_mult = 1.0,
                  except_bonds_from_updater = False,
                 ):
    
    # -------defining parameters----------
    #  -- basic loop extrusion parameters

    myfile = h5py.File(path_to_LEFs+"/LEFPositions.h5", mode='r')
    path_to = path_to_file
    if not os.path.exists(path_to):
        os.mkdir(path_to)
    path_to += '/'
    N = myfile.attrs["N"]
    #LEFNum = myfile.attrs["LEFNum"]
    LEFpositions = myfile["positions"]

    Nframes = LEFpositions.shape[0]


    if starting_conf_file == None:
        if starting_conformation == 'grow_cubic':
            dens = 0.1
            box = (N / dens) ** 0.33 
            data = smcBondDist*grow_cubic(N, int(box) - 2) 
        elif starting_conformation == 'random_walk':
            data = smcBondDist*np.array(random_walk(N)).T
        elif starting_conformation == 'one_line':
            #data = smcBondDist*create_line(N)
            x = np.arange(0, N, 1)
            y = np.zeros(N)
            z = np.zeros(N)
            data = smcBondDist*np.vstack([x, y, z]).T
        elif starting_conformation == 'sis_lines':
            dens = 0.1
            x = np.concatenate([np.arange(0, N//2, 1), np.arange(N//2, 0, -1)])
            y = np.concatenate([np.zeros(N//2),np.zeros(N//2)+1])
            z = np.concatenate([np.zeros(N//2),np.zeros(N//2)+1])
            print(len(x), len(y))
            data = smcBondDist*np.vstack([x, y, z]).T*dens
        elif starting_conformation == 'spiral':
            rt = 100
            x = rt*np.cos(np.arange(0, N, 1)*0.1)
            y = rt*np.sin(np.arange(0, N, 1)*0.1)
            z = np.arange(0, N, 1)*0.1
            data = smcBondDist*np.vstack([x, y, z]).T
        elif starting_conformation == 'fractal':
            base_transition = np.array([[1,0,1,0,1], [0,1,0,-1,0], [0]*5])
            conformation = np.hstack([[[0],[0],[0]], fractal_conf_maker(N//2, base_transition).cumsum(axis=1)])[:,:N//2]
            conformation = np.hstack([conformation, conformation[:,::-1] + np.array([[0],[0],[1]])])
            conformation = conformation - conformation.mean(axis=1)[:,np.newaxis]            
            data = smcBondDist * conformation.T
        else:
            print('Не выбрана ни одна из имеющихся конформаций, по умолчанию установлен grow_cubic')
            dens = 0.1
            box = (N / dens) ** 0.33 
            data = smcBondDist*grow_cubic(N, int(box) - 2)
    else:
        data = hdf5.load_hdf5_file(starting_conf_file)
        x = data['pos'][:,0] 
        y = data['pos'][:,1] 
        z = data['pos'][:,2]
        data = np.vstack([x, y, z]).T
        
    block = 0  # starting block

    # parameters for smc bonds
    # Нужно провести полноценную оценку длины связи между нуклеосомами. Грубая оценка: Max size 35 nm, min 5(?), average 17+-8
    smcBondWiggleDist = smcBondDist*robustness
#     bondWiggleDistance = bondLength*robustness

    # assertions for easy managing code below
    assert (Nframes % restartSimulationEveryBlocks) == 0, f'Nframes = {Nframes}'
    assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

    savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
    simInitsTotal  = (Nframes) // (restartSimulationEveryBlocks)
    print('Nframes: ', Nframes)
    print('simInitsTotal: ',simInitsTotal)
    
    milker = bondUpdater(LEFpositions, N = N)

    reporter = HDF5Reporter(folder=path_to, max_data_length=savesPerSim, overwrite=overwrite, blocks_only=False)
    time_hist = []
    eK = []
    eP = []
    global_eneries = []
    
    radiuses = pd.read_table(rna_info, delimiter=',', header=0 , names = ['Radiuses'])
    radiuses = radiuses['Radiuses']
    bondlenght=boundsfromradiuses(radiuses, bond_len_mult)
    #bondLength = [bondLength for i in range(2*(len(radiuses)-1)+1)]
    radiuses = np.concatenate((radiuses, radiuses[::-1]))
    tmpradiuses = np.zeros(len(radiuses))+2*min(radiuses)
    tmpradiuses = tmpradiuses.tolist()
    tmpbondlenght = np.array(bondlenght)
    tmpbondWiggleDist = robustness*np.array(tmpbondlenght)
    tmpbondlenght = tmpbondlenght.tolist()
    tmpbondWiggleDist = tmpbondWiggleDist.tolist()
    molstepsmul = 1
    masses = np.zeros(len(radiuses))+1500000
    time_step = 1 / (collision_rate * n_timesteps_mult) # femtosecs, ~1000 steps per "one collision"
    #print(masses)
    for iteration in range(simInitsTotal):   
        t1 = time.time()
        eK = []
        eP = []
        print('Iteration: ',iteration)
        # simulation parameters are defined below 
        a = Simulation(
                max_Ek=1e10,
                platform=platform,
                integrator="langevin",
#                 mass = masses.tolist(), #300000, # Это 300 кДа
                mass = mass,
                error_tol=1e-10,
                timestep = time_step,
                GPU = "0", 
                collision_rate=collision_rate, # collision rate in inverse picoseconds
                N = len(data),
                reporters=[reporter],
                #PBCbox=[box, box, box],
                precision="mixed",
                save_decimals = save_decimals,
        )  # timestep not necessary for variableLangevin

        ############################## New code ##############################
        a.set_data(data, center = True)  # loads a polymer, puts a center of mass at zero
        a.add_force(
            forcekits.polymer_chains(
                a,
#                 chains=[(0, None, 1)],
                chains=[(0, N//2, False), (N//2, None, False)],
                    # By default the library assumes you have one polymer chain
                    # If you want to make it a ring, or more than one chain, use self.setChains
                    # self.setChains([(0,50,1),(50,None,0)]) will set a 50-monomer ring and a chain from monomer 50 to the end

                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    # Distance between histons
                    'bondLength':tmpbondlenght,
                    'bondWiggleDistance':tmpbondWiggleDist, # Bond distance will fluctuate +- 0.05 on average
                 },

                angle_force_func=forces.angle_force,
                angle_force_kwargs={
                    'k':1e-4,
                    #'k':5
                    # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                    # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
                },

                nonbonded_force_func=forces.polynomial_2_repulsive,
                nonbonded_force_kwargs={
                    'trunc' : trunc, # Возможность самопересечений
                    # Radius of histone
                    'radiusMult': np.array(radiuses).tolist(), # Размер бусины (7 по умолчанию было) 
                    #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
#                     'nbCutOffDist': min(tmpbondlenght),
                },
                
                except_bonds=True,
            )
        )

        # ------------ initializing milker; adding bonds ---------
        # copied from addBond
        #print('a completed')
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        bondDist = smcBondDist * a.length_scale

        activeParams = {"length":bondDist,"k":kbond}
        inactiveParams = {"length":bondDist, "k":0}
        milker.setParams(activeParams, inactiveParams)

        # this step actually puts all bonds in and sets first bonds to be what they should be
        milker.setup(bondForce = a.force_dict['harmonic_bonds'],
                     nonbondForce = a.force_dict['polynomial_2_repulsive'],
                    blocks=restartSimulationEveryBlocks,
                    except_bonds = except_bonds_from_updater)

        # If your simulation does not start, consider using energy minimization below
        if iteration==0:
            a.local_energy_minimization(random_offset = min(tmpbondWiggleDist) * 1e-1) 
#             a.local_energy_minimization() 
        else:
            a._apply_forces()

        #print('Start restartSim-cycle')
        for i in range(restartSimulationEveryBlocks):           
            #print('Compleated:', '%.1f' % ((iteration+i/restartSimulationEveryBlocks)/simInitsTotal)*100, '%')
            if i % saveEveryBlocks == (saveEveryBlocks - 1):  
                a.do_block(steps=steps*molstepsmul)
            else:
                a.integrator.step(steps*molstepsmul)  # do steps without getting the positions from the GPU (faster)
            if i < restartSimulationEveryBlocks - 1: 
                curBonds, pastBonds = milker.step(a.context, except_bonds = except_bonds_from_updater)  # this updates bonds. You can do something with bonds here
            #eK1, eP1 = a.return_energies()
            
        data = a.get_data()  # save data and step, and delete the simulation
        del a

        reporter.blocks_only = True  # Write output hdf5-files only for blocks
        
        t = time.localtime() 
        t = time.strftime("%H:%M:%S", t)
        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)
        t2 = time.time()
        time_hist.append(t2-t1)
        remTime = np.mean(time_hist)*(simInitsTotal-iteration-1)
        print('Compleated: ', round(((1+iteration)/simInitsTotal)*100,1), '%',  ' Remained time: ', 
              round(remTime/3600,1), ' hours ', 'Time for one iteration:', np.mean(time_hist)/60, 'min'
              #, 'Mean energies eK/eP: ', np.mean(eK), ' / ', np.mean(eP)
             )
        with open(path_to_LEFs + "/Outputdata.txt", "a") as file:
            file.write(('Compleated: ' + str(round(((1+iteration)/simInitsTotal)*100,1))  + '%' + ' Remained time: ' + 
              str(round(remTime/3600,1)) + ' hours ' + 'Time for one iteration:' + str(np.mean(time_hist)/60)+ 'min\n') + 
                      f'Current time:{t}\n')
    print('\_(0_0)_/ '+'\_(-o-)_/ ' + '/_(0o0)_\ ')
    reporter.dump_data()