from solver import Solver
from model_clb import model_clb, param_references
import os
import numpy as np

if __name__ == "__main__":

    filename = "results_optimization\\cblb_random_start" 
    
    #model = BioProc(np.array(["protein_production", "protein_production", "protein_production", "protein_production", "protein_degradation", "protein_degradation", "Kd","hill", "protein_production", "protein_degradation", "Kd", "hill"]), model_mode=three_bit_processor_ext, parameter_values=param_values, avg_dev=30)                                         
    model = model_clb()
    solver = Solver(model, populationSize=1000, random_candidates=True)                         
    solver.run(filename, maxDepth=1) #do not cluster