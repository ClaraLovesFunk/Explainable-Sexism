import numpy as np
import pandas as pd
from IPython.display import display

results = np.load('results.npy',allow_pickle='TRUE').item()
#print(results)
#display(pd.DataFrame([results]))
'''        ###
        print('-----------------------------------------------------------')
        print('-----------------------------------------------------------')
        print(f'{model_id}_bal-{b}_trained-{t}:')
        print(f"f1: ", perf_metrics['f1'])
        print(f'acc: ', perf_metrics['acc'])
        print('-----------------------------------------------------------')
        print('-----------------------------------------------------------')'''