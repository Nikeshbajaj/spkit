import numpy as np
import pandas as pd


def ModelTable(model, style=0,show=False,include_zero_param=True):
    Table = {'name':[],'type':[],'input-shape':[],'output-shape':[],'#param':[]}
    for layer in model.layers:
        npar = np.sum([np.prod(w.shape) for w in layer.trainable_weights])
        if not(include_zero_param) and npar==0:
            pass
        else:
            Table['#param'].append(npar.astype(int))

            Table['name'].append(layer.name)
            Table['type'].append(layer.__class__.__name__)
            Table['input-shape'].append(layer.input_shape)
            Table['output-shape'].append(layer.output_shape)
        
                
    Table = pd.DataFrame(Table)
    
    if style==1:
        Table = Table[['type','output-shape','#param']]
    
    if show:
        try: 
            display(Table)
        except:
            try:
                from IPython import display
                display(Table)
            except:
                print(Table)
    return Table