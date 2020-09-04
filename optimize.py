"""Module to use developped neural network
to predict polish rod horsepower based on
number of input variables.
    user interface will be responsible to:
>> uplaod required model & normalization values.
>> get data of client's well existing conditions.
>> get data of client's well optimization options.
"""
#import required packages
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os


def load_model(model,norm_df):
    """function to load developed model
    args:
        model: path to the model. 
        norm_df: path to normaization values  
        used for training.
    return:
        model: keras model.  
        norm_df: normalization dataframe.
    """
    model = tf.keras.models.load_model(model)
    norm_df = pd.read_excel(norm_df).set_index('Unnamed: 0')
    return model , norm_df

def predict(pl_d,sl,spm,b_weight,pip, sp_gr, fil, norm_df=None,model=None):
    """use the dvelopped model to predict polish rod horsepower for a set
       of configurations.
       args: pl_d:plunger diameter 'in',spm: pump speed 'spm', bouyant rod weigh 'lb'
             pip: pump intake pressure 'psi', sp_gr:specific gravity, fil:fillage
             norm_df: normalization dataframe, model: developed model with ext. .h5
       returns:predicted polish rod horsepower
    """
    #normalize input data to be fed into NN model
    node_1=(pl_d-norm_df['mean']['P.Dia (in)'])/(norm_df['std']['P.Dia (in)'])
    node_2=(sl-norm_df['mean']['SL (in)'])/(norm_df['std']['SL (in)'])
    node_3=(spm-norm_df['mean']['SPM (spm)'])/(norm_df['std']['SPM (spm)'])
    node_4=(b_weight-norm_df['mean']['B.Weight (lbf)'])/(norm_df['std']['B.Weight (lbf)'])
    node_5=(pip-norm_df['mean']['PIP (psi)'])/(norm_df['std']['PIP (psi)'])
    node_6=(sp_gr-norm_df['mean']['SpGr'])/(norm_df['std']['SpGr'])
    if fil == 'low':
        node_7, node_8, node_9 =(1,0,0)
    elif fil == 'medium':
        node_7, node_8, node_9 =(0,1,0)
    elif fil == 'high':
        node_7, node_8, node_9 =(0,0,1)
    norm_input= np.array([[node_1,node_2,node_3,node_4,node_5,node_6,node_7,node_8,node_9]])
    pred = model.predict(np.array(norm_input[0],ndmin=2 ))
    pred =  pred.item()
    print(f' Expected PRHP: {pred}')
    return pred

def get_client_conditions():
    """Ask clien for existing well conditions
    inputs:
        pip: pump intake pressure (psi)
        fil: fillage (%)
        b_weight: bouyant weight (lb)
        sp_gr: specific gravity
    return: float and integer of client input for each params.
    """
    pip = float(input('PIP is: '))
    fil = input('How is fillage (low, medium, high): ')     # model use string value
    b_weight = float(input('Bouyant rod weight, lb: '))
    sp_gr= float(input('Fluid Specific gravity: '))
    return pip, fil, b_weight, sp_gr
  

def get_client_options():
    """function to ask client for existing production rate and params.
    and for avillbale options for pump dia. , stroke lenght, spm.

    return:
        list of options that produce the same production rate.
    """
    q_act = float(input('Current Production (BPD) is: '))
    pl_d_cur= float(input('Current Plunger Diameters (in): '))
    pds = input ('Selrect Avilable Plunger Diameters Separated by Space : ')
    pds = pds.split()
    sl_cur= int(input('Current Stroke lenght (in): '))
    sls = input ('Selrect Avilable Stroke Lenghts Separated by Space : ')
    sls = sls.split()
    spm_cur= float(input('Current Strokes per minute (spm): '))
    spm_min = int(input('Select SPM Minimu Value: '))
    spm_max= int(input('Select SPM Maximum Value: '))  
    eff = q_act/(0.11664*pl_d_cur*pl_d_cur*sl_cur*spm_cur)
    options = []
    for pl_d in pds:
        for sl in sls:
            spm = q_act/(0.11664*float(pl_d)*float(pl_d)*int(sl)*eff)
            if spm < spm_max and spm > spm_min:
                options.append([pl_d,sl,spm])
    print(options,'\n')
    return options

def calc_cost(prhp, power_cost):
    """function that calculate cost per year for specific PRHP
    args:
        prhp: value of polish rod horsepower
        power_cost: $/KW
    returns:
        cost_per_year: $/year
    """
    p_mot = prhp/0.8
    p_e = (p_mot/0.75) * 0.7457
    cost_per_year = p_e * power_cost * 365 * 24
    return cost_per_year

  
def main():
    results = []
    power_cost = 0.1
    # upload model & normalization matrix
    model, norm_df = load_model(model='NN_model.h5',norm_df='Normalization_matrix.xlsx')
    # get existing conditions
    pip, fil, b_weight, sp_gr = get_client_conditions()
    # get avilable options (plunger dia., SPM, SL)
    options = get_client_options()
    # iterate through options to predict for each prhp & calc cost
    for option in options:
        print(f'\n check for {option}')
        prhp = predict(float(option[0]),int(option[1]), option[2], b_weight, pip, sp_gr, fil,
                        model=model,norm_df=norm_df)
        if prhp > 0:
            cost = round(calc_cost(prhp, power_cost),2)
            print(f' cost equal: {cost}')
            option.extend([round(prhp,2), cost])
    print(options)

#driver block:
if __name__ == '__main__':
    main()