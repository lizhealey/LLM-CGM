# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pickle
import pandas as pd
import numpy as np


def get_questions():
    questions = ['Q1:What was my mean glucose?',
    'Q2:What was my maximum glucose?',
    'Q3:What was the standard deviation of my glucose? ',
    'Q4:What was my minimum glucose?',
    'Q5:What was my percent time in range? ',
    'Q6:What was my percent time in hyperglycemia?',
    'Q7:What was my percent time in hypoglycemia?',
    'Q8:What was my glycemic variability?',
    'Q9:What was my percent time in severe hyperglycemia?',
    'Q10:What is my estimated A1C?',
    'Q11:What was my percent time in severe hypoglycemia?',

    'Q12:What time was my blood glucose highest?',
    'Q13:What day was my glucose control the most out of range?',
    'Q14:What time of the day was my blood glucose lowest? ',
    'Q15:When did my most recent episode of hypoglycemia start?',
    'Q16:How long was my last episode of hypoglycemia?',
    'Q17:What was my longest time spent in hyperglycemia? ',
    'Q18:How many times did I experience hypoglycemia? ',
    'Q19:What was my mean overnight blood glucose? ',
    'Q20:What period of the day did I have the highest blood glucose?',
    'Q21:Did I have noctural hypoglycemia? ',
    'Q22:What was my highest glucose reading during dinner? ',

    'Q23:What percent of time was my CGM active?',
    'Q24:How many times did my sensor disconnect?',
    'Q25:Was my low blood glucose likely due to sensor error?',
    'Q26:Are there any artifacts in the CGM data?',

    'Q27:Was my average glucose control today better than yesterday?',
    'Q28:Was my time in range improved this week compared to last week? ',
    'Q29:Was my max glucose lower today than yesterday?',
    'Q30:Did I spend less time in hypoglycemia this week than last week?']
    return questions


def get_answers(data_all, name_, data):
    #################################################################################
    # Q 1-11 
    #################################################################################
    df = data.copy()
    data_all.loc[name_,'Q1:Average']  = np.mean(data['CGM'])
    data_all.loc[name_,'Q2:Maximum'] = np.max(data['CGM'])
    data_all.loc[name_,'Q3:STD' ]= np.std(data['CGM'])
    data_all.loc[name_,'Q4:Mininum'] = np.min(data['CGM'])
    tbh= data[data['CGM']<180]
    data_all.loc[name_,'Q5:TIR'] = len(tbh[tbh['CGM']>70])/(len(data['CGM']))
    data_all.loc[name_,'Q6:TAR 1 (>180)'] = len(data[data['CGM']>180])/(len(data['CGM']))
    data_all.loc[name_,'Q7:TBR 1 (<70)'] = len(data[data['CGM']<70])/(len(data['CGM']))
    data_all.loc[name_,'Q8:CV' ]= np.std(data['CGM'])/np.mean(data['CGM'])
    data_all.loc[name_,'Q9:TAR 2 (>250)'] = len(data[data['CGM']>250])/(len(data['CGM']))
    data_all.loc[name_,'Q10:eA1c'] =( 46.7+np.mean(data['CGM'])) / 28.7
    
    data_all.loc[name_,'Q11:TBR 2 (<54)'] = len(data[data['CGM']<54])/(len(data['CGM']))
    
    #################################################################################
    # Q 12-15
    #################################################################################
    data_all.loc[name_,'Q12:Highest time'] =df.loc[df['CGM'].idxmax(), 'Time']
    
    
    df['Date'] = df['Time'].dt.date
    oor = df[(df['CGM'] < 70) | (df['CGM'] > 180)]
    data_all.loc[name_,'Q13:Day out of range'] =oor['Date'].value_counts().idxmax()  
    
    #Q14
    data_all.loc[name_,'Q14: lowest'] = df.loc[df['CGM'].idxmin(), 'Time'].time()

    #Q15
    hypos = df[df['CGM'] < 70]
    if not hypos.empty:
        recent = hypos['Time'].iloc[-1]
    else:
        recent= np.nan
    data_all.loc[name_,'Q15: most recent low'] =recent

    #################################################################################
    # Q16 and Q17 
    #################################################################################
    df['lessthan70'] = df['CGM'] < 70
    df['starthypo'] = df['lessthan70'] & ~df['lessthan70'].shift(1, fill_value=False)
    df['Group'] = df['starthypo'].cumsum()
    hypo = df[df['lessthan70']]
    lengths = hypo.groupby('Group').size()
    if len(lengths) >1: 
        data_all.loc[name_,'Q16:most recent length of low'] =  lengths[-1:].values[0]
    else:
        data_all.loc[name_,'Q16:most recent length of low'] = np.nan
    df['hyper'] = df['CGM'] >180
    df['starthyper'] = df['hyper'] & ~df['hyper'].shift(1, fill_value=False)
    df['Group'] = df['starthyper'].cumsum()
    hyper = df[df['hyper']]
    lengths = hyper.groupby('Group').size()
    if len(lengths)>0:
        data_all.loc[name_,'Q17:max length in hyper'] =max(lengths)*5
    else:
        data_all.loc[name_,'Q17:max length in hyper'] =np.nan
    
    #################################################################################
    # Q18 and Q19 --> related to meal times
    #################################################################################
    df['temp'] = df['starthypo'].cumsum()
    df_counthypo = df[df['lessthan70']]
    lengths = df_counthypo.groupby('temp').size()
    data_all.loc[name_,'Q18:number of hypo'] =len(lengths)

    night = df[(df['Time'].dt.hour >= 0) & (df['Time'].dt.hour < 6)]
    mean_night = night['CGM'].mean()
    data_all.loc[name_,'Q19']=  mean_night
    
    def mealtime(time):
        if 6 <= time < 12:
            return 'morning'
        elif 12 <= time < 18:
            return 'afternoon'
        elif 18 <= time < 24:
            return 'evening'
        else:
            return 'night'
    df['Part'] = df['Time'].dt.hour.apply(mealtime)
    highest = df.loc[df['CGM'].idxmax(), 'Part'] 
    data_all.loc[name_,'Q20'] =highest

    # Noctural hypo between 0 and 6
    night = df[(df['Time'].dt.hour >= 0) & (df['Time'].dt.hour < 6)]
    night_hypo = night[night['CGM'] < 70]
    data_all.loc[name_,'Q21'] =not night_hypo.empty
    
    # Dinner between 5 and 10
    din= df[(df['Time'].dt.hour >= 17) & (df['Time'].dt.hour < 22)]
    highest_din = din['CGM'].max()
    data_all.loc[name_,'Q22']= highest_din
    
    
    #################################################################################
    # Q23 and Q24
    #################################################################################
    df = df.sort_values(by='Time')
    first_value = df['Time'].iloc[0]
    last_value= df['Time'].iloc[-1]
    total_time_span = (last_value - first_value).total_seconds() / 60
    expected = total_time_span / 5
    actual = len(df)
    data_all.loc[name_,'Q23:Percent active'] = (actual / expected) * 100
    
    df['diff_time'] = df['Time'].diff()
    data_all.loc[name_,'Q24:Time disconnect'] = df['diff_time'].max()
    
    
    #################################################################################
    # Sensor artifacts 
    # Will depend on sensor
    #################################################################################
    data_all.loc[name_,'Q25:Low glucose sensor error'] =np.nan
    data_all.loc[name_,'Q26:Artifacts'] = np.nan
    
    
    #################################################################################
    # Q27, Q29 daily comparisons
    #################################################################################
    most_recent= df['Time'].dt.date.max()
    yest = most_recent - pd.Timedelta(days=1)
    df_today= df[df['Time'].dt.date == most_recent]
    df_yest = df[df['Time'].dt.date == yest]
    avg_glucose_today = df_today['CGM'].mean()
    avg_glucose_yest = df_yest['CGM'].mean()
    data_all.loc[name_,'Q27:better'] =avg_glucose_today < avg_glucose_yest
    max_glucose_today = df_today['CGM'].max()
    max_glucose_yest = df_yest['CGM'].max()
    data_all.loc[name_,'Q29:max'] = max_glucose_today < max_glucose_yest
    
    
    #################################################################################
    #Q30, Q28  Weekly comparisons
    ### Update to reflect "This week/last week"
    #################################################################################
    lastweek = most_recent - pd.Timedelta(days=7)
    lastweek_start = most_recent- pd.Timedelta(days=14)
    df_thisweek = df[(df['Time'].dt.date > lastweek)]
    df_lastweek = df[(df['Time'].dt.date < lastweek) & (df['Time'].dt.date <= lastweek_start)]
    time_hypo_thisweek = (df_thisweek['CGM'] < 70).mean()
    time_hypo_lastweek = (df_lastweek['CGM'] < 70).mean()
    data_all.loc[name_,'Q30:hypo'] = time_hypo_thisweek<time_hypo_lastweek 
    tir_thisweek = ((df_thisweek['CGM'] > 70) & (df_thisweek['CGM'] <180 )).mean()
    tir_lastweek = ((df_lastweek['CGM'] > 70) & (df_lastweek['CGM'] <180)).mean()
    data_all.loc[name_,'Q28:tir'] = tir_thisweek>tir_lastweek 

    return data_all
