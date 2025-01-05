# Takes in data as dataframe and saves it as CSV with appropriate columns
def preprocess_cgm(data, cgm_column, time_column):
    data['CGM']=data[cgm_column]
    data['Time']=data[time_column]
    data.to_csv(f'data.csv')
    return data 
