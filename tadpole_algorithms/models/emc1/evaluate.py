def main(eval_df,flag_D3):
    from tadpole_algorithms.evaluation import evaluate_forecast
    import pandas as pd
    import os

    str_exp=os.path.dirname(os.path.realpath(__file__))
    os.chdir(str_exp)
    
    f = open("intermediatedata.path", "r")
    IntermediateFolder = f.read()
    f.close()

    if flag_D3==0:
        forecast_df=pd.read_excel(IntermediateFolder+'/TADPOLE_Submission_EMC1.xlsx',sheet_name='ID 1')
    elif flag_D3==1:
        forecast_df=pd.read_excel(IntermediateFolder+'/TADPOLE_Submission_EMC1.xlsx',sheet_name='ID 5')
    dictionary = evaluate_forecast(eval_df, forecast_df)
    return dictionary