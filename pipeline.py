from email_data import EmailDatasetGenerator
import pandas as pd 

df=pd.read_csv('email_dataset.csv')
generator=EmailDatasetGenerator(model='gpt-4o-mini',dataset=df,
                                    output_path='email_data_v1.csv',
                                    max_req_per_minute=100)
generator.get_goldens_from_dataset()