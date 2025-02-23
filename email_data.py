import tempfile
import os 
import asyncio
from tqdm.asyncio import tqdm_asyncio
import uuid
import json
from utils import get_or_create_event_loop
import pandas as pd
from prompts import EmailPromptTemplate
from llm import AsyncLLManagers
from models import EmailResponse,EvaluateResponse,EvolvedResponse

class EmailDatasetGenerator:
    def __init__(self,model,dataset,output_path,max_req_per_minute=25,evolve=False):
        self.llm=AsyncLLManagers(model=model)
        if isinstance(dataset,str):
            self.dataset=pd.read_csv(dataset)
        else:
            self.dataset=dataset 
        self.dataset=self.dataset[2000:]
        self.emails=list(self.dataset['email'])
        self.out_path=output_path
        self.max_req=max_req_per_minute

        self.evolve=evolve

        self.temp_dir = tempfile.mkdtemp(dir=os.getcwd())
        print(f'created temfile at {self.temp_dir}')

    def get_goldens_from_dataset(self):
        goldens=[]
        loop=get_or_create_event_loop()
        goldens.extend(
            loop.run_until_complete(
                self.a_generate_goldens_from_dataset(
                    max_request=self.max_req

                )
            )
        )
        #print(goldens[0])

        self._save_output(goldens=goldens)
    
    async def a_generate_goldens_from_dataset(self, max_request):
        goldens=[]
        semaphore=asyncio.Semaphore(max_request)

        tasks=[
            self.task_wrapper(
                semaphore,
                self._a_generate_email_response,
                email=email,
                goldens=goldens

            ) for index, email in enumerate(self.emails)
        ]

        await tqdm_asyncio.gather(*tasks,desc="Generating Goldens ...")

        return goldens
    
    async def _a_generate_email_response(self,email,goldens):
        #generating_reply 
        response_prompt=EmailPromptTemplate.generate_synthetic_email_response(email)
        response=await self.llm.get_completion(prompt=response_prompt,response_format=EmailResponse)

        golden_dict={'email':email,'response':response.email_response}
        file_name=f'{str(uuid.uuid4())}.json'
        file_path=os.path.join(self.temp_dir,file_name)
        with open(file_path,'w') as f:
            json.dump(golden_dict,f,indent=4)
        
        goldens.append(golden_dict)

    async def task_wrapper(self,sem:asyncio.Semaphore,func,*args,**kwargs):
        async with sem:
            return await func(*args,**kwargs)
    
    def _save_output(self,goldens):
        final_df=pd.DataFrame(goldens)
        final_df.to_csv(self.out_path)
        print(f'file saved at {self.out_path}')

if __name__=='__main__':
    
    df=pd.read_csv('email_dataset.csv')
    generator=EmailDatasetGenerator(model='gpt-4o-mini',dataset=df,
                                    output_path='email_data_v1.csv',
                                    max_req_per_minute=100)
    generator.get_goldens_from_dataset()