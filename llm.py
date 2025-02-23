from openai import AsyncAzureOpenAI, AsyncOpenAI, APIError, APIConnectionError, APITimeoutError
from dotenv import load_dotenv
import os 
import logging
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    before_log,
    after_log,
)
from pydantic import ValidationError
import asyncio

os.makedirs("logs",exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="logs/retriever_logging.txt"
)
logger = logging.getLogger(__name__)

load_dotenv(".env")

AZURE_GPT_MINI_DEPLOYMENT=os.getenv('AZURE_GPT_MINI_DEPLOYMENT')
AZURE_OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
API_KEY=os.getenv('OPENAI_API_KEY')


def get_async_azure_mini_openai_client() -> AsyncAzureOpenAI:
    azure_mini_client = AsyncAzureOpenAI(
        azure_deployment=AZURE_GPT_MINI_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-08-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    return azure_mini_client

class AsyncLLManagers:
    def __init__(self,model):
        self.client=self._get_async_client()
        self.model=model
    
    def _get_async_client(self):
        
        client=get_async_azure_mini_openai_client()
        #client=AsyncOpenAI(api_key=API_KEY)
        return client
    
    def should_retry(self,exception:Exception)->bool:
        
        logger.warning(f"Retrying due to exception: {exception}")
        return True  
    
    @retry(
        retry=(
        retry_if_exception_type(Exception)
        ),
        stop=stop_after_attempt(1000000), 
        wait=wait_exponential(multiplier=2, min=1, max=60), 
        before_sleep=before_sleep_log(logger,log_level=logging.WARNING),
        before=before_log(logger,logging.DEBUG),
        retry_error_callback=lambda retry_state: retry_state.outcome.result()

    )
    async def get_completion(self,prompt,response_format):

        messages=[{"role":"system","content":"return a valid response as specified"},
                  {"role":"user","content":prompt}]
        try:
            logger.info(f"Initiating async API Call.")
            response=await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                timeout=240,

            )
            logger.debug("Sucessfull API call.")
            parsed_response=response.choices[0].message.parsed

            if not isinstance(parsed_response, response_format):
                logger.warning(
                    f"Invalid response format: Expected {response_format}, but got {type(parsed_response)}"
                )
                raise TypeError("Response does not match the expected Pydantic model format.")

            return response.choices[0].message.parsed
        except Exception as e:
            logger.warning(f"API call failed: {str(e)}")
            #if not self.should_retry(e):
            #   logger.error("Non retryable error encountered")
            #  raise 
            logger.info("Retryable error detected. Scheduling retry...")
            raise e
        
if __name__=='__main__':

    azure_client=get_async_azure_mini_openai_client()

    from models import SyntheticData
    async def main():
        response=await azure_client.beta.chat.completions.parse(
            messages=[{"role":"system","content":"return me a valid question"}],
            model='gpt-4o-mini',
            response_format=SyntheticData
        )

        print(response.choices[0].message.parsed)
    
    asyncio.run(main())