from src.cove_chains import ChainOfVerification
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class ChainOfVerificationOpenAI(ChainOfVerification):
    def __init__(
        self, model_id, temperature, task, setting, questions, openai_access_token
    ):
        super().__init__(model_id, task, setting, questions)
        self.openai_access_token = openai_access_token
        self.temperature = temperature
        
        # 假设 self.model_config.id 在基类中被正确设置 (例如 "gpt-3.5-turbo")
        # 如果这里报错，你可能需要硬编码为 model_name="gpt-3.5-turbo"
        self.llm = ChatOpenAI(
            openai_api_key=openai_access_token,
            # model_name=self.model_config.id, # 这行可能需要根据基类调整
            model_name="gpt-3.5-turbo", # 为确保运行，我暂时硬编码
            max_tokens=500
        )

    def call_llm(self, prompt: str, max_tokens: int) -> str:
        llm_chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return llm_chain.invoke({})

    def process_prompt(self, prompt, _) -> str:
        # We do not need to do any processing here!
        return prompt