from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import OllamaLLM
from deepeval.models.base_model import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel 
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import re



class LocalOllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="qwen2.5:7b"):
        self.model_name = model_name
        self.model = self.load_model()

 
    def load_model(self):
        # as the quen model is giving the json format hence updated the format as json 
        return OllamaLLM(model=self.model_name, format="json")

    def _extract_json(self, text: str) -> str:
        #Extracts raw JSON 
        text = text.strip()
        
        # removing the  markdown code if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()
        
        # Extract JSON payload between outermost braces or brackets
        json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if json_match:
            return json_match.group(1).strip()
            
        return text


    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt)

    async def a_generate(self, prompt: str) -> str:
        # Using ainvoke for the asynchronous implementation
        return await self.model.ainvoke(prompt)

    def get_model_name(self) -> str:
        return self.model_name



class LocalOllamaEmbedder(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.model = OllamaEmbeddings(model=self.model_name)

    def load_model(self):
        return self.model

    def embed_text(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.model.aembed_query(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.model.aembed_documents(texts)

    def get_model_name(self) -> str:
        return self.model_name



if __name__ == "__main__":
    
    judge = LocalOllamaJudge(model_name="qwen2.5:7b")
    
    
    response = judge.generate("what is largest country on earth according to population ?")
    
    
    print("Response:",response)
    print("nModel Used:",judge.get_model_name())
    
