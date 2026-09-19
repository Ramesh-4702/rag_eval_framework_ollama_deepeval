import os 
from deepeval.synthesizer.synthesizer import Synthesizer
from src.evaluator import LocalOllamaJudge ,LocalOllamaEmbedder
from deepeval.synthesizer.config import ContextConstructionConfig


def generate_golden_dataset():
    
    print("Initializing local qwen2.5:7b  judge for synthesis from wrapper")
    judge = LocalOllamaJudge(model_name="llama3.2")
    embedder = LocalOllamaEmbedder(model_name="nomic-embed-text")
    
    synthesizer = Synthesizer(model=judge)
    doc_path = os.path.join("docs", "knowledge_base.txt")

    context_config = ContextConstructionConfig(embedder=embedder,critic_model=judge)

    #goldens = synthesizer.generate_goldens_from_docs(document_paths=[doc_path])
    goldens =synthesizer.generate_goldens_from_docs(document_paths=[doc_path] ,context_construction_config=context_config )

    output_path = os.path.join("data", "golden_dataset.json") 
    synthesizer.save_as( file_type="json", directory="data" ) 
    print(f"nSuccessfully generated {len(goldens)} golden test cases saved in 'data/'!")


if __name__ == "__main__": 
    generate_golden_dataset()