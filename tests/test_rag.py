import json # as the goldens got generated in json format 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric,ContextualPrecisionMetric,ContextualRecallMetric,FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from src.evaluator import LocalOllamaJudge
from src.rag import get_rag_response
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)



#  Loading Goldens in json  


dataset_path = os.path.join("data", "20260918_064344.json") 
with open(dataset_path, "r", encoding="utf-8") as f: 
    golden_data = json.load(f)


# Pytest Test Case Parameterized over Golden Dataset  


@pytest.mark.parametrize("item", golden_data) 

def test_rag_pipeline(item): 
    user_input = item["input"] 
    expected_output = item.get("expected_output", "") 

    # retriving the details
    rag_result = get_rag_response(user_input) 
    actual_output = rag_result["answer"] 
    retrieved_contexts = rag_result["retrieved_contexts"]

    # Creating DeepEval test case 
    test_case = LLMTestCase( input=user_input, actual_output=actual_output, expected_output=expected_output, retrieval_context=retrieved_contexts, )

    # Initialize Local Judge 
    judge = LocalOllamaJudge(model_name="qwen2.5:7b")

    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge) 
    answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge) 
    # hence we are getting the time out error we are calculating 2 metrics at a time 
    contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=judge) 
    contextual_recall = ContextualRecallMetric(threshold=0.7, model=judge) 


    print(f"n--- Evaluation Results for: '{user_input}' ---")
    print(f"Faithfulness: {faithfulness.score} (Reason: {faithfulness.reason})")
    print(f"Answer Relevancy: {answer_relevancy.score} (Reason: {answer_relevancy.reason})"    )

    
    assert_test( test_case, [faithfulness,answer_relevancy, contextual_precision, contextual_recall], ) # , answer_relevancy, contextual_precision, contextual_recall add this once you are measuring 

   

