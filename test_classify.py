"""Offline tests; no API key needed. Run: python -m unittest -v"""
import unittest
from unittest.mock import patch
import main

class TestClassify(unittest.TestCase):
    def test_keywords(self):
        tests = {"帮我写一副春节对联": "couplet", "讲个笑话": "joke", "推荐北京旅游攻略": "travel"}
        for query, expected in tests.items():
            with self.subTest(query=query), patch.object(main, "llm_classify", side_effect=AssertionError("LLM should not run")):
                result = main.classify({"user_input": query})
                self.assertEqual(result, {"category": expected, "classify_method": "keyword"})

    def test_model_fallback(self):
        with patch.object(main, "llm_classify", return_value="joke") as mock:
            result = main.classify({"user_input": "说点有趣的让我开心"})
            self.assertEqual(result, {"category": "joke", "classify_method": "llm"})
            mock.assert_called_once()

    def test_ambiguous_keywords(self):
        with patch.object(main, "llm_classify", return_value="travel"):
            result = main.classify({"user_input": "旅行途中讲个笑话"})
            self.assertEqual(result["classify_method"], "llm")

    def test_model_error(self):
        with patch.object(main, "llm_classify", side_effect=RuntimeError("network")):
            result = main.classify({"user_input": "你好"})
            self.assertEqual(result["category"], "other")
            self.assertEqual(result["classify_method"], "fallback_error")

    def test_graph_routing(self):
        with patch.object(main, "llm_text", return_value="测试回复"):
            result = main.build_graph().invoke({"user_input": "讲个笑话"})
            self.assertEqual(result["category"], "joke")
            self.assertEqual(result["answer"], "测试回复")

if __name__ == "__main__":
    unittest.main()
