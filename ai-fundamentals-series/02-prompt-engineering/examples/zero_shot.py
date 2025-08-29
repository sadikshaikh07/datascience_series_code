"""
Zero-Shot Prompting Examples
Blog 2: Prompt Engineering Fundamentals - Section 2.1

Zero-shot prompting means giving the AI a task without any examples.
Just the task description and expecting it to perform correctly.

Use cases: Simple, well-known tasks that LLMs can handle directly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.llm_providers.base_llm import get_default_provider, LLMResponse
from typing import List, Dict


class ZeroShotPrompter:
    """
    Zero-shot prompting techniques and examples.
    Demonstrates various zero-shot patterns and best practices.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.technique_name = "Zero-Shot Prompting"
    
    def basic_zero_shot(self, task: str, input_text: str) -> LLMResponse:
        """
        Basic zero-shot prompt: Direct task instruction.
        
        Args:
            task: Description of what to do
            input_text: The input to process
            
        Returns:
            LLMResponse: AI response
        """
        prompt = f"{task}\n\nInput: {input_text}\nOutput:"
        
        return self.provider.generate(prompt)
    
    def translation_zero_shot(self, text: str, target_language: str) -> LLMResponse:
        """Zero-shot translation task."""
        prompt = f"Translate the following text to {target_language}: '{text}'"
        
        return self.provider.generate(prompt)
    
    def classification_zero_shot(self, text: str, categories: List[str]) -> LLMResponse:
        """
        Zero-shot text classification.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            
        Returns:
            LLMResponse: Classification result
        """
        categories_str = ", ".join(categories)
        
        prompt = f"""Classify the following text into one of these categories: {categories_str}

Text: "{text}"

Category:"""
        
        return self.provider.generate(prompt)
    
    def sentiment_analysis_zero_shot(self, text: str) -> LLMResponse:
        """Zero-shot sentiment analysis."""
        prompt = f"""Analyze the sentiment of the following text. Respond with only one word: Positive, Negative, or Neutral.

Text: "{text}"

Sentiment:"""
        
        return self.provider.generate(prompt)
    
    def summarization_zero_shot(self, text: str, max_sentences: int = 3) -> LLMResponse:
        """Zero-shot text summarization."""
        prompt = f"""Summarize the following text in {max_sentences} sentences or less:

{text}

Summary:"""
        
        return self.provider.generate(prompt)
    
    def question_answering_zero_shot(self, context: str, question: str) -> LLMResponse:
        """Zero-shot question answering."""
        prompt = f"""Answer the following question based on the given context. If the answer cannot be found in the context, respond with "Information not available."

Context: {context}

Question: {question}

Answer:"""
        
        return self.provider.generate(prompt)
    
    def creative_writing_zero_shot(self, topic: str, style: str) -> LLMResponse:
        """Zero-shot creative writing."""
        prompt = f"Write a {style} about {topic}."
        
        return self.provider.generate(prompt)
    
    def code_generation_zero_shot(self, task: str, language: str) -> LLMResponse:
        """Zero-shot code generation."""
        prompt = f"""Write a {language} function that {task}. Include comments explaining the code.

```{language.lower()}"""
        
        return self.provider.generate(prompt)
    
    def data_extraction_zero_shot(self, text: str, fields: List[str]) -> LLMResponse:
        """Zero-shot structured data extraction."""
        fields_str = ", ".join(fields)
        
        prompt = f"""Extract the following information from the text: {fields_str}

Text: "{text}"

Extracted Information:"""
        
        return self.provider.generate(prompt)
    
    def logical_reasoning_zero_shot(self, premise: str, question: str) -> LLMResponse:
        """Zero-shot logical reasoning."""
        prompt = f"""Given the following information, answer the question using logical reasoning:

Information: {premise}

Question: {question}

Answer:"""
        
        return self.provider.generate(prompt)


def demo_zero_shot_techniques():
    """Comprehensive demonstration of zero-shot prompting techniques."""
    print("=== Zero-Shot Prompting Techniques Demo ===\n")
    
    try:
        prompter = ZeroShotPrompter()
        print(f"🤖 Using provider: {prompter.provider.provider_name}")
        print(f"📝 Technique: {prompter.technique_name}")
        print("-" * 60 + "\n")
        
        # 1. Basic Zero-Shot
        print("1. Basic Zero-Shot Example")
        print("Task: Simple instruction following")
        
        response = prompter.basic_zero_shot(
            "Convert this text to uppercase", 
            "hello world this is a test"
        )
        print(f"Result: {response.content}")
        print()
        
        # 2. Translation
        print("2. Zero-Shot Translation")
        print("Task: Language translation without examples")
        
        response = prompter.translation_zero_shot(
            "Good morning, how are you today?", 
            "Spanish"
        )
        print(f"Translation: {response.content}")
        print()
        
        # 3. Classification
        print("3. Zero-Shot Text Classification")
        print("Task: Categorize text without training examples")
        
        categories = ["Technology", "Sports", "Politics", "Entertainment", "Science"]
        text = "Scientists have discovered a new exoplanet that could potentially support life."
        
        response = prompter.classification_zero_shot(text, categories)
        print(f"Text: {text}")
        print(f"Categories: {categories}")
        print(f"Classification: {response.content}")
        print()
        
        # 4. Sentiment Analysis
        print("4. Zero-Shot Sentiment Analysis")
        print("Task: Determine emotional tone without examples")
        
        texts = [
            "I absolutely love this new restaurant! The food was amazing!",
            "The service was terrible and the food was cold.",
            "The movie was okay, nothing special but not bad either."
        ]
        
        for text in texts:
            response = prompter.sentiment_analysis_zero_shot(text)
            print(f"Text: '{text}'")
            print(f"Sentiment: {response.content.strip()}")
            print()
        
        # 5. Summarization
        print("5. Zero-Shot Text Summarization")
        print("Task: Condense long text without example summaries")
        
        long_text = """
        Artificial Intelligence (AI) has become increasingly important in modern technology. 
        Machine learning algorithms can analyze vast amounts of data to identify patterns and make predictions. 
        Deep learning, a subset of machine learning, uses neural networks with multiple layers to process complex data.
        Natural language processing allows computers to understand and generate human language.
        Computer vision enables machines to interpret and understand visual information.
        These technologies are being applied in various fields including healthcare, finance, transportation, and entertainment.
        However, there are also concerns about job displacement, privacy, and the ethical implications of AI systems.
        """
        
        response = prompter.summarization_zero_shot(long_text.strip(), max_sentences=2)
        print("Original text length:", len(long_text.split()))
        print(f"Summary: {response.content}")
        print()
        
        # 6. Question Answering
        print("6. Zero-Shot Question Answering")
        print("Task: Answer questions based on context without examples")
        
        context = """
        The Eiffel Tower was built between 1887 and 1889 for the 1889 World's Fair in Paris. 
        It was designed by Gustave Eiffel and stands 330 meters tall. The tower is made of iron 
        and weighs approximately 10,100 tons. It was initially criticized by artists and intellectuals
        but has become a global icon of France.
        """
        
        questions = [
            "Who designed the Eiffel Tower?",
            "When was it built?",
            "How tall is the Eiffel Tower?",
            "What is it made of?"
        ]
        
        for question in questions:
            response = prompter.question_answering_zero_shot(context, question)
            print(f"Q: {question}")
            print(f"A: {response.content.strip()}")
            print()
        
        # 7. Creative Writing
        print("7. Zero-Shot Creative Writing")
        print("Task: Generate creative content without examples")
        
        response = prompter.creative_writing_zero_shot("a robot learning to cook", "short story")
        print("Topic: A robot learning to cook")
        print("Style: Short story")
        print(f"Generated story: {response.content[:200]}...")
        print()
        
        # 8. Code Generation
        print("8. Zero-Shot Code Generation")
        print("Task: Generate code without examples")
        
        response = prompter.code_generation_zero_shot(
            "calculates the factorial of a number", 
            "Python"
        )
        print("Task: Calculate factorial")
        print("Language: Python")
        print(f"Generated code:\n{response.content}")
        print()
        
        # 9. Data Extraction
        print("9. Zero-Shot Data Extraction")
        print("Task: Extract structured information without examples")
        
        text = "John Smith, age 35, works as a software engineer at TechCorp Inc. He can be reached at john.smith@email.com or 555-123-4567."
        fields = ["Name", "Age", "Job Title", "Company", "Email", "Phone"]
        
        response = prompter.data_extraction_zero_shot(text, fields)
        print(f"Text: {text}")
        print(f"Fields to extract: {fields}")
        print(f"Extracted data: {response.content}")
        print()
        
        # 10. Logical Reasoning
        print("10. Zero-Shot Logical Reasoning")
        print("Task: Apply logic without reasoning examples")
        
        premise = "All birds can fly. Penguins are birds. Ostriches are birds."
        question = "Can all birds actually fly? Explain your reasoning."
        
        response = prompter.logical_reasoning_zero_shot(premise, question)
        print(f"Premise: {premise}")
        print(f"Question: {question}")
        print(f"Reasoning: {response.content}")
        
        print("\n" + "="*70)
        print("🎯 Zero-Shot Prompting Key Insights:")
        print("   ✅ Works well for common, well-defined tasks")
        print("   ✅ Simple and straightforward to implement")
        print("   ✅ Good starting point for most prompting needs")
        print("   ⚠️  May struggle with very specific or complex tasks")
        print("   ⚠️  Performance depends on task familiarity to the LLM")
        print("   💡 When zero-shot fails, consider few-shot prompting")
        
    except Exception as e:
        print(f"❌ Error during zero-shot demo: {e}")
        print("💡 Make sure your LLM provider is properly configured in .env")


def compare_zero_shot_variations():
    """Compare different ways to structure zero-shot prompts."""
    print("\n=== Zero-Shot Prompt Structure Comparison ===\n")
    
    prompter = ZeroShotPrompter()
    task_input = "The weather is beautiful today and I feel great!"
    
    # Different prompt structures for sentiment analysis
    variations = [
        # Basic structure
        f"What is the sentiment of this text: '{task_input}'",
        
        # More explicit structure
        f"Analyze the sentiment of the following text.\n\nText: '{task_input}'\n\nSentiment:",
        
        # With constraints
        f"Classify the sentiment of this text as Positive, Negative, or Neutral: '{task_input}'",
        
        # With reasoning request
        f"Analyze the sentiment of '{task_input}' and explain your reasoning in one sentence."
    ]
    
    for i, prompt in enumerate(variations, 1):
        print(f"Variation {i}:")
        print(f"Prompt: {prompt}")
        
        response = prompter.provider.generate(prompt)
        print(f"Response: {response.content.strip()}")
        print("-" * 40)


if __name__ == "__main__":
    demo_zero_shot_techniques()
    compare_zero_shot_variations()