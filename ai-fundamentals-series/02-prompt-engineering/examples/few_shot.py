"""
Few-Shot Prompting Examples
Blog 2: Prompt Engineering Fundamentals - Section 2.2

Few-shot prompting provides a few examples to show the AI the pattern
you want it to follow. This is more effective than zero-shot for
specific formats, styles, or complex tasks.

Use cases: When you need consistent formatting, specific style, or complex reasoning patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.llm_providers.base_llm import get_default_provider, LLMResponse
from typing import List, Dict, Tuple


class FewShotPrompter:
    """
    Few-shot prompting techniques and examples.
    Demonstrates various few-shot patterns and best practices.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.technique_name = "Few-Shot Prompting"
    
    def few_shot_classification(self, examples: List[Tuple[str, str]], new_input: str) -> LLMResponse:
        """
        Few-shot classification with examples.
        
        Args:
            examples: List of (input, output) example tuples
            new_input: New input to classify
            
        Returns:
            LLMResponse: Classification result
        """
        prompt = "Classify the following text based on these examples:\n\n"
        
        # Add examples
        for input_text, output_label in examples:
            prompt += f"Input: {input_text}\nOutput: {output_label}\n\n"
        
        # Add the new input
        prompt += f"Input: {new_input}\nOutput:"
        
        return self.provider.generate(prompt)
    
    def few_shot_translation(self, examples: List[Tuple[str, str]], new_input: str, 
                           source_lang: str, target_lang: str) -> LLMResponse:
        """Few-shot translation with style examples."""
        prompt = f"Translate from {source_lang} to {target_lang} following these examples:\n\n"
        
        for source, target in examples:
            prompt += f"{source_lang}: {source}\n{target_lang}: {target}\n\n"
        
        prompt += f"{source_lang}: {new_input}\n{target_lang}:"
        
        return self.provider.generate(prompt)
    
    def few_shot_format_conversion(self, examples: List[Tuple[str, str]], new_input: str,
                                 task_description: str) -> LLMResponse:
        """Few-shot format conversion (e.g., text to structured data)."""
        prompt = f"{task_description}\n\nExamples:\n\n"
        
        for input_example, output_example in examples:
            prompt += f"Input: {input_example}\nOutput: {output_example}\n\n"
        
        prompt += f"Input: {new_input}\nOutput:"
        
        return self.provider.generate(prompt)
    
    def few_shot_creative_writing(self, examples: List[Tuple[str, str]], new_prompt: str,
                                style_name: str) -> LLMResponse:
        """Few-shot creative writing with style examples."""
        prompt = f"Write in the {style_name} style following these examples:\n\n"
        
        for prompt_example, response_example in examples:
            prompt += f"Prompt: {prompt_example}\nResponse: {response_example}\n\n"
        
        prompt += f"Prompt: {new_prompt}\nResponse:"
        
        return self.provider.generate(prompt)
    
    def few_shot_code_generation(self, examples: List[Tuple[str, str]], new_task: str) -> LLMResponse:
        """Few-shot code generation with pattern examples."""
        prompt = "Generate code following these examples:\n\n"
        
        for task_desc, code_example in examples:
            prompt += f"Task: {task_desc}\nCode:\n{code_example}\n\n"
        
        prompt += f"Task: {new_task}\nCode:"
        
        return self.provider.generate(prompt)
    
    def few_shot_qa_format(self, examples: List[Tuple[str, str, str]], 
                          context: str, question: str) -> LLMResponse:
        """Few-shot question answering with specific answer format."""
        prompt = "Answer questions based on context using this format:\n\n"
        
        for ctx, q, a in examples:
            prompt += f"Context: {ctx}\nQuestion: {q}\nAnswer: {a}\n\n"
        
        prompt += f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        return self.provider.generate(prompt)
    
    def few_shot_sentiment_with_reasoning(self, examples: List[Tuple[str, str, str]], 
                                        new_text: str) -> LLMResponse:
        """Few-shot sentiment analysis with reasoning explanation."""
        prompt = "Analyze sentiment and provide reasoning:\n\n"
        
        for text, sentiment, reasoning in examples:
            prompt += f"Text: {text}\nSentiment: {sentiment}\nReasoning: {reasoning}\n\n"
        
        prompt += f"Text: {new_text}\nSentiment:"
        
        return self.provider.generate(prompt)
    
    def adaptive_few_shot(self, examples: List[Tuple[str, str]], new_input: str,
                         difficulty_level: str = "medium") -> LLMResponse:
        """
        Adaptive few-shot that adjusts number of examples based on complexity.
        
        Args:
            examples: Available examples
            new_input: Input to process
            difficulty_level: "easy", "medium", "hard"
        """
        # Adjust number of examples based on difficulty
        num_examples = {"easy": 1, "medium": 3, "hard": 5}.get(difficulty_level, 3)
        selected_examples = examples[:num_examples]
        
        prompt = f"Follow the pattern shown in these examples (difficulty: {difficulty_level}):\n\n"
        
        for inp, out in selected_examples:
            prompt += f"Input: {inp}\nOutput: {out}\n\n"
        
        prompt += f"Input: {new_input}\nOutput:"
        
        return self.provider.generate(prompt)


def demo_few_shot_techniques():
    """Comprehensive demonstration of few-shot prompting techniques."""
    print("=== Few-Shot Prompting Techniques Demo ===\n")
    
    try:
        prompter = FewShotPrompter()
        print(f"🤖 Using provider: {prompter.provider.provider_name}")
        print(f"📝 Technique: {prompter.technique_name}")
        print("-" * 60 + "\n")
        
        # 1. Few-Shot Classification
        print("1. Few-Shot Text Classification")
        print("Task: Classify customer feedback with examples")
        
        classification_examples = [
            ("The product arrived quickly and works perfectly!", "Positive"),
            ("Terrible quality, broke after one day.", "Negative"),
            ("It's okay, nothing special but does the job.", "Neutral"),
            ("Amazing customer service, highly recommend!", "Positive"),
            ("Worst purchase ever, demanding a refund.", "Negative")
        ]
        
        new_feedback = "The delivery was fast but the item had some minor scratches."
        response = prompter.few_shot_classification(classification_examples, new_feedback)
        
        print("Training examples:")
        for inp, out in classification_examples[:3]:  # Show first 3
            print(f"  '{inp}' → {out}")
        
        print(f"\nNew input: '{new_feedback}'")
        print(f"Prediction: {response.content.strip()}")
        print()
        
        # 2. Few-Shot Translation with Style
        print("2. Few-Shot Translation with Style")
        print("Task: Formal vs casual translation style")
        
        formal_examples = [
            ("Hello, how are you?", "Buenos días, ¿cómo está usted?"),
            ("Thank you very much", "Muchísimas gracias"),
            ("I would like to help", "Me gustaría ayudarle")
        ]
        
        response = prompter.few_shot_translation(
            formal_examples, 
            "Good morning, I hope you are well", 
            "English", 
            "Spanish"
        )
        
        print("Formal style examples:")
        for eng, spa in formal_examples:
            print(f"  '{eng}' → '{spa}'")
        
        print(f"\nNew input: 'Good morning, I hope you are well'")
        print(f"Translation: {response.content.strip()}")
        print()
        
        # 3. Few-Shot Format Conversion
        print("3. Few-Shot Format Conversion")
        print("Task: Convert text to structured JSON format")
        
        format_examples = [
            ("John Smith is 30 years old and works as an engineer", 
             '{"name": "John Smith", "age": 30, "occupation": "engineer"}'),
            ("Sarah Johnson, 25, teacher at local school",
             '{"name": "Sarah Johnson", "age": 25, "occupation": "teacher"}'),
            ("Mike Brown, age 45, doctor",
             '{"name": "Mike Brown", "age": 45, "occupation": "doctor"}')
        ]
        
        new_text = "Alice Wilson is a 28-year-old software developer"
        response = prompter.few_shot_format_conversion(
            format_examples, 
            new_text,
            "Convert person descriptions to JSON format:"
        )
        
        print("Format conversion examples:")
        for inp, out in format_examples[:2]:
            print(f"  Input: '{inp}'")
            print(f"  Output: {out}")
            print()
        
        print(f"New input: '{new_text}'")
        print(f"JSON output: {response.content.strip()}")
        print()
        
        # 4. Few-Shot Creative Writing
        print("4. Few-Shot Creative Writing Style")
        print("Task: Write in a specific poetic style")
        
        haiku_examples = [
            ("Write about spring", "Cherry blossoms bloom\nGentle breeze carries petals\nNature awakens"),
            ("Write about night", "Stars pierce darkness deep\nSilent moon watches over\nDreams take gentle flight"),
            ("Write about rain", "Droplets kiss the earth\nRhythmic dance on window pane\nLife drinks deeply now")
        ]
        
        response = prompter.few_shot_creative_writing(
            haiku_examples, 
            "Write about friendship", 
            "haiku"
        )
        
        print("Haiku style examples:")
        for prompt, haiku in haiku_examples[:2]:
            print(f"  Prompt: '{prompt}'")
            print(f"  Haiku: {haiku}")
            print()
        
        print("New prompt: 'Write about friendship'")
        print(f"Generated haiku: {response.content.strip()}")
        print()
        
        # 5. Few-Shot Code Generation
        print("5. Few-Shot Code Generation")
        print("Task: Generate Python functions with specific patterns")
        
        code_examples = [
            ("Create a function to add two numbers",
             "def add_numbers(a, b):\n    \"\"\"Add two numbers and return result.\"\"\"\n    return a + b"),
            ("Create a function to find maximum of two numbers",
             "def find_max(a, b):\n    \"\"\"Find maximum of two numbers.\"\"\"\n    return a if a > b else b"),
            ("Create a function to check if number is even",
             "def is_even(n):\n    \"\"\"Check if a number is even.\"\"\"\n    return n % 2 == 0")
        ]
        
        response = prompter.few_shot_code_generation(
            code_examples,
            "Create a function to calculate the square of a number"
        )
        
        print("Code generation examples:")
        for task, code in code_examples[:2]:
            print(f"  Task: {task}")
            print(f"  Code: {code.split('def')[1].split('return')[0]}...")
            print()
        
        print("New task: 'Create a function to calculate the square of a number'")
        print(f"Generated code:\n{response.content}")
        print()
        
        # 6. Few-Shot Q&A with Format
        print("6. Few-Shot Q&A with Specific Format")
        print("Task: Answer questions with confidence levels")
        
        qa_examples = [
            ("The cat sat on the mat", "What sat on the mat?", "The cat (Confidence: High)"),
            ("Paris is the capital of France", "What is the capital of France?", "Paris (Confidence: High)"),
            ("The story mentions a mysterious figure", "Who was the mysterious figure?", "Not specified in the text (Confidence: High)")
        ]
        
        response = prompter.few_shot_qa_format(
            qa_examples,
            "The blue car drove slowly down the winding mountain road",
            "What color was the car?"
        )
        
        print("Q&A format examples:")
        for ctx, q, a in qa_examples[:2]:
            print(f"  Context: '{ctx}'")
            print(f"  Question: {q}")
            print(f"  Answer: {a}")
            print()
        
        print("New context: 'The blue car drove slowly down the winding mountain road'")
        print("Question: 'What color was the car?'")
        print(f"Answer: {response.content.strip()}")
        print()
        
        # 7. Few-Shot Sentiment with Reasoning
        print("7. Few-Shot Sentiment Analysis with Reasoning")
        print("Task: Provide sentiment + explanation")
        
        sentiment_examples = [
            ("I love this product!", "Positive", "Uses enthusiastic language with 'love' and exclamation"),
            ("It's completely broken", "Negative", "Describes a defective state with absolute language 'completely'"),
            ("The item works as expected", "Neutral", "Factual statement without emotional indicators")
        ]
        
        response = prompter.few_shot_sentiment_with_reasoning(
            sentiment_examples,
            "This exceeded my expectations in every way!"
        )
        
        print("Sentiment analysis examples:")
        for text, sentiment, reasoning in sentiment_examples:
            print(f"  Text: '{text}' → {sentiment}")
            print(f"       Reasoning: {reasoning}")
            print()
        
        print("New text: 'This exceeded my expectations in every way!'")
        print(f"Analysis: {response.content.strip()}")
        print()
        
        # 8. Adaptive Few-Shot
        print("8. Adaptive Few-Shot (Difficulty-Based)")
        print("Task: Adjust examples based on task complexity")
        
        math_examples = [
            ("What is 2 + 2?", "4"),
            ("What is 15 × 7?", "105"), 
            ("What is the square root of 144?", "12"),
            ("What is 23% of 200?", "46"),
            ("Solve: 2x + 5 = 13", "x = 4")
        ]
        
        difficulties = ["easy", "medium", "hard"]
        new_question = "What is 8 × 9?"
        
        for difficulty in difficulties:
            response = prompter.adaptive_few_shot(math_examples, new_question, difficulty)
            print(f"Difficulty: {difficulty}")
            print(f"Question: '{new_question}'")
            print(f"Answer: {response.content.strip()}")
            print()
        
        print("=" * 70)
        print("🎯 Few-Shot Prompting Key Insights:")
        print("   ✅ More reliable than zero-shot for specific formats")
        print("   ✅ Excellent for consistent output structure")
        print("   ✅ Great for teaching specific reasoning patterns")
        print("   ✅ Can capture subtle style and tone preferences")
        print("   ⚠️  Requires careful example selection")
        print("   ⚠️  More tokens used (higher cost)")
        print("   💡 Quality of examples matters more than quantity")
        print("   💡 3-5 examples often optimal for most tasks")
        
    except Exception as e:
        print(f"❌ Error during few-shot demo: {e}")
        print("💡 Make sure your LLM provider is properly configured in .env")


def compare_zero_vs_few_shot():
    """Compare zero-shot vs few-shot performance on the same task."""
    print("\n=== Zero-Shot vs Few-Shot Comparison ===\n")
    
    prompter = FewShotPrompter()
    
    # Task: Convert casual text to professional tone
    test_input = "hey can u help me with this problem? its kinda urgent lol"
    
    # Zero-shot approach
    print("Zero-Shot Approach:")
    zero_shot_prompt = "Convert this casual text to professional tone: " + test_input
    zero_response = prompter.provider.generate(zero_shot_prompt)
    print(f"Input: {test_input}")
    print(f"Output: {zero_response.content.strip()}")
    print()
    
    # Few-shot approach
    print("Few-Shot Approach:")
    examples = [
        ("hey whats up?", "Hello, how are you?"),
        ("can u call me back asap", "Could you please call me back at your earliest convenience?"),
        ("this is really important!!!", "This matter is of high importance.")
    ]
    
    few_shot_response = prompter.few_shot_format_conversion(
        examples, 
        test_input,
        "Convert casual text to professional tone:"
    )
    
    print("Examples provided:")
    for casual, professional in examples:
        print(f"  '{casual}' → '{professional}'")
    
    print(f"\nInput: {test_input}")
    print(f"Output: {few_shot_response.content.strip()}")
    print()
    
    print("📊 Comparison Results:")
    print("   Zero-shot: May vary in formality level")
    print("   Few-shot: Consistent professional tone matching examples")
    print("   Winner: Few-shot for consistency and specific style requirements")


if __name__ == "__main__":
    demo_few_shot_techniques()
    compare_zero_vs_few_shot()