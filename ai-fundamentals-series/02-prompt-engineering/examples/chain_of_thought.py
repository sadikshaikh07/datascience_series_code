"""
Chain-of-Thought (CoT) Prompting Examples
Blog 2: Prompt Engineering Fundamentals - Section 2.3

Chain-of-Thought prompting asks the AI to show its reasoning process
step-by-step before giving the final answer. This improves accuracy
for complex reasoning tasks.

Use cases: Math problems, logical reasoning, multi-step analysis, debugging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.llm_providers.base_llm import get_default_provider, LLMResponse
from typing import List, Dict, Tuple


class ChainOfThoughtPrompter:
    """
    Chain-of-Thought prompting techniques and examples.
    Demonstrates various CoT patterns for complex reasoning tasks.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.technique_name = "Chain-of-Thought Prompting"
    
    def basic_cot_math(self, problem: str) -> LLMResponse:
        """Basic Chain-of-Thought for math problems."""
        prompt = f"""Solve this math problem step by step:

Problem: {problem}

Let me work through this step by step:"""
        
        return self.provider.generate(prompt)
    
    def few_shot_cot_math(self, examples: List[Tuple[str, str]], problem: str) -> LLMResponse:
        """Few-shot Chain-of-Thought for math problems."""
        prompt = "Solve math problems by showing your reasoning step by step:\n\n"
        
        # Add examples with step-by-step reasoning
        for prob, solution in examples:
            prompt += f"Problem: {prob}\n{solution}\n\n"
        
        prompt += f"Problem: {problem}\nLet me work through this step by step:"
        
        return self.provider.generate(prompt)
    
    def cot_logical_reasoning(self, premise: str, question: str) -> LLMResponse:
        """Chain-of-Thought for logical reasoning problems."""
        prompt = f"""Given the following information, answer the question using step-by-step logical reasoning:

Information: {premise}

Question: {question}

Let me think through this step by step:
1. First, I'll identify the key facts
2. Then, I'll apply logical rules
3. Finally, I'll reach a conclusion

Step 1 - Key Facts:"""
        
        return self.provider.generate(prompt)
    
    def cot_word_problem(self, problem: str) -> LLMResponse:
        """Chain-of-Thought for word problems."""
        prompt = f"""Solve this word problem by breaking it down step by step:

Problem: {problem}

Let me break this down:
1. What information am I given?
2. What am I trying to find?
3. What operations do I need?
4. Calculate step by step
5. Check if the answer makes sense

Step 1 - Given information:"""
        
        return self.provider.generate(prompt)
    
    def cot_code_debugging(self, code: str, error: str) -> LLMResponse:
        """Chain-of-Thought for debugging code."""
        prompt = f"""Debug this code by analyzing it step by step:

Code:
{code}

Error: {error}

Let me debug this systematically:
1. Understand what the code is supposed to do
2. Trace through the execution
3. Identify where the error occurs
4. Determine why it happens
5. Propose a fix

Step 1 - Code purpose:"""
        
        return self.provider.generate(prompt)
    
    def cot_reading_comprehension(self, passage: str, question: str) -> LLMResponse:
        """Chain-of-Thought for reading comprehension."""
        prompt = f"""Answer the question about this passage using step-by-step analysis:

Passage: {passage}

Question: {question}

Let me analyze this systematically:
1. Key information in the passage
2. What the question is asking
3. Which parts of the passage are relevant
4. How to connect the information
5. Final answer

Step 1 - Key information:"""
        
        return self.provider.generate(prompt)
    
    def cot_decision_making(self, scenario: str, options: List[str]) -> LLMResponse:
        """Chain-of-Thought for decision-making scenarios."""
        options_text = "\n".join([f"- {opt}" for opt in options])
        
        prompt = f"""Make a decision for this scenario using systematic analysis:

Scenario: {scenario}

Options:
{options_text}

Let me analyze this decision systematically:
1. Define the goal/criteria for success
2. Evaluate each option against the criteria
3. Consider pros and cons
4. Identify potential risks
5. Make the final decision

Step 1 - Success criteria:"""
        
        return self.provider.generate(prompt)
    
    def cot_scientific_reasoning(self, observation: str, question: str) -> LLMResponse:
        """Chain-of-Thought for scientific reasoning."""
        prompt = f"""Explain this scientific phenomenon using step-by-step reasoning:

Observation: {observation}

Question: {question}

Let me approach this scientifically:
1. What do we observe?
2. What scientific principles might apply?
3. Form a hypothesis
4. Test the hypothesis against the observation
5. Draw conclusions

Step 1 - Observations:"""
        
        return self.provider.generate(prompt)
    
    def cot_story_analysis(self, story: str, question: str) -> LLMResponse:
        """Chain-of-Thought for story/literature analysis."""
        prompt = f"""Analyze this story using systematic literary analysis:

Story: {story}

Question: {question}

Let me analyze this step by step:
1. Identify key elements (characters, setting, plot)
2. Look for themes and patterns
3. Analyze character motivations
4. Consider literary devices
5. Answer the question based on analysis

Step 1 - Key elements:"""
        
        return self.provider.generate(prompt)
    
    def auto_cot(self, problem: str, domain: str = "general") -> LLMResponse:
        """
        Automatic Chain-of-Thought that adapts reasoning steps to the domain.
        
        Args:
            problem: The problem to solve
            domain: Domain type (math, logic, science, literature, etc.)
        """
        domain_templates = {
            "math": "1. Identify given values\n2. Determine what to find\n3. Choose appropriate formula/method\n4. Calculate step by step\n5. Verify the answer",
            "logic": "1. List all given facts\n2. Identify logical relationships\n3. Apply logical rules\n4. Draw intermediate conclusions\n5. Reach final conclusion",
            "science": "1. Observe the phenomenon\n2. Identify relevant scientific principles\n3. Form hypothesis\n4. Test against observations\n5. Conclude",
            "literature": "1. Identify key literary elements\n2. Analyze themes and motifs\n3. Examine character development\n4. Consider context and symbolism\n5. Interpret meaning",
            "general": "1. Understand the problem\n2. Break into smaller parts\n3. Analyze each part\n4. Synthesize findings\n5. Draw conclusion"
        }
        
        template = domain_templates.get(domain, domain_templates["general"])
        
        prompt = f"""Solve this {domain} problem using systematic step-by-step reasoning:

Problem: {problem}

I'll approach this systematically:
{template}

Step 1:"""
        
        return self.provider.generate(prompt)


def demo_chain_of_thought_techniques():
    """Comprehensive demonstration of Chain-of-Thought prompting techniques."""
    print("=== Chain-of-Thought Prompting Techniques Demo ===\n")
    
    try:
        prompter = ChainOfThoughtPrompter()
        print(f"🤖 Using provider: {prompter.provider.provider_name}")
        print(f"📝 Technique: {prompter.technique_name}")
        print("-" * 60 + "\n")
        
        # 1. Basic CoT Math
        print("1. Basic Chain-of-Thought Math Problem")
        print("Task: Solve multi-step math with reasoning")
        
        math_problem = "A store has 150 apples. They sell 60 apples in the morning and 45 apples in the afternoon. How many apples are left?"
        response = prompter.basic_cot_math(math_problem)
        
        print(f"Problem: {math_problem}")
        print(f"CoT Solution:\n{response.content}")
        print()
        
        # 2. Few-Shot CoT Math
        print("2. Few-Shot Chain-of-Thought Math")
        print("Task: Show reasoning pattern through examples")
        
        cot_examples = [
            ("What is 15% of 240?",
             "Let me work through this step by step:\n1. Convert percentage to decimal: 15% = 0.15\n2. Multiply: 240 × 0.15 = 36\nAnswer: 36"),
            ("If a train travels 120 miles in 2 hours, what is its speed?",
             "Let me solve this step by step:\n1. Speed = Distance ÷ Time\n2. Speed = 120 miles ÷ 2 hours = 60 mph\nAnswer: 60 miles per hour")
        ]
        
        new_problem = "A rectangle has a length of 12 meters and width of 8 meters. What is its area and perimeter?"
        response = prompter.few_shot_cot_math(cot_examples, new_problem)
        
        print("Examples with step-by-step reasoning:")
        for prob, sol in cot_examples:
            print(f"  Problem: {prob}")
            print(f"  Solution: {sol[:50]}...")
            print()
        
        print(f"New problem: {new_problem}")
        print(f"CoT Solution:\n{response.content}")
        print()
        
        # 3. CoT Logical Reasoning
        print("3. Chain-of-Thought Logical Reasoning")
        print("Task: Complex logical deduction with steps")
        
        premise = "All doctors are educated. Some educated people are wealthy. John is a doctor."
        question = "Can we conclude that John is wealthy?"
        
        response = prompter.cot_logical_reasoning(premise, question)
        
        print(f"Premise: {premise}")
        print(f"Question: {question}")
        print(f"Logical reasoning:\n{response.content}")
        print()
        
        # 4. CoT Word Problem
        print("4. Chain-of-Thought Word Problem")
        print("Task: Break down complex word problem")
        
        word_problem = """Sarah has $500. She spends 1/4 of her money on groceries, 
        then spends $80 on gas, and finally spends half of what remains on clothes. 
        How much money does she have left?"""
        
        response = prompter.cot_word_problem(word_problem)
        
        print(f"Word problem: {word_problem}")
        print(f"Step-by-step solution:\n{response.content}")
        print()
        
        # 5. CoT Code Debugging
        print("5. Chain-of-Thought Code Debugging")
        print("Task: Systematic debugging process")
        
        buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    average = total / len(numbers)
    return average

# Error occurs when calling:
result = calculate_average([])
"""
        error_msg = "ZeroDivisionError: division by zero"
        
        response = prompter.cot_code_debugging(buggy_code, error_msg)
        
        print("Buggy code and error:")
        print(buggy_code)
        print(f"Error: {error_msg}")
        print(f"Debug analysis:\n{response.content}")
        print()
        
        # 6. CoT Reading Comprehension
        print("6. Chain-of-Thought Reading Comprehension")
        print("Task: Systematic text analysis")
        
        passage = """The industrial revolution began in Britain during the late 18th century. 
        It was characterized by the shift from manual labor to mechanized production. 
        Steam engines powered new machinery, leading to increased productivity. 
        However, this also resulted in poor working conditions and environmental pollution."""
        
        question = "What were both the positive and negative effects of the industrial revolution?"
        
        response = prompter.cot_reading_comprehension(passage, question)
        
        print(f"Passage: {passage}")
        print(f"Question: {question}")
        print(f"Analysis:\n{response.content}")
        print()
        
        # 7. CoT Decision Making
        print("7. Chain-of-Thought Decision Making")
        print("Task: Systematic decision analysis")
        
        scenario = "You have a job offer with higher salary but longer commute, and another with lower salary but remote work option."
        options = [
            "Take the higher salary job with commute",
            "Take the lower salary remote job", 
            "Negotiate remote work with the higher salary job",
            "Keep looking for other opportunities"
        ]
        
        response = prompter.cot_decision_making(scenario, options)
        
        print(f"Scenario: {scenario}")
        print("Options:", options)
        print(f"Decision analysis:\n{response.content}")
        print()
        
        # 8. CoT Scientific Reasoning
        print("8. Chain-of-Thought Scientific Reasoning")
        print("Task: Scientific hypothesis and testing")
        
        observation = "Plants near the window grow taller than plants in the center of the room"
        question = "Why do plants near the window grow better?"
        
        response = prompter.cot_scientific_reasoning(observation, question)
        
        print(f"Observation: {observation}")
        print(f"Question: {question}")
        print(f"Scientific reasoning:\n{response.content}")
        print()
        
        # 9. Auto-CoT for Different Domains
        print("9. Automatic Chain-of-Thought (Domain-Adaptive)")
        print("Task: Adapt reasoning style to domain")
        
        domains_problems = [
            ("math", "If 3x + 7 = 22, what is the value of x?"),
            ("science", "Why does ice float on water?"),
            ("literature", "What is the significance of the green light in The Great Gatsby?")
        ]
        
        for domain, problem in domains_problems:
            print(f"Domain: {domain.title()}")
            print(f"Problem: {problem}")
            
            response = prompter.auto_cot(problem, domain)
            print(f"Reasoning approach:\n{response.content[:200]}...")
            print("-" * 40)
        
        print("\n" + "=" * 70)
        print("🎯 Chain-of-Thought Prompting Key Insights:")
        print("   ✅ Dramatically improves complex reasoning accuracy")
        print("   ✅ Makes AI reasoning transparent and verifiable")
        print("   ✅ Great for math, logic, and multi-step problems")
        print("   ✅ Helps identify and correct reasoning errors")
        print("   ⚠️  Uses more tokens (higher cost)")
        print("   ⚠️  Can be verbose for simple problems")
        print("   💡 Most effective for problems requiring >2 steps")
        print("   💡 Few-shot CoT often better than zero-shot CoT")
        
    except Exception as e:
        print(f"❌ Error during CoT demo: {e}")
        print("💡 Make sure your LLM provider is properly configured in .env")


def compare_with_without_cot():
    """Compare problem solving with and without Chain-of-Thought."""
    print("\n=== With vs Without Chain-of-Thought Comparison ===\n")
    
    prompter = ChainOfThoughtPrompter()
    
    # Complex reasoning problem
    problem = """A classroom has 32 students. If 3/8 of the students are boys, 
    and 60% of the boys play sports, how many boys in the class play sports?"""
    
    # Without CoT
    print("WITHOUT Chain-of-Thought:")
    simple_prompt = f"Solve: {problem}"
    simple_response = prompter.provider.generate(simple_prompt)
    print(f"Problem: {problem}")
    print(f"Answer: {simple_response.content.strip()}")
    print()
    
    # With CoT
    print("WITH Chain-of-Thought:")
    cot_response = prompter.basic_cot_math(problem)
    print(f"Problem: {problem}")
    print(f"Step-by-step solution: {cot_response.content}")
    print()
    
    print("📊 Comparison Results:")
    print("   Without CoT: Direct answer (may skip steps or make errors)")
    print("   With CoT: Shows reasoning process (more reliable and verifiable)")
    print("   Winner: CoT for complex multi-step problems")


if __name__ == "__main__":
    demo_chain_of_thought_techniques()
    compare_with_without_cot()