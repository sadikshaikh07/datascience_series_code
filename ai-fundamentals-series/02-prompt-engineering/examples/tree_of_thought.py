"""
Tree-of-Thought (ToT) Prompting Examples
Blog 2: Prompt Engineering Fundamentals - Advanced Section

Tree-of-Thought prompting explores multiple reasoning paths simultaneously,
evaluating different approaches before converging on the best solution.
More sophisticated than Chain-of-Thought for complex problems.

Use cases: Complex problem-solving, creative tasks, strategic planning, optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.llm_providers.base_llm import get_default_provider, LLMResponse
from typing import List, Dict, Tuple, Any
import json


class TreeOfThoughtPrompter:
    """
    Tree-of-Thought prompting techniques and examples.
    Demonstrates multi-path reasoning and solution evaluation.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.technique_name = "Tree-of-Thought Prompting"
    
    def basic_tot_problem_solving(self, problem: str, num_approaches: int = 3) -> LLMResponse:
        """Basic Tree-of-Thought for problem-solving with multiple approaches."""
        prompt = f"""I need to solve this problem by exploring multiple approaches:

Problem: {problem}

Let me explore {num_approaches} different approaches and evaluate them:

Approach 1: [Describe first approach]
- Reasoning: 
- Pros:
- Cons:
- Likelihood of success:

Approach 2: [Describe second approach]
- Reasoning:
- Pros: 
- Cons:
- Likelihood of success:

Approach 3: [Describe third approach]
- Reasoning:
- Pros:
- Cons:
- Likelihood of success:

Evaluation and Selection:
Based on the analysis above, I recommend:"""
        
        return self.provider.generate(prompt)
    
    def tot_creative_writing(self, prompt_text: str, genre: str) -> LLMResponse:
        """Tree-of-Thought for creative writing with multiple plot directions."""
        prompt = f"""I want to write a {genre} story based on: "{prompt_text}"

Let me explore multiple narrative directions:

Direction 1: [Conventional approach]
- Plot outline:
- Character development:
- Conflict/tension:
- Estimated reader engagement:

Direction 2: [Unconventional/twist approach] 
- Plot outline:
- Character development:
- Conflict/tension:
- Estimated reader engagement:

Direction 3: [Experimental approach]
- Plot outline:
- Character development:
- Conflict/tension:
- Estimated reader engagement:

Best Direction Analysis:
Considering originality, engagement, and feasibility:"""
        
        return self.provider.generate(prompt)
    
    def tot_business_strategy(self, situation: str, goal: str) -> LLMResponse:
        """Tree-of-Thought for business strategy planning."""
        prompt = f"""Business Situation: {situation}

Goal: {goal}

Let me explore multiple strategic approaches:

Strategy A: Conservative Approach
- Key actions:
- Resource requirements:
- Timeline:
- Risk level:
- Expected ROI:

Strategy B: Aggressive Growth Approach
- Key actions:
- Resource requirements:
- Timeline:
- Risk level:
- Expected ROI:

Strategy C: Innovation-Focused Approach
- Key actions:
- Resource requirements:
- Timeline:
- Risk level:
- Expected ROI:

Strategic Recommendation:
After weighing all factors:"""
        
        return self.provider.generate(prompt)
    
    def tot_technical_architecture(self, requirements: str, constraints: str) -> LLMResponse:
        """Tree-of-Thought for technical system architecture."""
        prompt = f"""System Requirements: {requirements}

Constraints: {constraints}

Let me explore multiple architectural approaches:

Architecture 1: Monolithic Approach
- Components:
- Data flow:
- Scalability:
- Maintainability:
- Cost:

Architecture 2: Microservices Approach
- Components:
- Data flow:
- Scalability:
- Maintainability:
- Cost:

Architecture 3: Serverless/Event-Driven Approach
- Components:
- Data flow:
- Scalability:
- Maintainability:
- Cost:

Architecture Decision:
Given the requirements and constraints:"""
        
        return self.provider.generate(prompt)
    
    def tot_mathematical_proof(self, theorem: str) -> LLMResponse:
        """Tree-of-Thought for mathematical proof strategies."""
        prompt = f"""Theorem to prove: {theorem}

Let me explore multiple proof strategies:

Proof Strategy 1: Direct Proof
- Approach:
- Key steps:
- Difficulty level:
- Completeness:

Proof Strategy 2: Proof by Contradiction
- Approach:
- Key steps:
- Difficulty level:
- Completeness:

Proof Strategy 3: Mathematical Induction
- Approach:
- Key steps:
- Difficulty level:
- Completeness:

Best Proof Strategy:
Selecting the most suitable approach:"""
        
        return self.provider.generate(prompt)
    
    def tot_decision_tree(self, decision: str, factors: List[str]) -> LLMResponse:
        """Tree-of-Thought creating explicit decision trees."""
        factors_text = ", ".join(factors)
        
        prompt = f"""Decision to make: {decision}

Key factors to consider: {factors_text}

Let me build a decision tree exploring different paths:

Branch 1: If we prioritize {factors[0]}
- Immediate actions:
- Short-term outcomes:
- Long-term consequences:
- Success probability:

Branch 2: If we prioritize {factors[1] if len(factors) > 1 else 'alternative factor'}
- Immediate actions:
- Short-term outcomes:
- Long-term consequences:
- Success probability:

Branch 3: If we balance multiple factors
- Immediate actions:
- Short-term outcomes:
- Long-term consequences:
- Success probability:

Optimal Decision Path:
After analyzing all branches:"""
        
        return self.provider.generate(prompt)
    
    def tot_debugging_complex_system(self, system_description: str, problem: str) -> LLMResponse:
        """Tree-of-Thought for complex system debugging."""
        prompt = f"""System: {system_description}

Problem: {problem}

Let me explore multiple debugging approaches systematically:

Investigation Path 1: Hardware/Infrastructure Focus
- What to check:
- Expected findings:
- Diagnostic steps:
- Fix complexity:

Investigation Path 2: Software/Code Focus
- What to check:
- Expected findings:
- Diagnostic steps:
- Fix complexity:

Investigation Path 3: Data/Configuration Focus
- What to check:
- Expected findings:
- Diagnostic steps:
- Fix complexity:

Investigation Path 4: External Dependencies Focus
- What to check:
- Expected findings:
- Diagnostic steps:
- Fix complexity:

Debugging Strategy:
Based on symptom analysis and system knowledge:"""
        
        return self.provider.generate(prompt)
    
    def tot_research_methodology(self, research_question: str, field: str) -> LLMResponse:
        """Tree-of-Thought for research methodology design."""
        prompt = f"""Research Question: {research_question}

Field: {field}

Let me explore multiple research methodologies:

Methodology 1: Quantitative Approach
- Data collection method:
- Sample size/selection:
- Analysis techniques:
- Validity concerns:
- Timeline:

Methodology 2: Qualitative Approach  
- Data collection method:
- Sample size/selection:
- Analysis techniques:
- Validity concerns:
- Timeline:

Methodology 3: Mixed Methods Approach
- Data collection method:
- Sample size/selection:
- Analysis techniques:
- Validity concerns:
- Timeline:

Recommended Research Design:
Considering the research question and field constraints:"""
        
        return self.provider.generate(prompt)
    
    def tot_optimization_problem(self, problem: str, constraints: List[str]) -> LLMResponse:
        """Tree-of-Thought for optimization problems."""
        constraints_text = "\n".join([f"- {c}" for c in constraints])
        
        prompt = f"""Optimization Problem: {problem}

Constraints:
{constraints_text}

Let me explore multiple optimization approaches:

Approach 1: Greedy Algorithm
- Strategy:
- Implementation complexity:
- Time complexity:
- Solution quality:
- Pros/Cons:

Approach 2: Dynamic Programming
- Strategy:
- Implementation complexity:
- Time complexity:
- Solution quality:
- Pros/Cons:

Approach 3: Heuristic/Approximation
- Strategy:
- Implementation complexity:
- Time complexity:
- Solution quality:
- Pros/Cons:

Optimal Approach Selection:
Given the problem characteristics and constraints:"""
        
        return self.provider.generate(prompt)


def demo_tree_of_thought_techniques():
    """Comprehensive demonstration of Tree-of-Thought prompting techniques."""
    print("=== Tree-of-Thought Prompting Techniques Demo ===\n")
    
    try:
        prompter = TreeOfThoughtPrompter()
        print(f"🤖 Using provider: {prompter.provider.provider_name}")
        print(f"📝 Technique: {prompter.technique_name}")
        print("-" * 60 + "\n")
        
        # 1. Basic ToT Problem Solving
        print("1. Tree-of-Thought Problem Solving")
        print("Task: Explore multiple solution approaches")
        
        problem = "How can a small startup compete with established companies in a saturated market?"
        response = prompter.basic_tot_problem_solving(problem)
        
        print(f"Problem: {problem}")
        print(f"Multi-approach analysis:\n{response.content[:500]}...")
        print()
        
        # 2. ToT Creative Writing
        print("2. Tree-of-Thought Creative Writing")
        print("Task: Explore multiple narrative directions")
        
        creative_prompt = "A time traveler discovers they can only travel backwards, not forwards"
        response = prompter.tot_creative_writing(creative_prompt, "science fiction")
        
        print(f"Creative prompt: {creative_prompt}")
        print(f"Genre: Science Fiction")
        print(f"Narrative exploration:\n{response.content[:400]}...")
        print()
        
        # 3. ToT Business Strategy
        print("3. Tree-of-Thought Business Strategy")
        print("Task: Multiple strategic approaches")
        
        situation = "Our company's main product is being disrupted by new technology"
        goal = "Maintain market position and profitability over next 3 years"
        response = prompter.tot_business_strategy(situation, goal)
        
        print(f"Situation: {situation}")
        print(f"Goal: {goal}")
        print(f"Strategic analysis:\n{response.content[:400]}...")
        print()
        
        # 4. ToT Technical Architecture
        print("4. Tree-of-Thought Technical Architecture")
        print("Task: Compare architectural approaches")
        
        requirements = "Real-time data processing for 1M+ users, 99.9% uptime"
        constraints = "Limited budget, 6-month timeline, small team"
        response = prompter.tot_technical_architecture(requirements, constraints)
        
        print(f"Requirements: {requirements}")
        print(f"Constraints: {constraints}")
        print(f"Architecture analysis:\n{response.content[:400]}...")
        print()
        
        # 5. ToT Mathematical Proof
        print("5. Tree-of-Thought Mathematical Proof")
        print("Task: Explore multiple proof strategies")
        
        theorem = "For any integer n > 1, if n is prime, then n is odd or n = 2"
        response = prompter.tot_mathematical_proof(theorem)
        
        print(f"Theorem: {theorem}")
        print(f"Proof strategy analysis:\n{response.content[:400]}...")
        print()
        
        # 6. ToT Decision Tree
        print("6. Tree-of-Thought Decision Tree")
        print("Task: Systematic decision analysis")
        
        decision = "Should we launch the product now or wait 6 more months for additional features?"
        factors = ["Time to market", "Feature completeness", "Competition", "Budget"]
        response = prompter.tot_decision_tree(decision, factors)
        
        print(f"Decision: {decision}")
        print(f"Factors: {factors}")
        print(f"Decision tree analysis:\n{response.content[:400]}...")
        print()
        
        # 7. ToT Complex System Debugging
        print("7. Tree-of-Thought System Debugging")
        print("Task: Multi-path debugging investigation")
        
        system = "Distributed e-commerce platform with microservices"
        problem = "Random 5-second delays affecting 10% of checkout transactions"
        response = prompter.tot_debugging_complex_system(system, problem)
        
        print(f"System: {system}")
        print(f"Problem: {problem}")
        print(f"Debugging investigation:\n{response.content[:400]}...")
        print()
        
        # 8. ToT Research Methodology
        print("8. Tree-of-Thought Research Methodology")
        print("Task: Compare research approaches")
        
        research_q = "How does remote work affect employee productivity and job satisfaction?"
        field = "Organizational Psychology"
        response = prompter.tot_research_methodology(research_q, field)
        
        print(f"Research Question: {research_q}")
        print(f"Field: {field}")
        print(f"Methodology analysis:\n{response.content[:400]}...")
        print()
        
        # 9. ToT Optimization Problem
        print("9. Tree-of-Thought Optimization")
        print("Task: Compare optimization algorithms")
        
        opt_problem = "Schedule 100 tasks across 10 workers to minimize total completion time"
        constraints = ["Each task has different duration", "Workers have different skill levels", "Some tasks have dependencies", "Must complete within 8 hours"]
        response = prompter.tot_optimization_problem(opt_problem, constraints)
        
        print(f"Problem: {opt_problem}")
        print(f"Constraints: {len(constraints)} constraints")
        print(f"Algorithm analysis:\n{response.content[:400]}...")
        print()
        
        print("=" * 70)
        print("🎯 Tree-of-Thought Prompting Key Insights:")
        print("   ✅ Explores multiple solution paths simultaneously")
        print("   ✅ Excellent for complex, open-ended problems")
        print("   ✅ Reduces solution tunnel vision")
        print("   ✅ Provides comparative analysis of approaches")
        print("   ⚠️  More expensive (uses many tokens)")
        print("   ⚠️  Can be overwhelming for simple problems")
        print("   ⚠️  Requires good evaluation criteria")
        print("   💡 Best for creative, strategic, or optimization tasks")
        print("   💡 Combine with human expertise for best results")
        
    except Exception as e:
        print(f"❌ Error during ToT demo: {e}")
        print("💡 Make sure your LLM provider is properly configured in .env")


def compare_cot_vs_tot():
    """Compare Chain-of-Thought vs Tree-of-Thought approaches."""
    print("\n=== Chain-of-Thought vs Tree-of-Thought Comparison ===\n")
    
    prompter = TreeOfThoughtPrompter()
    
    problem = "Design a mobile app that helps people reduce food waste"
    
    # Chain-of-Thought approach
    print("CHAIN-OF-THOUGHT Approach:")
    cot_prompt = f"""Design a mobile app that helps people reduce food waste. 
    
    Let me work through this step by step:
    1. Identify the problem
    2. Define target users
    3. List key features
    4. Consider technical implementation
    5. Plan monetization strategy
    
    Step 1 - Problem identification:"""
    
    cot_response = prompter.provider.generate(cot_prompt)
    print(f"Linear reasoning:\n{cot_response.content[:300]}...")
    print()
    
    # Tree-of-Thought approach
    print("TREE-OF-THOUGHT Approach:")
    tot_response = prompter.basic_tot_problem_solving(problem)
    print(f"Multi-path exploration:\n{tot_response.content[:300]}...")
    print()
    
    print("📊 Comparison Results:")
    print("   CoT: Linear, step-by-step, single path")
    print("   ToT: Multi-path, comparative, explores alternatives")
    print("   Use CoT for: Problems with clear logical sequence")
    print("   Use ToT for: Creative, strategic, or optimization problems")


def demonstrate_tot_evolution():
    """Show how Tree-of-Thought builds on previous techniques."""
    print("\n=== Evolution: Zero-Shot → Few-Shot → CoT → ToT ===\n")
    
    prompter = TreeOfThoughtPrompter()
    problem = "Increase customer retention for a subscription service"
    
    print("1. Zero-Shot:")
    zero_shot = f"How to {problem.lower()}?"
    response = prompter.provider.generate(zero_shot)
    print(f"Direct answer: {response.content[:100]}...")
    print()
    
    print("2. Chain-of-Thought:")
    cot = f"How to {problem.lower()}? Let me think step by step:"
    response = prompter.provider.generate(cot)
    print(f"Linear reasoning: {response.content[:100]}...")
    print()
    
    print("3. Tree-of-Thought:")
    tot = prompter.basic_tot_problem_solving(problem)
    print(f"Multi-path analysis: {tot.content[:100]}...")
    print()
    
    print("🚀 Evolution Summary:")
    print("   Zero-Shot: Quick, direct answers")
    print("   Few-Shot: Pattern learning from examples")
    print("   Chain-of-Thought: Step-by-step reasoning")
    print("   Tree-of-Thought: Multi-path exploration and comparison")
    print("   Each builds on previous techniques for more sophisticated reasoning!")


if __name__ == "__main__":
    demo_tree_of_thought_techniques()
    compare_cot_vs_tot()
    demonstrate_tot_evolution()