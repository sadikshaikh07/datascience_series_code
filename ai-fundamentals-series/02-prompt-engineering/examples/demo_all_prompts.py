"""
Complete Demo Runner for All Prompt Engineering Techniques
Blog 2: Prompt Engineering Fundamentals

Run demonstrations of all prompting techniques in educational sequence
to see the evolution from simple zero-shot to sophisticated tree-of-thought.
"""

import sys
import time
from zero_shot import demo_zero_shot_techniques, compare_zero_shot_variations
from few_shot import demo_few_shot_techniques, compare_zero_vs_few_shot
from chain_of_thought import demo_chain_of_thought_techniques, compare_with_without_cot
from tree_of_thought import demo_tree_of_thought_techniques, compare_cot_vs_tot, demonstrate_tot_evolution


def print_header(title, subtitle=""):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"📝 {title}")
    if subtitle:
        print(f"   {subtitle}")
    print("="*80)


def print_separator():
    """Print section separator."""
    print("\n" + "-"*80 + "\n")


def wait_for_user():
    """Wait for user input to continue."""
    input("\n👉 Press Enter to continue to the next prompting technique...")


def check_provider_setup():
    """Check if LLM providers are properly configured."""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from shared.llm_providers import get_default_provider
        
        # Try to get a working provider
        provider = get_default_provider()
        
        # Test with a simple query
        test_response = provider.generate("Say 'Hello World'")
        
        print(f"✅ Using LLM Provider: {provider.provider_name}")
        print(f"🧪 Test successful: {test_response.content.strip()[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Provider Setup Error: {e}")
        print("\n🔧 Setup Instructions:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your API key to .env file")
        print("   3. Install requirements: pip install -r requirements.txt")
        print("   4. For Ollama: run 'ollama serve' and 'ollama pull llama3.1'")
        return False


def run_all_prompt_demos():
    """Run all prompting technique demonstrations in educational sequence."""
    print_header(
        "Prompt Engineering: Complete Educational Journey", 
        "From Zero-Shot to Tree-of-Thought Reasoning"
    )
    
    print("""
Welcome to the complete Prompt Engineering demonstration!

This demo will walk you through 4 different prompting techniques,
showing the evolution from simple direct instructions to sophisticated
multi-path reasoning approaches.

Each demonstration builds on previous concepts:
1. Zero-Shot - Direct task instructions
2. Few-Shot - Learning from examples  
3. Chain-of-Thought - Step-by-step reasoning
4. Tree-of-Thought - Multi-path exploration

You'll see practical examples of each technique and understand
when and how to use them effectively.
    """)
    
    # Check provider setup first
    if not check_provider_setup():
        print("\n⚠️  Cannot proceed without a working LLM provider.")
        print("Please check the README.md for setup instructions.")
        return
    
    input("\n🚀 Ready to explore prompt engineering? Press Enter to begin...")
    
    try:
        # 1. Zero-Shot Prompting
        print_header("1. Zero-Shot Prompting", "Direct Task Instructions Without Examples")
        print("💡 Key Concept: Give the AI a task description and expect it to perform correctly")
        print("✅ Best for: Simple, well-known tasks that LLMs can handle directly")
        print("⚠️  Limitation: May lack consistency for specific formats or complex tasks")
        
        demo_zero_shot_techniques()
        wait_for_user()
        
        # Bonus: Zero-shot variations
        print_header("Zero-Shot Bonus: Prompt Structure Variations")
        compare_zero_shot_variations()
        wait_for_user()
        
        # 2. Few-Shot Prompting
        print_header("2. Few-Shot Prompting", "Learning Patterns from Examples")
        print("💡 Key Concept: Provide examples to show the AI the pattern you want")
        print("✅ Best for: Specific formats, consistent style, complex patterns")
        print("⚠️  Trade-off: Uses more tokens but provides better consistency")
        
        demo_few_shot_techniques()
        wait_for_user()
        
        # Bonus: Zero vs Few comparison
        print_header("Zero-Shot vs Few-Shot Direct Comparison")
        compare_zero_vs_few_shot()
        wait_for_user()
        
        # 3. Chain-of-Thought Prompting
        print_header("3. Chain-of-Thought Prompting", "Step-by-Step Reasoning Process")
        print("💡 Key Concept: Ask AI to show its reasoning process before giving the final answer")
        print("✅ Best for: Math problems, logical reasoning, multi-step analysis")
        print("⚠️  Trade-off: More verbose and uses more tokens, but much more accurate")
        
        demo_chain_of_thought_techniques()
        wait_for_user()
        
        # Bonus: With vs without CoT comparison
        print_header("Chain-of-Thought Impact Demonstration")
        compare_with_without_cot()
        wait_for_user()
        
        # 4. Tree-of-Thought Prompting
        print_header("4. Tree-of-Thought Prompting", "Multi-Path Exploration and Evaluation")
        print("💡 Key Concept: Explore multiple reasoning paths simultaneously")
        print("✅ Best for: Creative tasks, strategic planning, complex optimization")
        print("⚠️  Trade-off: Most expensive approach but provides comprehensive analysis")
        
        demo_tree_of_thought_techniques()
        wait_for_user()
        
        # Bonus: CoT vs ToT comparison
        print_header("Chain-of-Thought vs Tree-of-Thought Comparison")
        compare_cot_vs_tot()
        wait_for_user()
        
        # Final: Complete Evolution
        print_header("Complete Evolution: Zero-Shot → Few-Shot → CoT → ToT")
        demonstrate_tot_evolution()
        
        # Summary and Conclusion
        print_header("🎓 Prompt Engineering Mastery Complete!", "Technique Selection Guide")
        print("""
You've now mastered the 4 core prompt engineering techniques!

📈 Technique Evolution Summary:
   Zero-Shot → Few-Shot → Chain-of-Thought → Tree-of-Thought

🧠 When to Use Each Technique:

   🎯 Zero-Shot Prompting:
      • Simple, well-defined tasks
      • Quick prototyping and testing
      • When you need fast responses
      • Example: "Translate this text to Spanish"

   📚 Few-Shot Prompting:
      • Need consistent output format
      • Specific style or tone requirements
      • Complex patterns or structures
      • Example: Converting text to structured JSON

   🔗 Chain-of-Thought:
      • Mathematical problems
      • Logical reasoning tasks
      • Multi-step analysis
      • When you need to verify reasoning
      • Example: Word problems, debugging logic

   🌳 Tree-of-Thought:
      • Creative and strategic tasks
      • Complex decision making
      • Need to explore alternatives
      • Optimization problems
      • Example: Business strategy, architectural design

🛠️ Practical Selection Guide:
   1. Start with Zero-Shot - simplest approach
   2. Add Few-Shot if you need consistency
   3. Use Chain-of-Thought for complex reasoning
   4. Apply Tree-of-Thought for creative/strategic problems

💰 Cost Considerations:
   Zero-Shot < Few-Shot < Chain-of-Thought < Tree-of-Thought
   (Choose based on your accuracy vs cost requirements)

🚀 Next Steps:
   → Experiment with hybrid approaches (combine techniques)
   → Apply to your specific domain problems
   → Explore Blog 3: Structured Outputs & Function Calling
   → Build production prompt engineering systems

🎯 Pro Tips:
   • Always test prompts with multiple examples
   • Iterate and refine based on results
   • Consider token costs in production
   • Document successful prompt patterns
   • Use version control for prompt templates

Thanks for mastering prompt engineering! 📝✨

The techniques you've learned are the foundation of modern AI applications.
Use them wisely to build intelligent, reliable systems.
        """)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user. Thanks for exploring prompt engineering!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check individual technique files if you encounter issues.")


def run_quick_demo():
    """Run abbreviated demo focusing on key differences."""
    print_header("Prompt Engineering: Quick Tour", "Core Techniques in 5 Minutes")
    
    if not check_provider_setup():
        print("\n⚠️  Cannot proceed without a working LLM provider.")
        return
    
    print("""
Quick demonstration of the 4 main prompting techniques.
Each will show a brief example highlighting the key concept.
    """)
    
    # Quick examples with same problem across all techniques
    problem = "How can I improve my team's productivity?"
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.llm_providers import get_default_provider
    provider = get_default_provider()
    
    techniques = [
        ("Zero-Shot", f"Answer this question: {problem}"),
        ("Few-Shot", f"Answer this question following these examples:\nQ: How to improve sales?\nA: 1. Analyze current metrics 2. Identify bottlenecks 3. Test solutions\n\nQ: {problem}\nA:"),
        ("Chain-of-Thought", f"Answer this question step by step: {problem}\n\nLet me think through this systematically:"),
        ("Tree-of-Thought", f"Answer this question by exploring multiple approaches:\n\n{problem}\n\nApproach 1: Process optimization\nApproach 2: Team motivation\nApproach 3: Technology solutions\n\nLet me analyze each:")
    ]
    
    for i, (name, prompt) in enumerate(techniques, 1):
        print(f"\n{i}. {name} Technique:")
        print("-" * 50)
        
        try:
            response = provider.generate(prompt)
            print(f"Response: {response.content[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
        
        if i < len(techniques):
            time.sleep(1)  # Brief pause between demos
    
    print_header("🎯 Quick Tour Complete!")
    print("Key Takeaway: Each technique offers different levels of sophistication")
    print("For detailed exploration, run: python demo_all_prompts.py --full")


def run_provider_comparison():
    """Compare different LLM providers on same prompts."""
    print_header("Multi-Provider Comparison", "Same Prompts, Different Models")
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.llm_providers import get_available_providers, get_default_provider, OpenAIProvider
    
    # Get information about available providers
    available_providers = get_available_providers()
    
    if len(available_providers) < 1:
        print("⚠️  No providers available")
        print(f"Available: {available_providers}")
        print("Configure providers in your .env file")
        return
    
    test_prompt = "Explain the concept of recursion in programming in simple terms."
    
    print(f"Testing prompt with available providers:")
    print(f"Prompt: '{test_prompt}'\n")
    
    # For now, we'll demonstrate with the default provider since only OpenAI is available
    # In the future, additional providers can be added here
    providers_to_test = [
        ("openai", lambda: get_default_provider()),
        # Could add more providers here in the future:
        # ("anthropic", lambda: AnthropicProvider()),
        # ("ollama", lambda: OllamaProvider()),
    ]
    
    for provider_name, provider_func in providers_to_test[:1]:  # Test available providers
        try:
            provider = provider_func()
            response = provider.generate(test_prompt)
            
            print(f"🤖 {provider_name.title()} Response:")
            print(f"   {response.content[:300]}...")
            print(f"   Tokens: {response.usage.get('total_tokens', 'Unknown') if response.usage else 'Unknown'}")
            print()
            
        except Exception as e:
            print(f"❌ {provider_name}: {e}\n")
    
    print("💡 Insights:")
    print("   • Different models have different strengths")
    print("   • Response styles vary between providers")
    print("   • Consider cost, speed, and quality trade-offs")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Engineering Demo Runner")
    parser.add_argument("--quick", action="store_true", 
                       help="Run abbreviated demo (5 minutes)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare different LLM providers")
    parser.add_argument("--full", action="store_true",
                       help="Run complete educational demo (default)")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_demo()
    elif args.compare:
        run_provider_comparison()
    else:
        run_all_prompt_demos()