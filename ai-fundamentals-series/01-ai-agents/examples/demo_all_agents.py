"""
Complete Demo Runner for All AI Agent Types
Blog 1: Understanding AI Agents

Run demonstrations of all agent types in sequence to see
the progression from simple reflex to sophisticated hybrid agents.
"""

import sys
import time
from reflex_agent import demo_reflex_agent
from model_based_agent import demo_model_based_agent
from goal_based_agent import demo_goal_based_agent
from utility_based_agent import demo_utility_based_agent
from learning_agent import demo_learning_agent
from hybrid_agent import demo_hybrid_agent
from smart_home_agent import demo_smart_home_agent


def print_header(title, subtitle=""):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"🤖 {title}")
    if subtitle:
        print(f"   {subtitle}")
    print("="*80)


def print_separator():
    """Print section separator."""
    print("\n" + "-"*80 + "\n")


def wait_for_user():
    """Wait for user input to continue."""
    input("\n👉 Press Enter to continue to the next agent type...")


def run_all_agent_demos():
    """Run all agent demonstrations in educational sequence."""
    print_header(
        "AI Agents: Complete Educational Journey", 
        "From Simple Reflex to Sophisticated Hybrid Intelligence"
    )
    
    print("""
Welcome to the complete AI Agents demonstration!

This demo will walk you through 6 different types of AI agents,
showing the evolution from simple reactive systems to sophisticated
hybrid agents that power modern AI applications.

Each demonstration builds on the previous concepts:
1. Reflex Agents - Immediate responses
2. Model-Based Agents - Memory and world modeling  
3. Goal-Based Agents - Planning and objectives
4. Utility-Based Agents - Balancing trade-offs
5. Learning Agents - Adaptation through experience
6. Hybrid Agents - Combining all approaches
    """)
    
    input("\n🚀 Ready to explore AI agents? Press Enter to begin...")
    
    try:
        # 1. Reflex Agents
        print_header("1. Reflex Agents", "Immediate Response Based on Current Input")
        print("💡 Key Concept: if-then rules with no memory of past events")
        demo_reflex_agent()
        wait_for_user()
        
        # 2. Model-Based Agents  
        print_header("2. Model-Based Agents", "Decision Making with Internal World Model")
        print("💡 Key Concept: Maintains memory of environment state for better decisions")
        demo_model_based_agent()
        wait_for_user()
        
        # 3. Goal-Based Agents
        print_header("3. Goal-Based Agents", "Planning Actions to Achieve Objectives")
        print("💡 Key Concept: Plans sequences of actions to reach specific goals")
        demo_goal_based_agent()
        wait_for_user()
        
        # 4. Utility-Based Agents
        print_header("4. Utility-Based Agents", "Balancing Multiple Competing Factors")
        print("💡 Key Concept: Optimizes decisions using utility functions with trade-offs")
        demo_utility_based_agent()
        wait_for_user()
        
        # 5. Learning Agents
        print_header("5. Learning Agents", "Improving Performance Through Experience")
        print("💡 Key Concept: Adapts behavior based on feedback and outcomes")
        demo_learning_agent()
        wait_for_user()
        
        # 6. Hybrid Agents
        print_header("6. Hybrid Agents", "Combining Multiple Intelligence Types")
        print("💡 Key Concept: Real-world systems using multiple approaches together")
        demo_hybrid_agent()
        wait_for_user()
        
        # 7. Complete Smart Home Demo
        print_header("7. Complete Integration", "Smart Home Agent Using All Concepts")
        print("💡 Final Demo: Comprehensive hybrid agent for home automation")
        demo_smart_home_agent()
        
        # Summary
        print_header("🎓 Learning Complete!", "Agent Intelligence Evolution Summary")
        print("""
You've now seen the complete evolution of AI agent intelligence:

📈 Progression Summary:
   Reflex → Model-Based → Goal-Based → Utility-Based → Learning → Hybrid

🧠 Intelligence Types Demonstrated:
   ✅ Immediate responses (Reflex)
   ✅ Memory and state tracking (Model-based)  
   ✅ Planning and goal achievement (Goal-based)
   ✅ Multi-factor optimization (Utility-based)
   ✅ Adaptive learning (Learning)
   ✅ Integrated hybrid approaches (Hybrid)

🌟 Real-World Applications:
   • Smart home automation (demonstrated)
   • Fraud detection systems (demonstrated)  
   • Recommendation engines (demonstrated)
   • Autonomous vehicles
   • Healthcare monitoring
   • Financial trading systems

🎯 Key Insights:
   1. Simple agents work for simple problems
   2. Complex environments need sophisticated agents
   3. Most real systems are hybrid combinations
   4. The right agent type depends on your specific needs
   5. Modern AI systems combine multiple intelligence types

🚀 Next Steps:
   → Experiment with the code examples
   → Modify agents for your own use cases
   → Explore Blog 2: Prompt Engineering Fundamentals
   → Build your own hybrid agents!

Thanks for exploring the fascinating world of AI agents! 🤖✨
        """)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user. Thanks for exploring AI agents!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check individual agent files if you encounter issues.")


def run_quick_demo():
    """Run abbreviated demo focusing on key differences."""
    print_header("AI Agents: Quick Tour", "Core Concepts in 5 Minutes")
    
    print("""
Quick demonstration of the 6 main agent types.
Each will show a brief example highlighting the key concept.
    """)
    
    # Quick examples of each type
    agents_info = [
        ("Reflex", "Immediate if-then responses", demo_reflex_agent),
        ("Model-Based", "Memory-driven decisions", demo_model_based_agent), 
        ("Goal-Based", "Planning for objectives", demo_goal_based_agent),
        ("Utility-Based", "Balancing trade-offs", demo_utility_based_agent),
        ("Learning", "Improving through experience", demo_learning_agent),
        ("Hybrid", "Combining all approaches", demo_hybrid_agent)
    ]
    
    for i, (name, description, demo_func) in enumerate(agents_info, 1):
        print(f"\n{i}. {name} Agents - {description}")
        print("-" * 50)
        
        # Run abbreviated demo (could modify demo functions to accept 'quick' parameter)
        demo_func()
        
        if i < len(agents_info):
            time.sleep(1)  # Brief pause between demos
    
    print_header("🎯 Quick Tour Complete!")
    print("For detailed exploration, run: python demo_all_agents.py --full")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agents Demo Runner")
    parser.add_argument("--quick", action="store_true", 
                       help="Run abbreviated demo (5 minutes)")
    parser.add_argument("--full", action="store_true",
                       help="Run complete educational demo (default)")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_demo()
    else:
        run_all_agent_demos()