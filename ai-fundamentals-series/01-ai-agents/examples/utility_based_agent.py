"""
Utility-Based Agent Implementation
Blog 1: Understanding AI Agents - Section 2.4

Utility-based agents maximize a utility function that measures
how "good" different outcomes are. They balance multiple competing
factors to make optimal decisions.

Example: Smart energy management system balancing comfort, cost, and environment
"""


def utility_based_agent(options):
    """
    A smart grid agent that balances comfort, cost, and environment.
    
    Args:
        options (list): List of dictionaries with 'action' and 'utility' keys
        
    Returns:
        str: Best action based on utility maximization
    """
    best_option = max(options, key=lambda x: x["utility"])
    return f"Best action: {best_option['action']} (Utility Score: {best_option['utility']})"


class SmartEnergyAgent:
    """
    Enhanced smart energy management agent that calculates utility
    based on multiple factors: comfort, cost, and environmental impact.
    """
    
    def __init__(self):
        self.name = "Smart Energy Manager"
        # Weights for different factors (must sum to 1.0)
        self.weights = {
            "comfort": 0.4,
            "cost": 0.35,
            "environmental": 0.25
        }
        self.preferences = {
            "preferred_temp": 22,
            "max_cost_per_hour": 5.0,
            "green_energy_preference": 0.8
        }
    
    def calculate_utility(self, action_data):
        """
        Calculate utility score for an action based on multiple factors.
        
        Args:
            action_data (dict): Action details including:
                - temperature_achieved: float
                - cost_per_hour: float
                - carbon_footprint: float (0-1, lower is better)
                - energy_source: str
        
        Returns:
            float: Utility score (0-10)
        """
        # Comfort score (0-10) - closer to preferred temp is better
        temp_diff = abs(action_data["temperature_achieved"] - self.preferences["preferred_temp"])
        comfort_score = max(0, 10 - temp_diff)
        
        # Cost score (0-10) - lower cost is better
        cost_ratio = action_data["cost_per_hour"] / self.preferences["max_cost_per_hour"]
        cost_score = max(0, 10 - (cost_ratio * 10))
        
        # Environmental score (0-10) - lower carbon footprint is better
        carbon_factor = action_data["carbon_footprint"]
        env_score = (1 - carbon_factor) * 10
        
        # Bonus for green energy
        if action_data["energy_source"] == "solar":
            env_score += 2
        elif action_data["energy_source"] == "wind":
            env_score += 1.5
        
        env_score = min(10, env_score)  # Cap at 10
        
        # Weighted utility calculation
        utility = (
            self.weights["comfort"] * comfort_score +
            self.weights["cost"] * cost_score +
            self.weights["environmental"] * env_score
        )
        
        return utility
    
    def choose_best_action(self, available_actions, current_conditions):
        """
        Choose the best action by evaluating utility of all options.
        
        Args:
            available_actions (list): List of possible actions
            current_conditions (dict): Current environment state
        
        Returns:
            dict: Best action with utility analysis
        """
        action_utilities = []
        
        for action in available_actions:
            utility = self.calculate_utility(action)
            action_utilities.append({
                "action": action["name"],
                "utility": utility,
                "breakdown": {
                    "comfort": abs(action["temperature_achieved"] - self.preferences["preferred_temp"]),
                    "cost": action["cost_per_hour"],
                    "environmental": action["carbon_footprint"],
                    "energy_source": action["energy_source"]
                }
            })
        
        # Find best action
        best_action = max(action_utilities, key=lambda x: x["utility"])
        
        return {
            "chosen_action": best_action,
            "all_options": action_utilities,
            "reasoning": self._explain_choice(best_action, action_utilities)
        }
    
    def _explain_choice(self, best_action, all_options):
        """Generate explanation for why this action was chosen."""
        reasons = []
        
        # Compare with alternatives
        sorted_options = sorted(all_options, key=lambda x: x["utility"], reverse=True)
        
        reasons.append(f"Selected '{best_action['action']}' with utility score {best_action['utility']:.2f}")
        
        if len(sorted_options) > 1:
            second_best = sorted_options[1]
            utility_diff = best_action['utility'] - second_best['utility']
            reasons.append(f"Beats '{second_best['action']}' by {utility_diff:.2f} utility points")
        
        # Explain key factors
        breakdown = best_action['breakdown']
        if breakdown['energy_source'] in ['solar', 'wind']:
            reasons.append(f"✅ Uses green energy ({breakdown['energy_source']})")
        
        if breakdown['cost'] < self.preferences["max_cost_per_hour"]:
            reasons.append(f"✅ Cost-effective at ${breakdown['cost']}/hour")
        
        temp_diff = breakdown['comfort']
        if temp_diff <= 2:
            reasons.append("✅ Maintains comfortable temperature")
        
        return reasons
    
    def adjust_preferences(self, new_weights=None, new_preferences=None):
        """
        Allow dynamic adjustment of utility weights and preferences.
        
        Args:
            new_weights (dict): New weight distribution
            new_preferences (dict): New user preferences
        """
        if new_weights:
            total = sum(new_weights.values())
            if abs(total - 1.0) < 0.01:  # Allow small floating point errors
                self.weights.update(new_weights)
                return f"Updated weights: {self.weights}"
            else:
                return f"Error: Weights must sum to 1.0, got {total}"
        
        if new_preferences:
            self.preferences.update(new_preferences)
            return f"Updated preferences: {new_preferences}"


def demo_utility_based_agent():
    """Demonstrate utility-based agent behavior."""
    print("=== Utility-Based Agent Demo ===\n")
    
    # Basic utility agent from blog
    print("1. Basic Smart Grid Agent (from blog):")
    options = [
        {"action": "Peak Hour Cooling", "utility": 3},
        {"action": "Off-Peak Cooling", "utility": 8},
        {"action": "Natural Cooling", "utility": 6}
    ]
    
    result = utility_based_agent(options)
    print(f"   {result}")
    
    print("\n" + "="*60 + "\n")
    
    # Advanced smart energy agent
    print("2. Advanced Smart Energy Management Agent:")
    energy_agent = SmartEnergyAgent()
    
    # Define available actions with detailed properties
    available_actions = [
        {
            "name": "High-Power AC (Grid Energy)",
            "temperature_achieved": 20,
            "cost_per_hour": 6.0,
            "carbon_footprint": 0.8,
            "energy_source": "grid"
        },
        {
            "name": "Efficient AC (Solar Energy)", 
            "temperature_achieved": 22,
            "cost_per_hour": 2.0,
            "carbon_footprint": 0.1,
            "energy_source": "solar"
        },
        {
            "name": "Fan + Natural Ventilation",
            "temperature_achieved": 25,
            "cost_per_hour": 0.5,
            "carbon_footprint": 0.05,
            "energy_source": "solar"
        },
        {
            "name": "No Cooling",
            "temperature_achieved": 28,
            "cost_per_hour": 0.0,
            "carbon_footprint": 0.0,
            "energy_source": "none"
        }
    ]
    
    current_conditions = {"outdoor_temp": 30, "indoor_temp": 26, "time": "afternoon"}
    
    # Agent chooses best action
    decision = energy_agent.choose_best_action(available_actions, current_conditions)
    
    print("   🎯 Decision Analysis:")
    chosen = decision["chosen_action"]
    print(f"   → Chosen Action: {chosen['action']}")
    print(f"   → Utility Score: {chosen['utility']:.2f}/10")
    
    print("\n   🧠 Reasoning:")
    for reason in decision["reasoning"]:
        print(f"      • {reason}")
    
    print("\n   📊 All Options Compared:")
    for option in sorted(decision["all_options"], key=lambda x: x["utility"], reverse=True):
        print(f"      {option['action']:30} | Utility: {option['utility']:.2f}")
    
    print("\n" + "="*60 + "\n")
    
    # Demonstrate preference adjustment
    print("3. Dynamic Preference Adjustment:")
    
    print("   Original weights:", energy_agent.weights)
    
    # Scenario 1: User prioritizes cost savings
    print("\n   Scenario: User wants to save money (high cost weight)")
    energy_agent.adjust_preferences(new_weights={"comfort": 0.2, "cost": 0.6, "environmental": 0.2})
    
    decision_cost = energy_agent.choose_best_action(available_actions, current_conditions)
    chosen_cost = decision_cost["chosen_action"]
    print(f"   → Cost-focused choice: {chosen_cost['action']} (Utility: {chosen_cost['utility']:.2f})")
    
    # Scenario 2: User prioritizes environment
    print("\n   Scenario: User prioritizes environment (high environmental weight)")
    energy_agent.adjust_preferences(new_weights={"comfort": 0.2, "cost": 0.2, "environmental": 0.6})
    
    decision_env = energy_agent.choose_best_action(available_actions, current_conditions)
    chosen_env = decision_env["chosen_action"]
    print(f"   → Environment-focused choice: {chosen_env['action']} (Utility: {chosen_env['utility']:.2f})")
    
    # Scenario 3: User prioritizes comfort
    print("\n   Scenario: User prioritizes comfort (high comfort weight)")
    energy_agent.adjust_preferences(new_weights={"comfort": 0.7, "cost": 0.15, "environmental": 0.15})
    
    decision_comfort = energy_agent.choose_best_action(available_actions, current_conditions)
    chosen_comfort = decision_comfort["chosen_action"]
    print(f"   → Comfort-focused choice: {chosen_comfort['action']} (Utility: {chosen_comfort['utility']:.2f})")
    
    print("\nKey Insight: Utility-based agents can balance multiple competing")
    print("objectives and adapt their decision-making based on changing priorities!")


if __name__ == "__main__":
    demo_utility_based_agent()