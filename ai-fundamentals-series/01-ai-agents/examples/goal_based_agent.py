"""
Goal-Based Agent Implementation
Blog 1: Understanding AI Agents - Section 2.3

Goal-based agents plan actions to reach specific objectives.
They can reason about different paths and choose actions
that bring them closer to their goals.

Example: Ride-sharing app planning optimal pickup routes
"""


class GoalBasedAgent:
    """
    A ride-sharing agent that plans optimal pickup routes.
    Demonstrates goal-based reasoning and path planning.
    """
    
    def __init__(self):
        self.routes = {
            'Downtown': [('Mall', 15), ('Airport', 45), ('University', 20)],
            'Mall': [('Downtown', 15), ('Airport', 30), ('University', 25)],
            'Airport': [('Downtown', 45), ('Mall', 30), ('University', 35)],
            'University': [('Downtown', 20), ('Mall', 25), ('Airport', 35)]
        }
        self.name = "Ride-sharing Goal Agent"
    
    def find_best_route(self, start, goal):
        """
        Find optimal route to reach goal from start location.
        
        Args:
            start (str): Starting location
            goal (str): Goal destination
            
        Returns:
            str: Planned route with time estimate
        """
        if goal in dict(self.routes.get(start, [])):
            time = dict(self.routes[start])[goal]
            return f"Best route from {start} to {goal}: Direct path ({time} min)"
        return f"Route planning needed for {start} to {goal}"


class AdvancedDeliveryAgent:
    """
    Enhanced delivery agent with sophisticated goal-based planning.
    Can handle multiple goals, constraints, and dynamic replanning.
    """
    
    def __init__(self, city_map):
        self.city_map = city_map  # Graph of city with distances and traffic
        self.current_location = "Depot"
        self.fuel_level = 100
        self.current_goals = []
        self.completed_goals = []
        self.constraints = {
            "max_distance_per_trip": 100,
            "fuel_efficiency": 10,  # km per fuel unit
            "traffic_multiplier": {"rush_hour": 1.5, "normal": 1.0, "late_night": 0.8}
        }
    
    def add_goal(self, goal_type, location, priority=1, deadline=None):
        """
        Add a new goal to the agent's goal list.
        
        Args:
            goal_type (str): Type of goal (delivery, pickup, maintenance)
            location (str): Target location
            priority (int): Priority level (1=highest, 5=lowest)
            deadline (str): Optional deadline
        """
        goal = {
            "type": goal_type,
            "location": location,
            "priority": priority,
            "deadline": deadline,
            "status": "pending"
        }
        self.current_goals.append(goal)
        return f"Added goal: {goal_type} at {location} (Priority: {priority})"
    
    def plan_optimal_route(self, time_of_day="normal"):
        """
        Plan optimal route to complete all current goals.
        
        Args:
            time_of_day (str): Current time affecting traffic
            
        Returns:
            dict: Planned route with actions and reasoning
        """
        if not self.current_goals:
            return {"plan": "No goals to plan for", "total_time": 0, "fuel_needed": 0}
        
        # Sort goals by priority and deadline
        sorted_goals = sorted(self.current_goals, 
                            key=lambda g: (g["priority"], g["deadline"] or "9999"))
        
        plan = {
            "route": [],
            "reasoning": [],
            "total_time": 0,
            "fuel_needed": 0,
            "feasible": True
        }
        
        current_pos = self.current_location
        traffic_factor = self.constraints["traffic_multiplier"][time_of_day]
        
        for goal in sorted_goals:
            destination = goal["location"]
            
            # Calculate distance and time
            if destination in self.city_map.get(current_pos, {}):
                base_time = self.city_map[current_pos][destination]
                actual_time = base_time * traffic_factor
                fuel_cost = base_time / self.constraints["fuel_efficiency"]
                
                # Check constraints
                if plan["fuel_needed"] + fuel_cost > self.fuel_level:
                    plan["reasoning"].append(f"⚠️  Goal {goal['type']} at {destination} may require refueling")
                    plan["feasible"] = False
                
                # Add to plan
                plan["route"].append({
                    "from": current_pos,
                    "to": destination,
                    "goal": goal["type"],
                    "time": actual_time,
                    "fuel_cost": fuel_cost
                })
                
                plan["reasoning"].append(
                    f"Goal: {goal['type']} at {destination} (Priority {goal['priority']}) - "
                    f"Time: {actual_time:.1f}min, Fuel: {fuel_cost:.1f}"
                )
                
                plan["total_time"] += actual_time
                plan["fuel_needed"] += fuel_cost
                current_pos = destination
            
            else:
                plan["reasoning"].append(f"❌ No route found to {destination}")
                plan["feasible"] = False
        
        return plan
    
    def execute_next_action(self):
        """
        Execute the next action in pursuit of current goals.
        
        Returns:
            str: Action taken and reasoning
        """
        if not self.current_goals:
            return "No current goals. Returning to depot for new assignments."
        
        # Get highest priority goal
        next_goal = min(self.current_goals, key=lambda g: g["priority"])
        destination = next_goal["location"]
        
        # Check if we can reach destination
        if destination in self.city_map.get(self.current_location, {}):
            travel_time = self.city_map[self.current_location][destination]
            fuel_needed = travel_time / self.constraints["fuel_efficiency"]
            
            if fuel_needed <= self.fuel_level:
                # Execute movement
                self.fuel_level -= fuel_needed
                old_location = self.current_location
                self.current_location = destination
                
                # Complete goal
                next_goal["status"] = "completed"
                self.completed_goals.append(next_goal)
                self.current_goals.remove(next_goal)
                
                return (f"🚚 Moved from {old_location} to {destination} "
                       f"(Goal: {next_goal['type']}) - "
                       f"Time: {travel_time}min, Fuel remaining: {self.fuel_level:.1f}")
            else:
                return f"⛽ Insufficient fuel to reach {destination}. Need refueling."
        else:
            return f"❌ No route available to {destination}"
    
    def replan_on_obstacle(self, blocked_location):
        """
        Replan goals when encountering obstacles.
        
        Args:
            blocked_location (str): Location that is now inaccessible
            
        Returns:
            str: Replanning decision
        """
        affected_goals = [g for g in self.current_goals if g["location"] == blocked_location]
        
        if not affected_goals:
            return f"Obstacle at {blocked_location} doesn't affect current goals"
        
        # Simple replanning: postpone affected goals
        for goal in affected_goals:
            goal["priority"] += 1  # Lower priority
            
        return (f"🚧 Replanned: {len(affected_goals)} goals affected by obstacle at "
               f"{blocked_location}. Priorities adjusted.")


def demo_goal_based_agent():
    """Demonstrate goal-based agent behavior."""
    print("=== Goal-Based Agent Demo ===\n")
    
    # Basic goal-based agent from blog
    print("1. Basic Ride-sharing Agent (from blog):")
    agent = GoalBasedAgent()
    
    test_routes = [
        ('Downtown', 'Airport'),
        ('Mall', 'University'),
        ('Airport', 'Downtown'),
        ('University', 'Mall')
    ]
    
    for start, goal in test_routes:
        result = agent.find_best_route(start, goal)
        print(f"   {result}")
    
    print("\n" + "="*60 + "\n")
    
    # Advanced delivery agent
    print("2. Advanced Delivery Agent with Multiple Goals:")
    
    # Define city map (simplified)
    city_map = {
        "Depot": {"Mall": 20, "Hospital": 15, "School": 25},
        "Mall": {"Depot": 20, "Hospital": 10, "Restaurant": 15},
        "Hospital": {"Depot": 15, "Mall": 10, "School": 12},
        "School": {"Depot": 25, "Hospital": 12, "Restaurant": 18},
        "Restaurant": {"Mall": 15, "School": 18}
    }
    
    delivery_agent = AdvancedDeliveryAgent(city_map)
    
    # Add multiple goals
    print("   Adding goals:")
    print(f"   → {delivery_agent.add_goal('delivery', 'Hospital', priority=1, deadline='urgent')}")
    print(f"   → {delivery_agent.add_goal('pickup', 'Mall', priority=2)}")
    print(f"   → {delivery_agent.add_goal('delivery', 'School', priority=1, deadline='2pm')}")
    print(f"   → {delivery_agent.add_goal('maintenance', 'Depot', priority=3)}")
    
    print("\n   Planning optimal route:")
    plan = delivery_agent.plan_optimal_route("rush_hour")
    
    print(f"   📋 Plan feasible: {plan['feasible']}")
    print(f"   📊 Total time: {plan['total_time']:.1f} min")
    print(f"   ⛽ Fuel needed: {plan['fuel_needed']:.1f} units")
    
    print("\n   🧠 Reasoning:")
    for reason in plan['reasoning']:
        print(f"      {reason}")
    
    print("\n   📍 Planned route:")
    for step in plan['route']:
        print(f"      {step['from']} → {step['to']} ({step['goal']}) - {step['time']:.1f}min")
    
    print("\n" + "="*60 + "\n")
    
    # Demonstrate goal execution and replanning
    print("3. Goal Execution and Dynamic Replanning:")
    
    for i in range(3):
        action = delivery_agent.execute_next_action()
        print(f"   Step {i+1}: {action}")
    
    # Simulate obstacle
    print("\n   🚧 Obstacle encountered!")
    replan_result = delivery_agent.replan_on_obstacle("School")
    print(f"   → {replan_result}")
    
    # Continue execution
    print("\n   Continuing with updated plan:")
    action = delivery_agent.execute_next_action()
    print(f"   → {action}")
    
    print(f"\n   📈 Goals completed: {len(delivery_agent.completed_goals)}")
    print(f"   📋 Goals remaining: {len(delivery_agent.current_goals)}")
    
    print("\nKey Insight: Goal-based agents plan sequences of actions")
    print("to achieve objectives, and can replan when circumstances change!")


if __name__ == "__main__":
    demo_goal_based_agent()