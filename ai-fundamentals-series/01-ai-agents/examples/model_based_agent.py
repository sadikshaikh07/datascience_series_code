"""
Model-Based Reflex Agent Implementation
Blog 1: Understanding AI Agents - Section 2.2

Model-based agents maintain an internal model of the world state,
allowing them to make decisions based on both current perceptions
and past experiences. Crucial for partially observable environments.

Example: Warehouse robot that maps its environment
"""


class ModelBasedAgent:
    """
    A warehouse robot that maps its environment and remembers obstacles.
    Demonstrates model-based behavior with internal state tracking.
    """
    
    def __init__(self):
        self.visited_locations = set()
        self.obstacles = set()
        self.name = "Warehouse Robot"
    
    def act(self, location, obstacle_detected=False):
        """
        Make decision based on current perception and internal model.
        
        Args:
            location (str): Current location
            obstacle_detected (bool): Whether obstacle detected at location
            
        Returns:
            str: Action decision based on model
        """
        if obstacle_detected:
            self.obstacles.add(location)
            return f"Obstacle detected at {location}. Finding alternate route."
        
        if location in self.visited_locations:
            return f"{location} already explored"
        
        self.visited_locations.add(location)
        return f"Exploring {location} and updating internal map"
    
    def get_model_state(self):
        """Return current internal model state."""
        return {
            "visited_locations": list(self.visited_locations),
            "known_obstacles": list(self.obstacles),
            "total_explored": len(self.visited_locations)
        }


class AdvancedWarehouseRobot:
    """
    Enhanced warehouse robot with more sophisticated internal modeling.
    Tracks inventory, optimal paths, and environmental conditions.
    """
    
    def __init__(self, warehouse_layout):
        # Internal world model
        self.world_model = {
            "visited_locations": set(),
            "obstacles": set(),
            "inventory_locations": {},
            "optimal_paths": {},
            "environmental_data": {}
        }
        self.warehouse_layout = warehouse_layout
        self.current_location = None
        self.battery_level = 100
        
    def perceive(self, location, sensors_data):
        """
        Update internal model based on sensor perceptions.
        
        Args:
            location (str): Current location
            sensors_data (dict): Sensor readings including:
                - obstacle_detected: bool
                - inventory_count: int
                - temperature: float
                - lighting: str
        """
        self.current_location = location
        self.world_model["visited_locations"].add(location)
        
        # Update obstacle information
        if sensors_data.get('obstacle_detected', False):
            self.world_model["obstacles"].add(location)
        
        # Update inventory model
        if 'inventory_count' in sensors_data:
            self.world_model["inventory_locations"][location] = sensors_data['inventory_count']
        
        # Update environmental conditions
        self.world_model["environmental_data"][location] = {
            "temperature": sensors_data.get('temperature', 20),
            "lighting": sensors_data.get('lighting', 'normal'),
            "timestamp": self._get_timestamp()
        }
    
    def reason_and_act(self, task_goal):
        """
        Use internal model to reason about best action for given goal.
        
        Args:
            task_goal (str): Current task goal
            
        Returns:
            str: Reasoned action based on internal model
        """
        # Use model to inform decision making
        if task_goal == "find_inventory":
            return self._find_optimal_inventory_location()
        elif task_goal == "avoid_obstacles":
            return self._navigate_around_obstacles()
        elif task_goal == "return_to_base":
            return self._return_to_base()
        elif task_goal == "explore_new_area":
            return self._explore_unvisited_areas()
        else:
            return f"Unknown goal: {task_goal}. Continuing standard patrol."
    
    def _find_optimal_inventory_location(self):
        """Find location with most inventory based on internal model."""
        if not self.world_model["inventory_locations"]:
            return "No inventory data in model. Exploring to gather information."
        
        best_location = max(
            self.world_model["inventory_locations"].items(),
            key=lambda x: x[1]
        )
        return f"Based on model: Heading to {best_location[0]} (highest inventory: {best_location[1]} items)"
    
    def _navigate_around_obstacles(self):
        """Plan path avoiding known obstacles."""
        known_obstacles = self.world_model["obstacles"]
        if not known_obstacles:
            return "Model shows no known obstacles. Proceeding with standard navigation."
        
        return f"Model-based navigation: Avoiding {len(known_obstacles)} known obstacles at {list(known_obstacles)}"
    
    def _return_to_base(self):
        """Return to base using optimal path from model."""
        if "Base" in self.world_model["visited_locations"]:
            return "Model confirms Base location known. Calculating optimal return path."
        else:
            return "Base location not in model. Initiating search pattern."
    
    def _explore_unvisited_areas(self):
        """Identify unexplored areas based on model."""
        all_areas = set(self.warehouse_layout)
        visited = self.world_model["visited_locations"]
        unvisited = all_areas - visited
        
        if unvisited:
            next_area = list(unvisited)[0]  # Simple selection
            return f"Model identifies unexplored areas: {list(unvisited)}. Heading to {next_area}"
        else:
            return "Model indicates all areas have been explored. Switching to maintenance mode."
    
    def _get_timestamp(self):
        """Simple timestamp simulation."""
        import time
        return int(time.time() * 1000) % 100000  # Simplified for demo
    
    def print_model_summary(self):
        """Print current state of internal model."""
        print(f"\n🧠 {self.current_location} - Internal World Model:")
        print(f"   📍 Visited: {len(self.world_model['visited_locations'])} locations")
        print(f"   🚧 Obstacles: {len(self.world_model['obstacles'])} known")
        print(f"   📦 Inventory data: {len(self.world_model['inventory_locations'])} locations")
        print(f"   🌡️  Environmental data: {len(self.world_model['environmental_data'])} readings")


def demo_model_based_agent():
    """Demonstrate model-based agent behavior."""
    print("=== Model-Based Agent Demo ===\n")
    
    # Basic model-based agent from blog
    print("1. Basic Warehouse Robot (from blog):")
    robot = ModelBasedAgent()
    
    test_scenarios = [
        ("Section A", False),
        ("Section B", True),
        ("Section A", False),  # Revisiting
        ("Section C", False),
        ("Section B", False)   # Revisiting obstacle area
    ]
    
    for location, obstacle in test_scenarios:
        action = robot.act(location, obstacle)
        print(f"   {location} (obstacle={obstacle}) → {action}")
    
    print(f"\n   Final Model State: {robot.get_model_state()}")
    
    print("\n" + "="*60 + "\n")
    
    # Advanced warehouse robot
    print("2. Advanced Warehouse Robot with Rich World Model:")
    warehouse_layout = ["Section A", "Section B", "Section C", "Loading Dock", "Base"]
    advanced_robot = AdvancedWarehouseRobot(warehouse_layout)
    
    # Simulate robot experiences
    experiences = [
        ("Section A", {"inventory_count": 50, "temperature": 22, "lighting": "good"}),
        ("Section B", {"obstacle_detected": True, "temperature": 25, "lighting": "poor"}),
        ("Section C", {"inventory_count": 75, "temperature": 20, "lighting": "good"}),
        ("Loading Dock", {"inventory_count": 25, "temperature": 18, "lighting": "excellent"}),
        ("Base", {"temperature": 22, "lighting": "normal"})
    ]
    
    # Robot explores and builds model
    for location, sensor_data in experiences:
        advanced_robot.perceive(location, sensor_data)
        advanced_robot.print_model_summary()
    
    print("\n" + "="*60 + "\n")
    
    # Test different goals using the built model
    print("3. Model-Based Decision Making:")
    goals = ["find_inventory", "avoid_obstacles", "return_to_base", "explore_new_area"]
    
    for goal in goals:
        action = advanced_robot.reason_and_act(goal)
        print(f"   Goal: {goal}")
        print(f"   → Decision: {action}\n")
    
    print("Key Insight: The agent uses its internal model (memory) to make")
    print("informed decisions, not just react to immediate perceptions!")


if __name__ == "__main__":
    demo_model_based_agent()