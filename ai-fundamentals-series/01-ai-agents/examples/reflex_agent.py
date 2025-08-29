"""
Simple Reflex Agent Implementation
Blog 1: Understanding AI Agents - Section 2.1

A simple reflex agent that operates on if-then rules,
reacting instantly to current situations without memory.

Example: Smart doorbell security agent
"""


def reflex_agent(state):
    """
    A simple doorbell security agent that reacts to motion detection.
    
    Args:
        state (str): Current environment state
        
    Returns:
        str: Action to take based on current state
    """
    if state == "motion_detected":
        return "Start recording and send alert"
    elif state == "no_motion":
        return "Stay in standby mode"
    return "Monitor continuously"


class DoorbellSecurityAgent:
    """
    Extended doorbell security agent with multiple sensor types.
    Demonstrates pure reflex behavior without memory.
    """
    
    def __init__(self):
        self.name = "Doorbell Security Agent"
    
    def perceive_and_act(self, sensor_input):
        """
        Process sensor input and return immediate action.
        
        Args:
            sensor_input (dict): Dictionary containing sensor data
                - motion: bool or str
                - sound_level: int (0-100)
                - time_of_day: str
        
        Returns:
            str: Action to take
        """
        motion = sensor_input.get('motion', False)
        sound_level = sensor_input.get('sound_level', 0)
        time_of_day = sensor_input.get('time_of_day', 'day')
        
        # Reflex rules - no memory, just immediate reactions
        if motion == "motion_detected" or motion is True:
            if time_of_day == "night":
                return "🚨 Motion detected at night! Recording + Bright light + Alert homeowner"
            else:
                return "📹 Motion detected during day. Recording + Standard alert"
        
        if sound_level > 80:
            return "🔊 Loud noise detected! Enhanced recording + Sound alert"
        
        if sound_level > 50:
            return "👂 Moderate sound detected. Standard monitoring"
        
        return "😴 All quiet. Standby mode"


def demo_reflex_agent():
    """Demonstration of reflex agent behavior."""
    print("=== Simple Reflex Agent Demo ===\n")
    
    # Basic reflex agent
    print("1. Basic Doorbell Agent:")
    test_states = ["motion_detected", "no_motion", "unknown_state"]
    
    for state in test_states:
        action = reflex_agent(state)
        print(f"   State: {state} → Action: {action}")
    
    print("\n" + "="*50 + "\n")
    
    # Enhanced doorbell agent
    print("2. Enhanced Doorbell Security Agent:")
    agent = DoorbellSecurityAgent()
    
    test_scenarios = [
        {"motion": "motion_detected", "sound_level": 30, "time_of_day": "day"},
        {"motion": "motion_detected", "sound_level": 45, "time_of_day": "night"},
        {"motion": False, "sound_level": 85, "time_of_day": "day"},
        {"motion": False, "sound_level": 60, "time_of_day": "evening"},
        {"motion": False, "sound_level": 20, "time_of_day": "night"}
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        action = agent.perceive_and_act(scenario)
        print(f"   Scenario {i}: {scenario}")
        print(f"   → {action}\n")
    
    print("Key Insight: Notice how the agent has NO memory between scenarios.")
    print("Each decision is made purely on current input - this is reflex behavior!")


if __name__ == "__main__":
    demo_reflex_agent()