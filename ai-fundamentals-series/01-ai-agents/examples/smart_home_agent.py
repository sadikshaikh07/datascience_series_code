"""
Smart Home Agent - Complete Demo
Blog 1: Understanding AI Agents - Section 4

A comprehensive hybrid agent that manages home automation intelligently,
combining all agent types: reflex, model-based, goal-based, utility-based, and learning.

This demonstrates how real-world AI systems integrate multiple intelligence types.
"""

import random
from datetime import datetime


class SmartHomeAgent:
    """
    A hybrid agent that manages home automation intelligently.
    Combines multiple intelligence types for comprehensive home management.
    """
    
    def __init__(self):
        self.name = "Smart Home Hybrid Agent"
        
        # Model-based: Internal state tracking
        self.sensor_data = {}
        self.device_states = {
            "lights": {"living_room": False, "bedroom": False, "kitchen": False},
            "hvac": {"temperature": 22, "mode": "auto"},
            "security": {"armed": False, "cameras": {"front": True, "back": True}},
            "appliances": {"dishwasher": "idle", "washing_machine": "idle"}
        }
        
        # Learning: User preferences and patterns
        self.user_preferences = {}
        self.usage_patterns = {}
        self.energy_history = []
        
        # Utility: Cost and environmental factors
        self.energy_cost = {'peak': 0.25, 'off_peak': 0.12}
        self.current_time_period = 'off_peak'
        
        # Goal-based: Current objectives
        self.active_goals = [
            {"type": "comfort", "priority": 1, "target": "maintain_temperature"},
            {"type": "security", "priority": 1, "target": "monitor_home"},
            {"type": "efficiency", "priority": 2, "target": "minimize_energy"}
        ]
        
        # Reflex: Emergency contacts and immediate responses
        self.emergency_contacts = ['911', 'security_service']
        self.emergency_rules = {
            "fire": "call_emergency_and_evacuate",
            "intrusion": "activate_alarm_and_notify", 
            "gas_leak": "shut_off_gas_and_ventilate"
        }
        
    def perceive(self, sensor_type, value):
        """
        Perception: Gather data from various home sensors.
        Updates internal model with new sensor information.
        
        Args:
            sensor_type (str): Type of sensor
            value: Sensor reading value
            
        Returns:
            tuple: Processed sensor data
        """
        timestamp = datetime.now().strftime("%H:%M")
        
        # Update internal model (model-based behavior)
        self.sensor_data[sensor_type] = {
            'value': value, 
            'time': timestamp,
            'history': self.sensor_data.get(sensor_type, {}).get('history', [])
        }
        
        # Store history for learning
        self.sensor_data[sensor_type]['history'].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Keep only recent history (last 10 readings)
        if len(self.sensor_data[sensor_type]['history']) > 10:
            self.sensor_data[sensor_type]['history'] = \
                self.sensor_data[sensor_type]['history'][-10:]
        
        print(f"📡 Perceived: {sensor_type} = {value} at {timestamp}")
        return sensor_type, value
    
    def reason_and_act(self, sensor_type, value):
        """
        Reasoning and Action: Intelligent decision making using hybrid approach.
        
        Args:
            sensor_type (str): Sensor that triggered this reasoning
            value: Sensor value
            
        Returns:
            str: Action taken with reasoning
        """
        # 1. REFLEX: Emergency situations need immediate action
        emergency_action = self._check_emergency_reflex(sensor_type, value)
        if emergency_action:
            return emergency_action
        
        # 2. MODEL-BASED: Use historical patterns and current state
        context = self._build_context_model(sensor_type, value)
        
        # 3. GOAL-BASED: Consider active goals
        goal_recommendations = self._evaluate_goals(sensor_type, value, context)
        
        # 4. UTILITY-BASED: Balance competing factors
        utility_decision = self._calculate_utility_decision(
            sensor_type, value, context, goal_recommendations
        )
        
        # 5. LEARNING: Update preferences based on action
        self._update_learning(sensor_type, value, utility_decision)
        
        # Execute the decided action
        return self._execute_action(utility_decision)
    
    def _check_emergency_reflex(self, sensor_type, value):
        """Reflex agent behavior for emergencies - immediate response."""
        if sensor_type == 'smoke_detector' and value == 'smoke_detected':
            return self._emergency_action("🚨 FIRE DETECTED! Calling emergency services and unlocking doors!")
        
        if sensor_type == 'security_sensor' and value == 'intrusion':
            return self._emergency_action("🚨 SECURITY BREACH! Activating alarm and notifying authorities!")
        
        if sensor_type == 'gas_detector' and value == 'gas_leak':
            return self._emergency_action("⚠️ GAS LEAK! Shutting off gas valve and activating ventilation!")
        
        if sensor_type == 'water_sensor' and value == 'flood_detected':
            return self._emergency_action("💧 FLOOD DETECTED! Shutting off main water valve!")
        
        return None
    
    def _build_context_model(self, sensor_type, value):
        """Model-based: Build context from internal state and history."""
        current_hour = int(datetime.now().strftime("%H"))
        
        context = {
            "time_of_day": self._categorize_time(current_hour),
            "energy_period": "peak" if 17 <= current_hour <= 21 else "off_peak",
            "occupancy_detected": self._infer_occupancy(),
            "weather_context": self._infer_weather_context(),
            "recent_patterns": self._analyze_recent_patterns(sensor_type)
        }
        
        return context
    
    def _categorize_time(self, hour):
        """Categorize time of day for decision making."""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon" 
        elif 18 <= hour < 23:
            return "evening"
        else:
            return "night"
    
    def _infer_occupancy(self):
        """Infer if home is occupied based on recent sensor data."""
        motion_detected = False
        for sensor, data in self.sensor_data.items():
            if 'motion' in sensor and data.get('value') == 'detected':
                motion_detected = True
                break
        return motion_detected
    
    def _infer_weather_context(self):
        """Simple weather inference from temperature trends."""
        if 'outdoor_temperature' in self.sensor_data:
            temp = self.sensor_data['outdoor_temperature']['value']
            if temp > 25:
                return "hot"
            elif temp < 10:
                return "cold"
            else:
                return "mild"
        return "unknown"
    
    def _analyze_recent_patterns(self, sensor_type):
        """Analyze recent patterns in sensor data."""
        if sensor_type not in self.sensor_data:
            return {}
        
        history = self.sensor_data[sensor_type].get('history', [])
        if len(history) < 2:
            return {}
        
        recent_values = [h['value'] for h in history[-3:]]
        
        return {
            "trend": self._calculate_trend(recent_values),
            "stability": self._calculate_stability(recent_values),
            "frequency": len(history)
        }
    
    def _calculate_trend(self, values):
        """Calculate trend in recent values."""
        if len(values) < 2:
            return "stable"
        
        if isinstance(values[0], (int, float)):
            if values[-1] > values[0]:
                return "increasing"
            elif values[-1] < values[0]:
                return "decreasing"
        
        return "stable"
    
    def _calculate_stability(self, values):
        """Calculate stability of recent values."""
        if len(set(str(v) for v in values)) == 1:
            return "very_stable"
        elif len(set(str(v) for v in values)) <= len(values) // 2:
            return "stable" 
        else:
            return "variable"
    
    def _evaluate_goals(self, sensor_type, value, context):
        """Goal-based: Evaluate actions against active goals."""
        recommendations = []
        
        for goal in self.active_goals:
            if goal["type"] == "comfort":
                if sensor_type == "temperature":
                    recommendations.append(self._comfort_goal_action(value, context))
            
            elif goal["type"] == "security":
                if sensor_type in ["motion_sensor", "door_sensor", "window_sensor"]:
                    recommendations.append(self._security_goal_action(sensor_type, value, context))
            
            elif goal["type"] == "efficiency":
                recommendations.append(self._efficiency_goal_action(sensor_type, value, context))
        
        return [r for r in recommendations if r]  # Filter None values
    
    def _comfort_goal_action(self, temperature, context):
        """Recommend action for comfort goal."""
        target_temp = self.user_preferences.get('preferred_temperature', 22)
        
        if temperature > target_temp + 2:
            return {"action": "cooling", "reason": "temperature above comfort zone"}
        elif temperature < target_temp - 2:
            return {"action": "heating", "reason": "temperature below comfort zone"}
        else:
            return {"action": "maintain", "reason": "temperature in comfort zone"}
    
    def _security_goal_action(self, sensor_type, value, context):
        """Recommend action for security goal."""
        if context["time_of_day"] == "night" and value == "detected":
            return {"action": "enhanced_monitoring", "reason": "nighttime activity detected"}
        elif not context["occupancy_detected"] and value == "detected":
            return {"action": "security_alert", "reason": "activity when home appears empty"}
        else:
            return {"action": "standard_monitoring", "reason": "normal security monitoring"}
    
    def _efficiency_goal_action(self, sensor_type, value, context):
        """Recommend action for efficiency goal."""
        if context["energy_period"] == "peak":
            return {"action": "reduce_consumption", "reason": "peak energy period - optimize usage"}
        elif not context["occupancy_detected"]:
            return {"action": "standby_mode", "reason": "no occupancy - enter energy saving"}
        else:
            return {"action": "normal_operation", "reason": "occupied home - normal efficiency"}
    
    def _calculate_utility_decision(self, sensor_type, value, context, goal_recommendations):
        """Utility-based: Calculate optimal decision balancing multiple factors."""
        # Utility weights (can be learned/adjusted over time)
        weights = {
            "comfort": 0.4,
            "cost": 0.3,
            "security": 0.2,
            "environmental": 0.1
        }
        
        # Evaluate different possible actions
        possible_actions = self._generate_possible_actions(sensor_type, value, goal_recommendations)
        
        best_action = None
        best_utility = -1
        
        for action in possible_actions:
            utility = self._calculate_action_utility(action, context, weights)
            if utility > best_utility:
                best_utility = utility
                best_action = action
        
        best_action["utility_score"] = best_utility
        return best_action
    
    def _generate_possible_actions(self, sensor_type, value, goal_recommendations):
        """Generate possible actions based on sensor input and goals."""
        actions = []
        
        # Add goal-based recommendations
        for rec in goal_recommendations:
            actions.append({
                "type": rec["action"],
                "reason": rec["reason"],
                "source": "goal_based"
            })
        
        # Add sensor-specific actions
        if sensor_type == "temperature":
            actions.extend([
                {"type": "hvac_auto", "reason": "automatic temperature control", "source": "sensor_based"},
                {"type": "no_action", "reason": "maintain current state", "source": "sensor_based"}
            ])
        elif sensor_type == "motion_sensor":
            actions.extend([
                {"type": "lights_on", "reason": "motion detected", "source": "sensor_based"},
                {"type": "delayed_lights_off", "reason": "motion stopped", "source": "sensor_based"}
            ])
        
        return actions
    
    def _calculate_action_utility(self, action, context, weights):
        """Calculate utility score for a specific action."""
        comfort_score = self._score_comfort_impact(action, context)
        cost_score = self._score_cost_impact(action, context)
        security_score = self._score_security_impact(action, context)
        environmental_score = self._score_environmental_impact(action, context)
        
        total_utility = (
            weights["comfort"] * comfort_score +
            weights["cost"] * cost_score +
            weights["security"] * security_score +
            weights["environmental"] * environmental_score
        )
        
        return total_utility
    
    def _score_comfort_impact(self, action, context):
        """Score action's impact on comfort (0-10)."""
        action_type = action["type"]
        
        if action_type in ["cooling", "heating", "hvac_auto"]:
            return 9  # High comfort impact
        elif action_type in ["lights_on", "maintain"]:
            return 7  # Medium comfort impact
        else:
            return 5  # Neutral comfort impact
    
    def _score_cost_impact(self, action, context):
        """Score action's cost efficiency (0-10, higher = lower cost)."""
        action_type = action["type"]
        energy_period = context["energy_period"]
        
        cost_penalties = {
            "cooling": 3 if energy_period == "peak" else 1,
            "heating": 3 if energy_period == "peak" else 1,
            "lights_on": 1,
            "enhanced_monitoring": 2
        }
        
        penalty = cost_penalties.get(action_type, 0)
        return max(0, 10 - penalty)
    
    def _score_security_impact(self, action, context):
        """Score action's security impact (0-10)."""
        action_type = action["type"]
        
        security_benefits = {
            "enhanced_monitoring": 10,
            "security_alert": 9,
            "lights_on": 7,  # Deters intruders
            "standard_monitoring": 5
        }
        
        return security_benefits.get(action_type, 5)
    
    def _score_environmental_impact(self, action, context):
        """Score action's environmental friendliness (0-10)."""
        action_type = action["type"]
        
        if action_type in ["standby_mode", "reduce_consumption", "no_action"]:
            return 10  # Very eco-friendly
        elif action_type in ["maintain", "delayed_lights_off"]:
            return 7   # Moderately eco-friendly
        else:
            return 4   # Standard environmental impact
    
    def _update_learning(self, sensor_type, value, decision):
        """Learning: Update preferences and patterns based on actions."""
        # Track user satisfaction patterns (simplified)
        learning_entry = {
            "sensor": sensor_type,
            "value": value,
            "action": decision["type"],
            "utility": decision.get("utility_score", 0),
            "timestamp": datetime.now().strftime("%H:%M")
        }
        
        # Update usage patterns
        pattern_key = f"{sensor_type}_{value}"
        if pattern_key not in self.usage_patterns:
            self.usage_patterns[pattern_key] = []
        self.usage_patterns[pattern_key].append(learning_entry)
        
        # Simple preference learning: track most successful actions
        action_key = f"{sensor_type}_{decision['type']}"
        if action_key not in self.user_preferences:
            self.user_preferences[action_key] = {"count": 0, "avg_utility": 0}
        
        pref = self.user_preferences[action_key]
        pref["avg_utility"] = (pref["avg_utility"] * pref["count"] + decision.get("utility_score", 0)) / (pref["count"] + 1)
        pref["count"] += 1
    
    def _execute_action(self, decision):
        """Execute the decided action and return description."""
        action_type = decision["type"]
        reason = decision["reason"]
        utility = decision.get("utility_score", 0)
        
        # Map action types to specific implementations
        action_implementations = {
            "cooling": "🌡️ AC ON - Cooling home",
            "heating": "🌡️ Heating ON - Warming home", 
            "hvac_auto": "🌡️ HVAC Auto mode - Maintaining optimal temperature",
            "lights_on": "💡 Lights ON - Motion detected",
            "delayed_lights_off": "💡 Lights OFF (5min delay) - Energy saving",
            "enhanced_monitoring": "🔒 Enhanced Security - Increased monitoring",
            "security_alert": "🚨 Security Alert - Notifying homeowner",
            "standard_monitoring": "👁️ Standard monitoring - All systems normal",
            "reduce_consumption": "⚡ Energy saver mode - Reducing peak usage",
            "standby_mode": "💤 Standby mode - Minimal energy usage",
            "maintain": "✅ Maintaining current state",
            "no_action": "➖ No action needed"
        }
        
        action_description = action_implementations.get(action_type, f"🤖 {action_type}")
        
        return f"{action_description} | Reason: {reason} | Utility: {utility:.2f}"
    
    def _emergency_action(self, message):
        """Handle emergency situations with immediate actions."""
        return f"🚨 EMERGENCY: {message}"
    
    def run_simulation(self, scenarios):
        """
        Run the smart home agent through various scenarios.
        
        Args:
            scenarios (list): List of sensor scenarios to simulate
        """
        print("🏠 Smart Home Agent Starting...\n")
        print("🧠 Intelligence Types: Reflex | Model-based | Goal-based | Utility-based | Learning\n")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"--- Scenario {i} ---")
            sensor_type, value = self.perceive(scenario['sensor'], scenario['value'])
            response = self.reason_and_act(sensor_type, value)
            print(f"🤖 Action: {response}\n")
        
        # Show learning summary
        print("📊 Learning Summary:")
        print(f"   Sensor readings processed: {len(self.sensor_data)}")
        print(f"   Patterns learned: {len(self.usage_patterns)}")
        print(f"   Preferences tracked: {len(self.user_preferences)}")
        
        if self.user_preferences:
            print("   Top learned preferences:")
            sorted_prefs = sorted(self.user_preferences.items(), 
                                key=lambda x: x[1]["avg_utility"], reverse=True)
            for pref, data in sorted_prefs[:3]:
                print(f"      {pref}: {data['avg_utility']:.2f} utility (used {data['count']} times)")


def demo_smart_home_agent():
    """Run comprehensive smart home agent demonstration."""
    print("=== Complete Smart Home Agent Demo ===\n")
    
    agent = SmartHomeAgent()
    
    # Comprehensive test scenarios
    scenarios = [
        # Normal day scenarios
        {'sensor': 'temperature', 'value': 28},
        {'sensor': 'motion_sensor', 'value': 'detected'},
        {'sensor': 'user_command', 'value': 'dim_lights'},
        
        # Evening scenarios
        {'sensor': 'outdoor_temperature', 'value': 15},
        {'sensor': 'motion_sensor', 'value': 'not_detected'},
        
        # Security scenarios
        {'sensor': 'door_sensor', 'value': 'opened'},
        {'sensor': 'window_sensor', 'value': 'opened'},
        
        # Emergency scenarios
        {'sensor': 'smoke_detector', 'value': 'smoke_detected'},
        
        # Recovery scenarios
        {'sensor': 'temperature', 'value': 22},
        {'sensor': 'user_command', 'value': 'dim_lights'},
        
        # Energy efficiency scenarios
        {'sensor': 'motion_sensor', 'value': 'not_detected'},
        {'sensor': 'user_command', 'value': 'energy_save_mode'}
    ]
    
    agent.run_simulation(scenarios)
    
    print("\n" + "="*70)
    print("🎯 Key Demonstration Points:")
    print("   ✅ Reflex: Immediate emergency responses")
    print("   ✅ Model-based: Context-aware decisions using sensor history")
    print("   ✅ Goal-based: Actions aligned with comfort, security, efficiency goals")
    print("   ✅ Utility-based: Balanced decisions considering multiple factors")
    print("   ✅ Learning: Adaptive behavior based on usage patterns")
    print("\nThis hybrid approach enables robust, intelligent home automation!")


if __name__ == "__main__":
    demo_smart_home_agent()