"""
Hybrid Agent Implementation
Blog 1: Understanding AI Agents - Section 2.6

Hybrid agents combine multiple intelligence types to handle
complex real-world scenarios. Most sophisticated AI systems
use hybrid approaches for robustness and effectiveness.

Example: Banking fraud detection system combining all agent types
"""


class HybridFraudAgent:
    """
    A banking fraud detection agent with multiple intelligence types.
    Combines reflex, model-based, goal-based, utility-based, and learning approaches.
    """
    
    def __init__(self):
        self.user_patterns = {}  # Model-based memory
        self.blacklist = {'suspicious_country'}  # Known bad actors
        self.learning_data = []  # For continuous improvement
        self.name = "Hybrid Fraud Detection Agent"
    
    def evaluate_transaction(self, user_id, amount, location, is_training=False):
        """
        Evaluate transaction using hybrid intelligence approach.
        
        Args:
            user_id (str): User identifier
            amount (float): Transaction amount
            location (str): Transaction location
            is_training (bool): Whether this is for training the learning component
            
        Returns:
            str: Fraud evaluation decision
        """
        # Reflex: immediate red flags
        if location in self.blacklist or amount > 10000:
            return "BLOCKED: High-risk transaction detected!"
        
        # Model-based: check against user patterns
        if user_id in self.user_patterns:
            avg_amount = self.user_patterns[user_id]['avg_amount']
            if amount > avg_amount * 3:
                return f"FLAGGED: Unusual amount for user {user_id}"
        
        # Goal-based + Utility: balance security vs user experience
        risk_score = self._calculate_risk(amount, location)
        if risk_score > 7:
            return "REVIEW: Medium risk - human verification needed"
        elif risk_score > 4:
            return "APPROVED: Low risk - additional monitoring"
        else:
            return "APPROVED: Normal transaction"
    
    def _calculate_risk(self, amount, location):
        """Simplified utility calculation for risk assessment."""
        risk = 0
        if amount > 1000: risk += 3
        if location == 'new_location': risk += 2
        return risk


class AdvancedHybridSecurityAgent:
    """
    Enhanced hybrid security agent demonstrating sophisticated
    integration of multiple intelligence types for cybersecurity.
    """
    
    def __init__(self):
        self.name = "Advanced Hybrid Security Agent"
        
        # Reflex rules (immediate responses)
        self.emergency_rules = {
            "malware_signature": "BLOCK_IMMEDIATELY",
            "known_attack_ip": "BLOCK_IMMEDIATELY", 
            "system_intrusion": "LOCKDOWN_IMMEDIATE"
        }
        
        # Model-based knowledge (behavioral patterns)
        self.user_baselines = {}
        self.network_patterns = {}
        self.threat_signatures = set()
        
        # Goal-based objectives (prioritized)
        self.security_goals = [
            {"goal": "prevent_data_breach", "priority": 1, "status": "active"},
            {"goal": "maintain_system_availability", "priority": 2, "status": "active"},
            {"goal": "minimize_false_positives", "priority": 3, "status": "active"}
        ]
        
        # Utility weights for decision making
        self.utility_weights = {
            "security": 0.5,
            "usability": 0.3, 
            "performance": 0.2
        }
        
        # Learning components
        self.threat_history = []
        self.false_positive_history = []
        self.adaptation_rate = 0.1
        
    def analyze_security_event(self, event_data):
        """
        Analyze security event using hybrid approach.
        
        Args:
            event_data (dict): Security event details
        
        Returns:
            dict: Analysis result with decision and reasoning
        """
        event_type = event_data.get("type")
        source_ip = event_data.get("source_ip")
        user_id = event_data.get("user_id")
        severity = event_data.get("severity", "medium")
        
        analysis = {
            "event": event_data,
            "intelligence_types_used": [],
            "decision": None,
            "reasoning": [],
            "confidence": 0.0
        }
        
        # 1. REFLEX: Check for immediate threats
        reflex_result = self._reflex_analysis(event_type, source_ip)
        if reflex_result:
            analysis["intelligence_types_used"].append("reflex")
            analysis["decision"] = reflex_result
            analysis["reasoning"].append("Immediate threat pattern detected")
            analysis["confidence"] = 0.95
            return analysis
        
        # 2. MODEL-BASED: Compare against known patterns
        model_result = self._model_based_analysis(event_data)
        analysis["intelligence_types_used"].append("model_based")
        analysis["reasoning"].extend(model_result["reasoning"])
        
        # 3. GOAL-BASED: Consider security objectives
        goal_result = self._goal_based_analysis(event_data, model_result)
        analysis["intelligence_types_used"].append("goal_based")
        analysis["reasoning"].extend(goal_result["reasoning"])
        
        # 4. UTILITY-BASED: Balance competing factors
        utility_result = self._utility_based_analysis(event_data, model_result, goal_result)
        analysis["intelligence_types_used"].append("utility_based")
        analysis["decision"] = utility_result["decision"]
        analysis["reasoning"].extend(utility_result["reasoning"])
        analysis["confidence"] = utility_result["confidence"]
        
        # 5. LEARNING: Update knowledge based on results
        self._update_learning(event_data, analysis)
        analysis["intelligence_types_used"].append("learning")
        
        return analysis
    
    def _reflex_analysis(self, event_type, source_ip):
        """Immediate reflex responses to critical threats."""
        if event_type in self.emergency_rules:
            return self.emergency_rules[event_type]
        
        if source_ip and source_ip.startswith("192.168.1."):
            return None  # Internal IP, continue analysis
        
        if event_type == "login_failure" and source_ip in self.threat_signatures:
            return "BLOCK_IP_24H"
        
        return None
    
    def _model_based_analysis(self, event_data):
        """Analyze based on behavioral models and patterns."""
        user_id = event_data.get("user_id")
        event_type = event_data.get("type")
        timestamp = event_data.get("timestamp", 0)
        
        result = {"anomaly_score": 0, "reasoning": []}
        
        # Check user behavioral patterns
        if user_id and user_id in self.user_baselines:
            baseline = self.user_baselines[user_id]
            
            # Check login time patterns
            if event_type == "login_attempt":
                usual_hours = baseline.get("usual_login_hours", [9, 17])
                current_hour = (timestamp % 86400) // 3600  # Simplified hour extraction
                
                if current_hour not in range(usual_hours[0], usual_hours[1]):
                    result["anomaly_score"] += 3
                    result["reasoning"].append(f"Login outside usual hours for {user_id}")
            
            # Check access patterns
            if event_type == "file_access":
                usual_files = baseline.get("usual_files", set())
                accessed_file = event_data.get("file_path", "")
                
                if accessed_file not in usual_files:
                    result["anomaly_score"] += 2
                    result["reasoning"].append(f"Unusual file access pattern for {user_id}")
        
        # Check network patterns
        source_ip = event_data.get("source_ip")
        if source_ip:
            if source_ip in self.network_patterns:
                pattern = self.network_patterns[source_ip]
                if pattern.get("reputation", "good") == "suspicious":
                    result["anomaly_score"] += 4
                    result["reasoning"].append(f"Source IP {source_ip} has suspicious history")
            else:
                result["anomaly_score"] += 1
                result["reasoning"].append(f"Unknown source IP: {source_ip}")
        
        return result
    
    def _goal_based_analysis(self, event_data, model_result):
        """Analyze based on security goals and objectives."""
        result = {"actions": [], "reasoning": []}
        
        anomaly_score = model_result.get("anomaly_score", 0)
        event_type = event_data.get("type")
        severity = event_data.get("severity", "medium")
        
        # Goal 1: Prevent data breach (highest priority)
        if event_type in ["data_exfiltration", "privilege_escalation"] or anomaly_score > 5:
            result["actions"].append("enhanced_monitoring")
            result["reasoning"].append("High priority: Preventing potential data breach")
        
        # Goal 2: Maintain system availability
        if event_type == "dos_attempt" or severity == "high":
            result["actions"].append("rate_limiting")
            result["reasoning"].append("Medium priority: Maintaining system availability")
        
        # Goal 3: Minimize false positives
        if anomaly_score < 3 and event_type not in ["malware_detected", "intrusion_attempt"]:
            result["actions"].append("standard_monitoring")
            result["reasoning"].append("Low priority: Minimizing disruption to normal operations")
        
        return result
    
    def _utility_based_analysis(self, event_data, model_result, goal_result):
        """Make final decision balancing security, usability, and performance."""
        security_score = 0
        usability_score = 10  # Start with max usability
        performance_score = 10  # Start with max performance
        
        anomaly_score = model_result.get("anomaly_score", 0)
        suggested_actions = goal_result.get("actions", [])
        
        # Calculate security utility
        security_score = min(10, anomaly_score * 2)
        
        # Calculate usability impact
        if "BLOCK" in str(suggested_actions):
            usability_score -= 8
        elif "enhanced_monitoring" in suggested_actions:
            usability_score -= 2
        
        # Calculate performance impact
        if "enhanced_monitoring" in suggested_actions:
            performance_score -= 3
        if "rate_limiting" in suggested_actions:
            performance_score -= 2
        
        # Weighted utility calculation
        total_utility = (
            self.utility_weights["security"] * security_score +
            self.utility_weights["usability"] * usability_score + 
            self.utility_weights["performance"] * performance_score
        )
        
        # Decision logic based on utility
        if total_utility > 8:
            decision = "ALLOW_WITH_MONITORING"
            confidence = 0.8
        elif total_utility > 5:
            decision = "REQUIRE_ADDITIONAL_AUTH"
            confidence = 0.6
        else:
            decision = "BLOCK_PENDING_REVIEW"
            confidence = 0.9
        
        return {
            "decision": decision,
            "confidence": confidence,
            "utility_score": total_utility,
            "reasoning": [
                f"Security utility: {security_score}/10",
                f"Usability utility: {usability_score}/10", 
                f"Performance utility: {performance_score}/10",
                f"Total weighted utility: {total_utility:.1f}/10"
            ]
        }
    
    def _update_learning(self, event_data, analysis):
        """Update learning components based on analysis results."""
        # Store threat pattern for learning
        learning_entry = {
            "event_type": event_data.get("type"),
            "decision": analysis.get("decision"),
            "anomaly_score": analysis.get("anomaly_score", 0),
            "was_threat": None  # Would be updated later with ground truth
        }
        
        self.threat_history.append(learning_entry)
        
        # Simple adaptation: adjust weights if pattern emerges
        if len(self.threat_history) > 10:
            recent_threats = self.threat_history[-10:]
            high_risk_decisions = [t for t in recent_threats if "BLOCK" in str(t["decision"])]
            
            if len(high_risk_decisions) > 7:  # High threat environment
                self.utility_weights["security"] = min(0.7, self.utility_weights["security"] + 0.05)
                self.utility_weights["usability"] = max(0.2, self.utility_weights["usability"] - 0.05)


def demo_hybrid_agent():
    """Demonstrate hybrid agent behavior."""
    print("=== Hybrid Agent Demo ===\n")
    
    # Basic hybrid fraud agent from blog
    print("1. Basic Fraud Detection Agent (from blog):")
    fraud_agent = HybridFraudAgent()
    
    test_transactions = [
        ("user123", 15000, "home"),
        ("user123", 500, "new_location"),
        ("user456", 800, "home"),
        ("user789", 50, "suspicious_country")
    ]
    
    for user_id, amount, location in test_transactions:
        result = fraud_agent.evaluate_transaction(user_id, amount, location)
        print(f"   Transaction: {user_id}, ${amount}, {location}")
        print(f"   → Decision: {result}\n")
    
    print("="*70 + "\n")
    
    # Advanced hybrid security agent
    print("2. Advanced Hybrid Security Agent:")
    security_agent = AdvancedHybridSecurityAgent()
    
    # Add some baseline user patterns for demonstration
    security_agent.user_baselines = {
        "alice": {"usual_login_hours": [8, 18], "usual_files": {"report.pdf", "data.csv"}},
        "bob": {"usual_login_hours": [9, 17], "usual_files": {"code.py", "docs.txt"}}
    }
    
    security_agent.network_patterns = {
        "203.0.113.1": {"reputation": "suspicious", "last_seen": "2024-01-01"},
        "192.168.1.100": {"reputation": "good", "last_seen": "2024-01-15"}
    }
    
    # Test various security events
    security_events = [
        {
            "type": "login_attempt",
            "user_id": "alice", 
            "source_ip": "192.168.1.100",
            "timestamp": 25200,  # 7 AM
            "severity": "low"
        },
        {
            "type": "file_access",
            "user_id": "alice",
            "source_ip": "203.0.113.1",
            "file_path": "/etc/passwd",
            "severity": "high"
        },
        {
            "type": "malware_signature",
            "source_ip": "unknown",
            "severity": "critical"
        },
        {
            "type": "login_failure",
            "user_id": "bob",
            "source_ip": "192.168.1.50", 
            "timestamp": 3600,  # 1 AM
            "severity": "medium"
        }
    ]
    
    for i, event in enumerate(security_events, 1):
        print(f"   🔍 Security Event {i}:")
        print(f"   Event: {event['type']} by {event.get('user_id', 'unknown')}")
        
        analysis = security_agent.analyze_security_event(event)
        
        print(f"   Decision: {analysis['decision']}")
        print(f"   Confidence: {analysis['confidence']:.2f}")
        print(f"   Intelligence Types: {', '.join(analysis['intelligence_types_used'])}")
        
        print(f"   Reasoning:")
        for reason in analysis['reasoning'][:3]:  # Show first 3 reasons
            print(f"      • {reason}")
        
        print()
    
    print("Key Insight: Hybrid agents combine multiple intelligence types")
    print("to handle complex real-world scenarios that no single approach")  
    print("could handle effectively on its own!")


if __name__ == "__main__":
    demo_hybrid_agent()