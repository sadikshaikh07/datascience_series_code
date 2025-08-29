"""
Learning Agent Implementation
Blog 1: Understanding AI Agents - Section 2.5

Learning agents improve their performance over time by analyzing
the results of their actions. They adapt and optimize based on
feedback and experience.

Example: Music streaming service that learns user preferences
"""


class LearningAgent:
    """
    A music streaming agent that learns user preferences over time.
    Demonstrates basic learning behavior with preference tracking.
    """
    
    def __init__(self):
        self.genre_scores = {}
        self.total_interactions = 0
        self.name = "Music Streaming Learning Agent"
    
    def update_preferences(self, genre, user_rating):
        """
        Update genre preferences based on user feedback.
        
        Args:
            genre (str): Music genre
            user_rating (int): User rating (1-5)
            
        Returns:
            str: Update confirmation with average score
        """
        if genre not in self.genre_scores:
            self.genre_scores[genre] = []
        
        self.genre_scores[genre].append(user_rating)
        self.total_interactions += 1
        
        avg_score = sum(self.genre_scores[genre]) / len(self.genre_scores[genre])
        return f"Updated {genre} preference: {avg_score:.1f}/5 (based on {len(self.genre_scores[genre])} ratings)"
    
    def recommend(self):
        """
        Recommend music based on learned preferences.
        
        Returns:
            str: Recommendation based on highest-rated genre
        """
        if not self.genre_scores:
            return "Please rate some music to get personalized recommendations!"
        
        best_genre = max(self.genre_scores.keys(), 
                        key=lambda g: sum(self.genre_scores[g])/len(self.genre_scores[g]))
        return f"Based on your preferences, try more {best_genre} music!"


class AdvancedRecommendationAgent:
    """
    Advanced recommendation agent with sophisticated learning capabilities.
    Includes exploration vs exploitation, temporal preferences, and context learning.
    """
    
    def __init__(self):
        self.name = "Advanced Recommendation Engine"
        # Learning components
        self.genre_preferences = {}
        self.artist_preferences = {}
        self.temporal_patterns = {}  # Time-based preferences
        self.context_preferences = {}  # Situation-based preferences
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.2  # 20% exploration, 80% exploitation
        self.total_recommendations = 0
        self.successful_recommendations = 0
        
        # Experience history
        self.interaction_history = []
        
    def learn_from_feedback(self, recommendation, feedback, context=None):
        """
        Learn from user feedback on recommendations.
        
        Args:
            recommendation (dict): The recommendation made
            feedback (dict): User feedback including rating and engagement
            context (dict): Context when recommendation was made
        
        Returns:
            str: Learning summary
        """
        # Record interaction
        interaction = {
            "recommendation": recommendation,
            "feedback": feedback,
            "context": context or {},
            "timestamp": self._get_timestamp()
        }
        self.interaction_history.append(interaction)
        
        # Update genre preferences
        genre = recommendation.get("genre")
        if genre:
            self._update_preference_score(self.genre_preferences, genre, feedback["rating"])
        
        # Update artist preferences
        artist = recommendation.get("artist")
        if artist:
            self._update_preference_score(self.artist_preferences, artist, feedback["rating"])
        
        # Learn temporal patterns
        if context and "time_of_day" in context:
            time_slot = context["time_of_day"]
            if genre:
                self._update_temporal_preference(time_slot, genre, feedback["rating"])
        
        # Learn contextual preferences
        if context and "mood" in context:
            mood = context["mood"]
            if genre:
                self._update_context_preference(mood, genre, feedback["rating"])
        
        # Update success metrics
        self.total_recommendations += 1
        if feedback["rating"] >= 4:  # Consider 4+ stars as successful
            self.successful_recommendations += 1
        
        return self._generate_learning_summary(interaction)
    
    def _update_preference_score(self, preference_dict, key, rating):
        """Update preference score using exponential smoothing."""
        if key not in preference_dict:
            preference_dict[key] = {"score": rating, "count": 1}
        else:
            old_score = preference_dict[key]["score"]
            new_score = old_score + self.learning_rate * (rating - old_score)
            preference_dict[key] = {
                "score": new_score,
                "count": preference_dict[key]["count"] + 1
            }
    
    def _update_temporal_preference(self, time_slot, genre, rating):
        """Learn time-based preferences."""
        if time_slot not in self.temporal_patterns:
            self.temporal_patterns[time_slot] = {}
        
        self._update_preference_score(self.temporal_patterns[time_slot], genre, rating)
    
    def _update_context_preference(self, mood, genre, rating):
        """Learn context-based preferences."""
        if mood not in self.context_preferences:
            self.context_preferences[mood] = {}
        
        self._update_preference_score(self.context_preferences[mood], genre, rating)
    
    def make_recommendation(self, context=None):
        """
        Make a recommendation using learned preferences and exploration.
        
        Args:
            context (dict): Current context (time, mood, etc.)
        
        Returns:
            dict: Recommendation with reasoning
        """
        import random
        
        # Decide: explore or exploit?
        should_explore = random.random() < self.exploration_rate
        
        if should_explore and self.total_recommendations > 10:  # Only explore after some learning
            return self._explore_recommendation(context)
        else:
            return self._exploit_recommendation(context)
    
    def _exploit_recommendation(self, context):
        """Make recommendation based on learned preferences."""
        # Start with general preferences
        best_genres = self._get_top_preferences(self.genre_preferences, n=3)
        
        # Adjust for temporal context
        if context and "time_of_day" in context:
            time_slot = context["time_of_day"]
            if time_slot in self.temporal_patterns:
                temporal_genres = self._get_top_preferences(self.temporal_patterns[time_slot], n=2)
                best_genres = temporal_genres + [g for g in best_genres if g not in temporal_genres]
        
        # Adjust for mood context
        if context and "mood" in context:
            mood = context["mood"]
            if mood in self.context_preferences:
                mood_genres = self._get_top_preferences(self.context_preferences[mood], n=2)
                best_genres = mood_genres + [g for g in best_genres if g not in mood_genres]
        
        # Select best genre
        chosen_genre = best_genres[0] if best_genres else "Pop"
        
        # Select artist based on preferences
        top_artists = self._get_top_preferences(self.artist_preferences, n=3)
        chosen_artist = top_artists[0] if top_artists else f"Popular {chosen_genre} Artist"
        
        return {
            "type": "exploit",
            "genre": chosen_genre,
            "artist": chosen_artist,
            "song": f"{chosen_artist} - Best {chosen_genre} Hit",
            "confidence": self._calculate_confidence(chosen_genre, chosen_artist),
            "reasoning": f"Based on your high rating for {chosen_genre} music and {chosen_artist}"
        }
    
    def _explore_recommendation(self, context):
        """Make exploratory recommendation to learn new preferences."""
        import random
        
        # List of genres to explore
        all_genres = ["Rock", "Jazz", "Classical", "Electronic", "Country", "R&B", "Indie", "Folk"]
        unexplored_genres = [g for g in all_genres if g not in self.genre_preferences]
        
        if unexplored_genres:
            chosen_genre = random.choice(unexplored_genres)
        else:
            # Explore less-rated genres
            low_count_genres = [g for g, data in self.genre_preferences.items() if data["count"] < 3]
            chosen_genre = random.choice(low_count_genres) if low_count_genres else random.choice(all_genres)
        
        return {
            "type": "explore",
            "genre": chosen_genre,
            "artist": f"New {chosen_genre} Artist",
            "song": f"Discovering {chosen_genre}",
            "confidence": 0.3,  # Low confidence for exploration
            "reasoning": f"Exploring new genre: {chosen_genre} to learn your preferences"
        }
    
    def _get_top_preferences(self, preference_dict, n=5):
        """Get top N preferences sorted by score."""
        if not preference_dict:
            return []
        
        sorted_prefs = sorted(preference_dict.items(), 
                            key=lambda x: x[1]["score"], reverse=True)
        return [pref[0] for pref in sorted_prefs[:n]]
    
    def _calculate_confidence(self, genre, artist):
        """Calculate confidence in recommendation."""
        genre_confidence = 0.5
        if genre in self.genre_preferences:
            score = self.genre_preferences[genre]["score"]
            count = self.genre_preferences[genre]["count"]
            genre_confidence = min(1.0, (score / 5.0) * (count / 10.0))
        
        artist_confidence = 0.3
        if artist in self.artist_preferences:
            score = self.artist_preferences[artist]["score"]
            artist_confidence = score / 5.0
        
        return (genre_confidence + artist_confidence) / 2
    
    def _generate_learning_summary(self, interaction):
        """Generate summary of what was learned."""
        feedback_rating = interaction["feedback"]["rating"]
        recommendation = interaction["recommendation"]
        
        summary = f"Learned from {feedback_rating}★ rating for {recommendation.get('genre', 'music')}"
        
        # Add specific learning insights
        if feedback_rating >= 4:
            summary += f" ✅ Reinforced preference"
        elif feedback_rating <= 2:
            summary += f" ❌ Noted dislike"
        else:
            summary += f" ➡️  Neutral feedback"
        
        return summary
    
    def _get_timestamp(self):
        """Simple timestamp for demo."""
        import time
        return int(time.time()) % 1000000
    
    def get_learning_stats(self):
        """Return learning statistics."""
        success_rate = (self.successful_recommendations / max(1, self.total_recommendations)) * 100
        
        return {
            "total_interactions": len(self.interaction_history),
            "success_rate": f"{success_rate:.1f}%",
            "genres_learned": len(self.genre_preferences),
            "artists_learned": len(self.artist_preferences),
            "temporal_patterns": len(self.temporal_patterns),
            "context_patterns": len(self.context_preferences),
            "exploration_rate": f"{self.exploration_rate * 100:.0f}%"
        }


def demo_learning_agent():
    """Demonstrate learning agent behavior."""
    print("=== Learning Agent Demo ===\n")
    
    # Basic learning agent from blog
    print("1. Basic Music Streaming Agent (from blog):")
    music_agent = LearningAgent()
    
    interactions = [
        ("Jazz", 4),
        ("Rock", 5),
        ("Jazz", 5),
        ("Pop", 3),
        ("Rock", 4),
        ("Classical", 5)
    ]
    
    for genre, rating in interactions:
        feedback = music_agent.update_preferences(genre, rating)
        print(f"   → {feedback}")
    
    recommendation = music_agent.recommend()
    print(f"   🎵 {recommendation}")
    
    print("\n" + "="*70 + "\n")
    
    # Advanced recommendation agent
    print("2. Advanced Learning Recommendation Agent:")
    advanced_agent = AdvancedRecommendationAgent()
    
    # Simulate learning interactions
    learning_scenarios = [
        # Morning preferences
        {"rec": {"genre": "Classical", "artist": "Bach"}, 
         "feedback": {"rating": 5, "engagement": "high"}, 
         "context": {"time_of_day": "morning", "mood": "focused"}},
        
        {"rec": {"genre": "Jazz", "artist": "Miles Davis"}, 
         "feedback": {"rating": 4, "engagement": "medium"}, 
         "context": {"time_of_day": "evening", "mood": "relaxed"}},
        
        # Workout preferences  
        {"rec": {"genre": "Electronic", "artist": "Daft Punk"}, 
         "feedback": {"rating": 5, "engagement": "high"}, 
         "context": {"time_of_day": "afternoon", "mood": "energetic"}},
        
        {"rec": {"genre": "Pop", "artist": "Taylor Swift"}, 
         "feedback": {"rating": 2, "engagement": "low"}, 
         "context": {"time_of_day": "evening", "mood": "relaxed"}},
        
        # More learning
        {"rec": {"genre": "Rock", "artist": "Queen"}, 
         "feedback": {"rating": 5, "engagement": "high"}, 
         "context": {"time_of_day": "afternoon", "mood": "energetic"}},
        
        {"rec": {"genre": "Classical", "artist": "Mozart"}, 
         "feedback": {"rating": 4, "engagement": "medium"}, 
         "context": {"time_of_day": "morning", "mood": "focused"}}
    ]
    
    print("   🧠 Learning Phase:")
    for i, scenario in enumerate(learning_scenarios, 1):
        learning_result = advanced_agent.learn_from_feedback(
            scenario["rec"], 
            scenario["feedback"], 
            scenario["context"]
        )
        print(f"      Interaction {i}: {learning_result}")
    
    print("\n   📊 Learning Statistics:")
    stats = advanced_agent.get_learning_stats()
    for key, value in stats.items():
        print(f"      {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "="*70 + "\n")
    
    # Test learned recommendations
    print("3. Testing Learned Recommendations:")
    
    test_contexts = [
        {"time_of_day": "morning", "mood": "focused"},
        {"time_of_day": "afternoon", "mood": "energetic"}, 
        {"time_of_day": "evening", "mood": "relaxed"},
        {}  # No context
    ]
    
    for i, context in enumerate(test_contexts, 1):
        recommendation = advanced_agent.make_recommendation(context)
        
        print(f"   Test {i}: Context: {context}")
        print(f"   → Recommendation ({recommendation['type']}): {recommendation['song']}")
        print(f"   → Confidence: {recommendation['confidence']:.2f}")
        print(f"   → Reasoning: {recommendation['reasoning']}\n")
    
    print("Key Insight: Learning agents improve over time by analyzing feedback")
    print("and adapting their behavior based on experience and context!")


if __name__ == "__main__":
    demo_learning_agent()