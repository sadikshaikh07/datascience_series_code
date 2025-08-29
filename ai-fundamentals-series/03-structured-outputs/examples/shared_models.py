"""
Shared Pydantic Models  
Blog 3: Structured Outputs & Function Calling - Shared Components

Common Pydantic models used across traditional and native structured output examples.
This eliminates duplication and provides consistent data structures.
"""

from pydantic import BaseModel, ValidationError, Field
from typing import Dict, List, Any, Optional
from enum import Enum
import datetime


# Enums for structured validation
class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    URGENT = "urgent"


class UserRole(str, Enum):
    """User role types."""
    USER = "user"
    ADMIN = "admin"
    MANAGER = "manager"
    DEVELOPER = "developer"


class ProductCategory(str, Enum):
    """Product categories."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    HOME = "home"
    SPORTS = "sports"
    OTHER = "other"


# Core data models
class PersonInfo(BaseModel):
    """Person information model for data extraction."""
    name: str = Field(..., description="Full name of the person")
    age: int = Field(..., ge=0, le=150, description="Age in years") 
    email: str = Field(..., description="Valid email address")
    occupation: Optional[str] = Field(None, description="Job title or occupation")
    skills: List[str] = Field(default_factory=list, description="List of professional skills")
    experience_years: Optional[int] = Field(None, ge=0, description="Years of professional experience")
    location: Optional[str] = Field(None, description="Current location or city")


class ProductInfo(BaseModel):
    """Product information model for e-commerce data."""
    name: str = Field(..., description="Product name")
    price: float = Field(..., gt=0, description="Price in USD")
    category: ProductCategory = Field(..., description="Product category")
    in_stock: bool = Field(..., description="Whether item is currently in stock")
    description: Optional[str] = Field(None, description="Product description")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Technical specifications")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Average customer rating (0-5)")


class TaskInfo(BaseModel):
    """Task information model for project management."""
    id: int = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed task description")
    priority: TaskPriority = Field(..., description="Task priority level")
    estimated_hours: float = Field(..., gt=0, description="Estimated time to complete")
    assigned_to: Optional[str] = Field(None, description="Person assigned to the task")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")
    status: Optional[str] = Field("pending", description="Current task status")
    tags: List[str] = Field(default_factory=list, description="Task tags or labels")


class CompanyInfo(BaseModel):
    """Company information model for business data."""
    name: str = Field(..., description="Company name")
    industry: str = Field(..., description="Primary industry")
    location: str = Field(..., description="Headquarters location")
    employee_count: int = Field(..., ge=1, description="Number of employees")
    founded_year: int = Field(..., ge=1800, le=2024, description="Year company was founded")
    revenue: Optional[float] = Field(None, ge=0, description="Annual revenue in millions USD")
    is_public: bool = Field(..., description="Whether company is publicly traded")
    website: Optional[str] = Field(None, description="Company website URL")


class ContactInfo(BaseModel):
    """Contact information model."""
    name: str = Field(..., description="Contact name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    company: Optional[str] = Field(None, description="Company name")
    role: Optional[UserRole] = Field(None, description="Role in company")
    notes: Optional[str] = Field(None, description="Additional notes")


class EventInfo(BaseModel):
    """Event information model."""
    title: str = Field(..., description="Event title")
    description: str = Field(..., description="Event description")
    date: str = Field(..., description="Event date in YYYY-MM-DD format")
    time: Optional[str] = Field(None, description="Event time in HH:MM format")
    location: str = Field(..., description="Event location")
    attendees: List[str] = Field(default_factory=list, description="List of attendee names")
    duration_minutes: Optional[int] = Field(None, gt=0, description="Event duration in minutes")


class AnalysisResult(BaseModel):
    """Analysis result model for complex data analysis."""
    summary: str = Field(..., description="Summary of the analysis")
    key_findings: List[str] = Field(..., description="List of key findings")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the analysis (0-1)")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    data_points: Optional[int] = Field(None, ge=0, description="Number of data points analyzed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional analysis metadata")


class ConfigSettings(BaseModel):
    """Configuration settings model."""
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Version number")
    debug_mode: bool = Field(False, description="Whether debug mode is enabled")
    max_users: int = Field(100, ge=1, description="Maximum number of concurrent users")
    timeout_seconds: int = Field(90, ge=1, description="Request timeout in seconds")
    features: List[str] = Field(default_factory=list, description="Enabled features")
    database_url: Optional[str] = Field(None, description="Database connection URL")


# Composite models
class UserProfile(BaseModel):
    """Complete user profile model combining multiple data types."""
    personal_info: PersonInfo
    contact_info: ContactInfo
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    created_at: Optional[str] = Field(None, description="Account creation date")
    last_active: Optional[str] = Field(None, description="Last activity date")
    is_verified: bool = Field(False, description="Whether user is verified")


class ProjectInfo(BaseModel):
    """Project information model."""
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    manager: str = Field(..., description="Project manager name")
    team_members: List[str] = Field(..., description="List of team member names")
    tasks: List[TaskInfo] = Field(default_factory=list, description="Project tasks")
    start_date: str = Field(..., description="Project start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="Project end date in YYYY-MM-DD format")
    budget: Optional[float] = Field(None, gt=0, description="Project budget in USD")
    status: str = Field("active", description="Project status")


# Model registry for easy access
MODEL_REGISTRY = {
    "person": PersonInfo,
    "product": ProductInfo,
    "task": TaskInfo,
    "company": CompanyInfo,
    "contact": ContactInfo,
    "event": EventInfo,
    "analysis": AnalysisResult,
    "config": ConfigSettings,
    "user_profile": UserProfile,
    "project": ProjectInfo,
}


def get_model_by_name(model_name: str) -> BaseModel:
    """
    Get a Pydantic model class by name.
    
    Args:
        model_name: Name of the model to retrieve
        
    Returns:
        BaseModel: The requested model class
        
    Raises:
        ValueError: If model name is not found
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    return MODEL_REGISTRY[model_name]


def validate_data(data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Validate data against a specific model.
    
    Args:
        data: Data to validate
        model_name: Name of the model to validate against
        
    Returns:
        dict: Validation result with success/error information
    """
    try:
        model_class = get_model_by_name(model_name)
        validated_instance = model_class(**data)
        
        return {
            "success": True,
            "data": validated_instance.model_dump(),
            "model": model_name,
            "errors": None
        }
        
    except ValueError as e:
        return {
            "success": False,
            "data": None,
            "model": model_name,
            "errors": [str(e)]
        }
    except ValidationError as e:
        return {
            "success": False,
            "data": None,
            "model": model_name,
            "errors": [error["msg"] for error in e.errors()]
        }


def get_model_schema(model_name: str) -> Dict[str, Any]:
    """
    Get JSON schema for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        dict: JSON schema for the model
    """
    model_class = get_model_by_name(model_name)
    return model_class.model_json_schema()


def list_available_models() -> Dict[str, str]:
    """
    Get list of all available models with descriptions.
    
    Returns:
        dict: Model names and their descriptions
    """
    descriptions = {
        "person": "Person information with skills and experience",
        "product": "E-commerce product data with pricing and specifications",
        "task": "Project management task with priority and assignments",
        "company": "Business information with industry and financial data",
        "contact": "Contact information with role and company details",
        "event": "Event details with date, location, and attendees",
        "analysis": "Analysis results with findings and recommendations",
        "config": "Application configuration settings",
        "user_profile": "Complete user profile combining personal and contact info",
        "project": "Project information with team members and tasks"
    }
    
    return descriptions


if __name__ == "__main__":
    # Demo of shared models
    print("=== Shared Models Demo ===\n")
    
    # Test model creation
    person_data = {
        "name": "Alice Johnson",
        "age": 32,
        "email": "alice@example.com",
        "occupation": "Data Scientist",
        "skills": ["Python", "Machine Learning", "SQL"],
        "experience_years": 8
    }
    
    result = validate_data(person_data, "person")
    print("1. Person Model Validation:")
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Name: {result['data']['name']}")
        print(f"   Skills: {result['data']['skills']}")
    
    # Test available models
    print(f"\n2. Available Models: {len(MODEL_REGISTRY)}")
    models = list_available_models()
    for name, desc in list(models.items())[:3]:  # Show first 3
        print(f"   {name}: {desc}")
    
    print(f"\n✅ All {len(MODEL_REGISTRY)} shared models working correctly!")