"""
ABOUTME: Demographic profile management with validation and serialization
ABOUTME: Defines core demographic attributes for synthetic consumer personas

This module implements the demographic schema that enables the SSR system
to achieve 90% of human test-retest reliability through demographic conditioning.

Key Demographics (from paper):
- Age: 18-80+ years (strong impact on purchase intent)
- Gender: Male, Female, Non-binary, Other
- Income Level: US household income brackets (strong impact)
- Location: City, State, Country (moderate impact for regional preferences)
- Ethnicity: US Census categories (variable impact, product-dependent)

Paper Finding: Age and income level are the most reliably replicated by LLMs.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime


@dataclass
class Location:
    """
    Geographic location information

    Attributes:
        city: City name
        state: State/province (2-letter code for US)
        country: Country name or ISO code
        region: Optional regional classification (e.g., "West Coast", "Midwest")
    """

    city: str
    state: str
    country: str = "USA"
    region: Optional[str] = None

    def __str__(self) -> str:
        """Format as readable string"""
        if self.country == "USA":
            return f"{self.city}, {self.state}, USA"
        else:
            return f"{self.city}, {self.state}, {self.country}"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "region": self.region,
        }


@dataclass
class DemographicProfile:
    """
    Complete demographic profile for synthetic consumer

    Attributes:
        age: Age in years (18-80+)
        gender: Gender identity
        income_level: Annual household income bracket
        location: Geographic location
        ethnicity: Ethnicity/race category
        id: Unique identifier (auto-generated)
        created_at: Creation timestamp
        metadata: Additional attributes for extensibility
    """

    age: int
    gender: str
    income_level: str
    location: Location
    ethnicity: str
    id: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize profile"""
        # Validate age
        if not 18 <= self.age <= 120:
            raise ValueError(f"Age must be 18-120, got {self.age}")

        # Validate gender
        valid_genders = ["Male", "Female", "Non-binary", "Other"]
        if self.gender not in valid_genders:
            raise ValueError(
                f"Gender must be one of {valid_genders}, got '{self.gender}'"
            )

        # Validate income level format
        if not self._validate_income_level(self.income_level):
            raise ValueError(f"Invalid income level format: '{self.income_level}'")

        # Validate ethnicity
        valid_ethnicities = [
            "White",
            "Black or African American",
            "Hispanic or Latino",
            "Asian",
            "Native American or Alaska Native",
            "Native Hawaiian or Pacific Islander",
            "Two or More Races",
            "Other",
        ]
        if self.ethnicity not in valid_ethnicities:
            raise ValueError(
                f"Ethnicity must be one of {valid_ethnicities}, got '{self.ethnicity}'"
            )

        # Auto-generate ID if not provided
        if self.id is None:
            self.id = self._generate_id()

        # Set creation timestamp if not provided
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def _validate_income_level(self, income: str) -> bool:
        """Validate income level format"""
        # Standard US income brackets
        valid_brackets = [
            "Less than $25,000",
            "$25,000-$49,999",
            "$50,000-$74,999",
            "$75,000-$99,999",
            "$100,000-$149,999",
            "$150,000 or more",
        ]
        return income in valid_brackets

    def _generate_id(self) -> str:
        """Generate unique identifier"""
        import hashlib

        # Create hash from demographic attributes
        data = f"{self.age}_{self.gender}_{self.income_level}_{self.location}_{self.ethnicity}_{datetime.now().timestamp()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for API/storage

        Returns:
            Dictionary representation with all attributes
        """
        return {
            "id": self.id,
            "age": self.age,
            "gender": self.gender,
            "income_level": self.income_level,
            "location": self.location.to_dict()
            if isinstance(self.location, Location)
            else self.location,
            "ethnicity": self.ethnicity,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DemographicProfile":
        """
        Create profile from dictionary

        Args:
            data: Dictionary with demographic attributes

        Returns:
            DemographicProfile instance
        """
        # Handle location
        location_data = data.get("location")
        if isinstance(location_data, dict):
            location = Location(**location_data)
        elif isinstance(location_data, Location):
            location = location_data
        else:
            location = Location(city="Unknown", state="Unknown", country="USA")

        return cls(
            age=data["age"],
            gender=data["gender"],
            income_level=data["income_level"],
            location=location,
            ethnicity=data["ethnicity"],
            id=data.get("id"),
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DemographicProfile":
        """Create profile from JSON string"""
        return cls.from_dict(json.loads(json_str))

    def get_age_group(self) -> str:
        """
        Get age group category

        Returns:
            Age group string (e.g., "18-24", "25-34")
        """
        if self.age < 25:
            return "18-24"
        elif self.age < 35:
            return "25-34"
        elif self.age < 45:
            return "35-44"
        elif self.age < 55:
            return "45-54"
        elif self.age < 65:
            return "55-64"
        else:
            return "65+"

    def get_income_group(self) -> str:
        """
        Get simplified income group

        Returns:
            Income group: "low", "middle", "high"
        """
        if "Less than" in self.income_level or "$25,000" in self.income_level:
            return "low"
        elif "$50,000" in self.income_level or "$75,000" in self.income_level:
            return "middle"
        else:
            return "high"

    def __str__(self) -> str:
        """Human-readable representation"""
        return (
            f"DemographicProfile(age={self.age}, gender={self.gender}, "
            f"income={self.income_level}, location={self.location}, "
            f"ethnicity={self.ethnicity})"
        )

    def __repr__(self) -> str:
        """Developer representation"""
        return self.__str__()


# Predefined demographic profiles for testing
class DemographicProfiles:
    """Collection of predefined demographic profiles for testing"""

    @staticmethod
    def young_tech_professional() -> DemographicProfile:
        """Young tech professional in San Francisco"""
        return DemographicProfile(
            age=28,
            gender="Female",
            income_level="$100,000-$149,999",
            location=Location(city="San Francisco", state="CA", region="West Coast"),
            ethnicity="Asian",
            metadata={
                "occupation": "Software Engineer",
                "interests": ["technology", "fitness"],
            },
        )

    @staticmethod
    def middle_aged_family() -> DemographicProfile:
        """Middle-aged parent in suburban area"""
        return DemographicProfile(
            age=42,
            gender="Male",
            income_level="$75,000-$99,999",
            location=Location(city="Columbus", state="OH", region="Midwest"),
            ethnicity="White",
            metadata={"occupation": "Teacher", "family_size": 4},
        )

    @staticmethod
    def retired_senior() -> DemographicProfile:
        """Retired senior citizen"""
        return DemographicProfile(
            age=68,
            gender="Female",
            income_level="$50,000-$74,999",
            location=Location(city="Miami", state="FL", region="Southeast"),
            ethnicity="Hispanic or Latino",
            metadata={"occupation": "Retired", "interests": ["travel", "gardening"]},
        )

    @staticmethod
    def young_student() -> DemographicProfile:
        """College student"""
        return DemographicProfile(
            age=21,
            gender="Non-binary",
            income_level="Less than $25,000",
            location=Location(city="Boston", state="MA", region="Northeast"),
            ethnicity="Two or More Races",
            metadata={"occupation": "Student", "education": "College"},
        )

    @staticmethod
    def all_test_profiles() -> list[DemographicProfile]:
        """Get all test profiles"""
        return [
            DemographicProfiles.young_tech_professional(),
            DemographicProfiles.middle_aged_family(),
            DemographicProfiles.retired_senior(),
            DemographicProfiles.young_student(),
        ]


# Example usage and testing
if __name__ == "__main__":
    print("Demographic Profile System Testing")
    print("=" * 60)

    # Test profile creation
    print("\n1. Creating demographic profile:")
    profile = DemographicProfile(
        age=28,
        gender="Female",
        income_level="$50,000-$74,999",
        location=Location(city="San Francisco", state="CA"),
        ethnicity="Asian",
    )
    print(profile)
    print(f"ID: {profile.id}")
    print(f"Age Group: {profile.get_age_group()}")
    print(f"Income Group: {profile.get_income_group()}")

    # Test serialization
    print("\n2. Testing serialization:")
    profile_dict = profile.to_dict()
    print(f"Dictionary keys: {list(profile_dict.keys())}")

    profile_json = profile.to_json()
    print(f"JSON length: {len(profile_json)} characters")

    # Test deserialization
    print("\n3. Testing deserialization:")
    restored_profile = DemographicProfile.from_json(profile_json)
    print(f"Restored: {restored_profile}")
    print(f"Match: {restored_profile.age == profile.age}")

    # Test predefined profiles
    print("\n4. Testing predefined profiles:")
    test_profiles = DemographicProfiles.all_test_profiles()
    for i, p in enumerate(test_profiles, 1):
        print(f"\nProfile {i}:")
        print(f"  Age: {p.age} ({p.get_age_group()})")
        print(f"  Gender: {p.gender}")
        print(f"  Income: {p.income_level} ({p.get_income_group()})")
        print(f"  Location: {p.location}")
        print(f"  Ethnicity: {p.ethnicity}")

    # Test validation
    print("\n5. Testing validation:")
    try:
        invalid = DemographicProfile(
            age=15,  # Too young
            gender="Female",
            income_level="$50,000-$74,999",
            location=Location(city="Test", state="TX"),
            ethnicity="Asian",
        )
    except ValueError as e:
        print(f"✓ Caught invalid age: {e}")

    try:
        invalid = DemographicProfile(
            age=25,
            gender="Invalid",  # Invalid gender
            income_level="$50,000-$74,999",
            location=Location(city="Test", state="TX"),
            ethnicity="Asian",
        )
    except ValueError as e:
        print(f"✓ Caught invalid gender: {e}")

    print("\n" + "=" * 60)
    print("Demographic Profile testing complete")
