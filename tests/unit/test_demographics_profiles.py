"""
ABOUTME: Unit tests for demographic profile system using real component instances
ABOUTME: Tests Location, DemographicProfile, and DemographicProfiles factory classes

This module tests the demographic profile system following project testing standards:
- Location: Geographic location dataclass and methods
- DemographicProfile: Main profile with validation and serialization
- DemographicProfiles: Factory for predefined test profiles
- All validation logic for age, gender, income, ethnicity
- Serialization/deserialization (to_dict, from_dict, to_json, from_json)
- Helper methods (get_age_group, get_income_group)

All tests use real component instances following the project testing standards.
"""

import pytest
import json
from datetime import datetime

from src.demographics.profiles import Location, DemographicProfile, DemographicProfiles


class TestLocation:
    """Test Location dataclass and methods"""

    def test_location_creation(self):
        """Test creating a Location instance"""
        location = Location(
            city="San Francisco", state="CA", country="USA", region="West Coast"
        )

        assert location.city == "San Francisco"
        assert location.state == "CA"
        assert location.country == "USA"
        assert location.region == "West Coast"

    def test_location_defaults(self):
        """Test Location default values"""
        location = Location(city="Boston", state="MA")

        assert location.country == "USA"  # Default
        assert location.region is None  # Default

    def test_location_str_usa(self):
        """Test Location string formatting for USA"""
        location = Location(city="New York", state="NY", country="USA")

        assert str(location) == "New York, NY, USA"

    def test_location_str_international(self):
        """Test Location string formatting for non-USA"""
        location = Location(city="Toronto", state="ON", country="Canada")

        assert str(location) == "Toronto, ON, Canada"

    def test_location_to_dict(self):
        """Test Location to_dict serialization"""
        location = Location(
            city="Seattle", state="WA", country="USA", region="Pacific Northwest"
        )

        data = location.to_dict()

        assert data["city"] == "Seattle"
        assert data["state"] == "WA"
        assert data["country"] == "USA"
        assert data["region"] == "Pacific Northwest"

    def test_location_to_dict_with_none_region(self):
        """Test Location to_dict with None region"""
        location = Location(city="Austin", state="TX")
        data = location.to_dict()

        assert data["region"] is None


class TestDemographicProfileValidation:
    """Test DemographicProfile validation logic"""

    def test_valid_profile_creation(self):
        """Test creating a valid DemographicProfile"""
        location = Location(city="Chicago", state="IL")

        profile = DemographicProfile(
            age=35,
            gender="Female",
            income_level="$75,000-$99,999",
            location=location,
            ethnicity="White",
        )

        assert profile.age == 35
        assert profile.gender == "Female"
        assert profile.income_level == "$75,000-$99,999"
        assert profile.location == location
        assert profile.ethnicity == "White"

    def test_age_validation_minimum(self):
        """Test age validation rejects age < 18"""
        location = Location(city="Test", state="TX")

        with pytest.raises(ValueError, match="Age must be 18-120"):
            DemographicProfile(
                age=17,  # Too young
                gender="Female",
                income_level="$50,000-$74,999",
                location=location,
                ethnicity="Asian",
            )

    def test_age_validation_maximum(self):
        """Test age validation rejects age > 120"""
        location = Location(city="Test", state="TX")

        with pytest.raises(ValueError, match="Age must be 18-120"):
            DemographicProfile(
                age=121,  # Too old
                gender="Male",
                income_level="$50,000-$74,999",
                location=location,
                ethnicity="White",
            )

    def test_age_validation_boundaries(self):
        """Test age validation boundaries (18 and 120 should be valid)"""
        location = Location(city="Test", state="TX")

        # Age 18 should be valid
        profile_18 = DemographicProfile(
            age=18,
            gender="Male",
            income_level="$25,000-$49,999",
            location=location,
            ethnicity="White",
        )
        assert profile_18.age == 18

        # Age 120 should be valid
        profile_120 = DemographicProfile(
            age=120,
            gender="Female",
            income_level="$50,000-$74,999",
            location=location,
            ethnicity="Asian",
        )
        assert profile_120.age == 120

    def test_gender_validation(self):
        """Test gender validation with valid values"""
        location = Location(city="Test", state="TX")

        valid_genders = ["Male", "Female", "Non-binary", "Other"]

        for gender in valid_genders:
            profile = DemographicProfile(
                age=30,
                gender=gender,
                income_level="$50,000-$74,999",
                location=location,
                ethnicity="White",
            )
            assert profile.gender == gender

    def test_gender_validation_invalid(self):
        """Test gender validation rejects invalid values"""
        location = Location(city="Test", state="TX")

        with pytest.raises(ValueError, match="Gender must be one of"):
            DemographicProfile(
                age=30,
                gender="InvalidGender",
                income_level="$50,000-$74,999",
                location=location,
                ethnicity="White",
            )

    def test_income_level_validation_valid_brackets(self):
        """Test income level validation with all valid brackets"""
        location = Location(city="Test", state="TX")

        valid_brackets = [
            "Less than $25,000",
            "$25,000-$49,999",
            "$50,000-$74,999",
            "$75,000-$99,999",
            "$100,000-$149,999",
            "$150,000 or more",
        ]

        for bracket in valid_brackets:
            profile = DemographicProfile(
                age=30,
                gender="Female",
                income_level=bracket,
                location=location,
                ethnicity="Asian",
            )
            assert profile.income_level == bracket

    def test_income_level_validation_invalid(self):
        """Test income level validation rejects invalid format"""
        location = Location(city="Test", state="TX")

        with pytest.raises(ValueError, match="Invalid income level format"):
            DemographicProfile(
                age=30,
                gender="Female",
                income_level="$50,000",  # Invalid format
                location=location,
                ethnicity="White",
            )

    def test_ethnicity_validation_valid(self):
        """Test ethnicity validation with all valid values"""
        location = Location(city="Test", state="TX")

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

        for ethnicity in valid_ethnicities:
            profile = DemographicProfile(
                age=30,
                gender="Male",
                income_level="$50,000-$74,999",
                location=location,
                ethnicity=ethnicity,
            )
            assert profile.ethnicity == ethnicity

    def test_ethnicity_validation_invalid(self):
        """Test ethnicity validation rejects invalid values"""
        location = Location(city="Test", state="TX")

        with pytest.raises(ValueError, match="Ethnicity must be one of"):
            DemographicProfile(
                age=30,
                gender="Male",
                income_level="$50,000-$74,999",
                location=location,
                ethnicity="InvalidEthnicity",
            )


class TestDemographicProfileAutoGeneration:
    """Test auto-generated fields (id, created_at)"""

    def test_id_auto_generation(self):
        """Test that ID is auto-generated when not provided"""
        location = Location(city="Test", state="TX")

        profile = DemographicProfile(
            age=28,
            gender="Female",
            income_level="$50,000-$74,999",
            location=location,
            ethnicity="Asian",
        )

        assert profile.id is not None
        assert isinstance(profile.id, str)
        assert len(profile.id) == 12  # MD5 hash truncated to 12 chars

    def test_id_provided(self):
        """Test that provided ID is preserved"""
        location = Location(city="Test", state="TX")

        profile = DemographicProfile(
            age=28,
            gender="Female",
            income_level="$50,000-$74,999",
            location=location,
            ethnicity="Asian",
            id="custom_id_123",
        )

        assert profile.id == "custom_id_123"

    def test_created_at_auto_generation(self):
        """Test that created_at is auto-generated when not provided"""
        location = Location(city="Test", state="TX")

        before = datetime.now()

        profile = DemographicProfile(
            age=28,
            gender="Female",
            income_level="$50,000-$74,999",
            location=location,
            ethnicity="Asian",
        )

        after = datetime.now()

        assert profile.created_at is not None

        # Parse and verify timestamp is between before and after
        created_dt = datetime.fromisoformat(profile.created_at)
        assert before <= created_dt <= after

    def test_created_at_provided(self):
        """Test that provided created_at is preserved"""
        location = Location(city="Test", state="TX")
        custom_timestamp = "2024-01-15T10:30:00"

        profile = DemographicProfile(
            age=28,
            gender="Female",
            income_level="$50,000-$74,999",
            location=location,
            ethnicity="Asian",
            created_at=custom_timestamp,
        )

        assert profile.created_at == custom_timestamp

    def test_id_uniqueness(self):
        """Test that different profiles get different IDs"""
        location = Location(city="Test", state="TX")

        profile1 = DemographicProfile(
            age=25,
            gender="Male",
            income_level="$50,000-$74,999",
            location=location,
            ethnicity="White",
        )

        profile2 = DemographicProfile(
            age=35,
            gender="Female",
            income_level="$75,000-$99,999",
            location=location,
            ethnicity="Asian",
        )

        assert profile1.id != profile2.id


class TestDemographicProfileSerialization:
    """Test serialization and deserialization methods"""

    def test_to_dict_complete(self):
        """Test to_dict returns all attributes"""
        location = Location(city="Boston", state="MA", region="Northeast")

        profile = DemographicProfile(
            age=42,
            gender="Male",
            income_level="$75,000-$99,999",
            location=location,
            ethnicity="White",
            id="test_id_123",
            created_at="2024-01-15T10:00:00",
            metadata={"occupation": "Teacher"},
        )

        data = profile.to_dict()

        assert data["id"] == "test_id_123"
        assert data["age"] == 42
        assert data["gender"] == "Male"
        assert data["income_level"] == "$75,000-$99,999"
        assert data["location"]["city"] == "Boston"
        assert data["location"]["state"] == "MA"
        assert data["location"]["region"] == "Northeast"
        assert data["ethnicity"] == "White"
        assert data["created_at"] == "2024-01-15T10:00:00"
        assert data["metadata"]["occupation"] == "Teacher"

    def test_to_json(self):
        """Test to_json serialization"""
        location = Location(city="Seattle", state="WA")

        profile = DemographicProfile(
            age=28,
            gender="Non-binary",
            income_level="$100,000-$149,999",
            location=location,
            ethnicity="Asian",
        )

        json_str = profile.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["age"] == 28
        assert parsed["gender"] == "Non-binary"
        assert parsed["income_level"] == "$100,000-$149,999"

    def test_from_dict_complete(self):
        """Test from_dict reconstruction"""
        data = {
            "id": "test_id_456",
            "age": 50,
            "gender": "Female",
            "income_level": "$150,000 or more",
            "location": {
                "city": "Austin",
                "state": "TX",
                "country": "USA",
                "region": "Southwest",
            },
            "ethnicity": "Hispanic or Latino",
            "created_at": "2024-02-20T15:30:00",
            "metadata": {"interests": ["travel", "music"]},
        }

        profile = DemographicProfile.from_dict(data)

        assert profile.id == "test_id_456"
        assert profile.age == 50
        assert profile.gender == "Female"
        assert profile.income_level == "$150,000 or more"
        assert profile.location.city == "Austin"
        assert profile.location.state == "TX"
        assert profile.location.region == "Southwest"
        assert profile.ethnicity == "Hispanic or Latino"
        assert profile.created_at == "2024-02-20T15:30:00"
        assert profile.metadata["interests"] == ["travel", "music"]

    def test_from_dict_with_location_object(self):
        """Test from_dict with Location object instead of dict"""
        location = Location(city="Miami", state="FL")

        data = {
            "age": 68,
            "gender": "Female",
            "income_level": "$50,000-$74,999",
            "location": location,  # Already a Location object
            "ethnicity": "Hispanic or Latino",
        }

        profile = DemographicProfile.from_dict(data)

        assert profile.location.city == "Miami"
        assert profile.location.state == "FL"

    def test_from_json(self):
        """Test from_json deserialization"""
        json_str = """
        {
            "age": 32,
            "gender": "Male",
            "income_level": "$75,000-$99,999",
            "location": {
                "city": "Denver",
                "state": "CO",
                "country": "USA"
            },
            "ethnicity": "Two or More Races"
        }
        """

        profile = DemographicProfile.from_json(json_str)

        assert profile.age == 32
        assert profile.gender == "Male"
        assert profile.location.city == "Denver"
        assert profile.ethnicity == "Two or More Races"

    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization preserve data"""
        original_location = Location(
            city="Portland", state="OR", region="Pacific Northwest"
        )

        original = DemographicProfile(
            age=29,
            gender="Female",
            income_level="$100,000-$149,999",
            location=original_location,
            ethnicity="White",
            metadata={"occupation": "Designer"},
        )

        # Round trip through dict
        data = original.to_dict()
        restored = DemographicProfile.from_dict(data)

        assert restored.age == original.age
        assert restored.gender == original.gender
        assert restored.income_level == original.income_level
        assert restored.location.city == original.location.city
        assert restored.ethnicity == original.ethnicity
        assert restored.metadata == original.metadata


class TestDemographicProfileHelperMethods:
    """Test helper methods (get_age_group, get_income_group)"""

    def test_get_age_group_18_24(self):
        """Test age group categorization for 18-24"""
        location = Location(city="Test", state="TX")

        for age in [18, 20, 24]:
            profile = DemographicProfile(
                age=age,
                gender="Male",
                income_level="$25,000-$49,999",
                location=location,
                ethnicity="White",
            )
            assert profile.get_age_group() == "18-24"

    def test_get_age_group_25_34(self):
        """Test age group categorization for 25-34"""
        location = Location(city="Test", state="TX")

        for age in [25, 30, 34]:
            profile = DemographicProfile(
                age=age,
                gender="Female",
                income_level="$50,000-$74,999",
                location=location,
                ethnicity="Asian",
            )
            assert profile.get_age_group() == "25-34"

    def test_get_age_group_35_44(self):
        """Test age group categorization for 35-44"""
        location = Location(city="Test", state="TX")

        profile = DemographicProfile(
            age=40,
            gender="Male",
            income_level="$75,000-$99,999",
            location=location,
            ethnicity="White",
        )
        assert profile.get_age_group() == "35-44"

    def test_get_age_group_45_54(self):
        """Test age group categorization for 45-54"""
        location = Location(city="Test", state="TX")

        profile = DemographicProfile(
            age=50,
            gender="Female",
            income_level="$100,000-$149,999",
            location=location,
            ethnicity="Black or African American",
        )
        assert profile.get_age_group() == "45-54"

    def test_get_age_group_55_64(self):
        """Test age group categorization for 55-64"""
        location = Location(city="Test", state="TX")

        profile = DemographicProfile(
            age=60,
            gender="Male",
            income_level="$150,000 or more",
            location=location,
            ethnicity="Asian",
        )
        assert profile.get_age_group() == "55-64"

    def test_get_age_group_65_plus(self):
        """Test age group categorization for 65+"""
        location = Location(city="Test", state="TX")

        for age in [65, 70, 80, 100]:
            profile = DemographicProfile(
                age=age,
                gender="Female",
                income_level="$50,000-$74,999",
                location=location,
                ethnicity="White",
            )
            assert profile.get_age_group() == "65+"

    def test_get_income_group_low(self):
        """Test income group categorization for low income"""
        location = Location(city="Test", state="TX")

        low_brackets = ["Less than $25,000", "$25,000-$49,999"]

        for bracket in low_brackets:
            profile = DemographicProfile(
                age=25,
                gender="Male",
                income_level=bracket,
                location=location,
                ethnicity="White",
            )
            assert profile.get_income_group() == "low"

    def test_get_income_group_middle(self):
        """Test income group categorization for middle income"""
        location = Location(city="Test", state="TX")

        middle_brackets = ["$50,000-$74,999", "$75,000-$99,999"]

        for bracket in middle_brackets:
            profile = DemographicProfile(
                age=35,
                gender="Female",
                income_level=bracket,
                location=location,
                ethnicity="Asian",
            )
            assert profile.get_income_group() == "middle"

    def test_get_income_group_high(self):
        """Test income group categorization for high income"""
        location = Location(city="Test", state="TX")

        high_brackets = ["$100,000-$149,999", "$150,000 or more"]

        for bracket in high_brackets:
            profile = DemographicProfile(
                age=45,
                gender="Male",
                income_level=bracket,
                location=location,
                ethnicity="White",
            )
            assert profile.get_income_group() == "high"


class TestDemographicProfileStringRepresentation:
    """Test string representation methods"""

    def test_str_representation(self):
        """Test __str__ method"""
        location = Location(city="Chicago", state="IL")

        profile = DemographicProfile(
            age=38,
            gender="Female",
            income_level="$75,000-$99,999",
            location=location,
            ethnicity="Black or African American",
        )

        str_repr = str(profile)

        assert "age=38" in str_repr
        assert "gender=Female" in str_repr
        assert "income=$75,000-$99,999" in str_repr
        assert "Chicago, IL, USA" in str_repr
        assert "ethnicity=Black or African American" in str_repr

    def test_repr_representation(self):
        """Test __repr__ method"""
        location = Location(city="Boston", state="MA")

        profile = DemographicProfile(
            age=28,
            gender="Non-binary",
            income_level="$50,000-$74,999",
            location=location,
            ethnicity="Two or More Races",
        )

        # repr should equal str
        assert repr(profile) == str(profile)


class TestDemographicProfilesFactory:
    """Test DemographicProfiles factory class"""

    def test_young_tech_professional(self):
        """Test young_tech_professional factory method"""
        profile = DemographicProfiles.young_tech_professional()

        assert profile.age == 28
        assert profile.gender == "Female"
        assert profile.income_level == "$100,000-$149,999"
        assert profile.location.city == "San Francisco"
        assert profile.location.state == "CA"
        assert profile.location.region == "West Coast"
        assert profile.ethnicity == "Asian"
        assert profile.metadata["occupation"] == "Software Engineer"

    def test_middle_aged_family(self):
        """Test middle_aged_family factory method"""
        profile = DemographicProfiles.middle_aged_family()

        assert profile.age == 42
        assert profile.gender == "Male"
        assert profile.income_level == "$75,000-$99,999"
        assert profile.location.city == "Columbus"
        assert profile.location.state == "OH"
        assert profile.location.region == "Midwest"
        assert profile.ethnicity == "White"
        assert profile.metadata["occupation"] == "Teacher"
        assert profile.metadata["family_size"] == 4

    def test_retired_senior(self):
        """Test retired_senior factory method"""
        profile = DemographicProfiles.retired_senior()

        assert profile.age == 68
        assert profile.gender == "Female"
        assert profile.income_level == "$50,000-$74,999"
        assert profile.location.city == "Miami"
        assert profile.location.state == "FL"
        assert profile.location.region == "Southeast"
        assert profile.ethnicity == "Hispanic or Latino"
        assert profile.metadata["occupation"] == "Retired"

    def test_young_student(self):
        """Test young_student factory method"""
        profile = DemographicProfiles.young_student()

        assert profile.age == 21
        assert profile.gender == "Non-binary"
        assert profile.income_level == "Less than $25,000"
        assert profile.location.city == "Boston"
        assert profile.location.state == "MA"
        assert profile.location.region == "Northeast"
        assert profile.ethnicity == "Two or More Races"
        assert profile.metadata["occupation"] == "Student"

    def test_all_test_profiles(self):
        """Test all_test_profiles returns all 4 profiles"""
        profiles = DemographicProfiles.all_test_profiles()

        assert len(profiles) == 4

        # Verify all profiles are valid DemographicProfile instances
        for profile in profiles:
            assert isinstance(profile, DemographicProfile)
            assert profile.id is not None
            assert profile.created_at is not None

    def test_factory_profiles_are_valid(self):
        """Test that all factory profiles pass validation"""
        profiles = DemographicProfiles.all_test_profiles()

        # All profiles should be created successfully (validation passed)
        assert all(isinstance(p, DemographicProfile) for p in profiles)

        # Verify they have distinct characteristics
        ages = [p.age for p in profiles]
        genders = [p.gender for p in profiles]

        # Should have variety in demographics
        assert len(set(ages)) == 4  # All different ages
        assert len(set(genders)) >= 3  # At least 3 different genders


class TestDemographicProfileMetadata:
    """Test metadata field functionality"""

    def test_metadata_default_empty(self):
        """Test that metadata defaults to empty dict"""
        location = Location(city="Test", state="TX")

        profile = DemographicProfile(
            age=30,
            gender="Male",
            income_level="$50,000-$74,999",
            location=location,
            ethnicity="White",
        )

        assert profile.metadata == {}

    def test_metadata_custom_values(self):
        """Test setting custom metadata"""
        location = Location(city="Test", state="TX")

        profile = DemographicProfile(
            age=30,
            gender="Female",
            income_level="$100,000-$149,999",
            location=location,
            ethnicity="Asian",
            metadata={
                "occupation": "Data Scientist",
                "education": "PhD",
                "interests": ["AI", "hiking"],
            },
        )

        assert profile.metadata["occupation"] == "Data Scientist"
        assert profile.metadata["education"] == "PhD"
        assert "AI" in profile.metadata["interests"]

    def test_metadata_preserved_in_serialization(self):
        """Test that metadata is preserved through serialization"""
        location = Location(city="Test", state="TX")

        original = DemographicProfile(
            age=35,
            gender="Male",
            income_level="$75,000-$99,999",
            location=location,
            ethnicity="White",
            metadata={"key": "value"},
        )

        # Round trip
        restored = DemographicProfile.from_dict(original.to_dict())

        assert restored.metadata == {"key": "value"}
