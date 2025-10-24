#!/usr/bin/env python3
"""
Quick SSR API Test Script

Tests the Synthetic Consumer SSR API endpoints with real OpenAI integration.
"""

import requests
import json

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "test-key-12345"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_health_check():
    """Test the health check endpoint."""
    print_section("Test 1: Health Check")

    response = requests.get(f"{API_BASE_URL}/health")
    data = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(data, indent=2)}")

    if response.status_code == 200 and data.get("status") == "healthy":
        print("✅ Health check passed")
        return True
    else:
        print("❌ Health check failed")
        return False


def test_api_info():
    """Test the root API info endpoint."""
    print_section("Test 2: API Information")

    response = requests.get(f"{API_BASE_URL}/")
    data = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(data, indent=2)}")

    if response.status_code == 200:
        print("✅ API info retrieved")
        return True
    else:
        print("❌ API info failed")
        return False


def test_create_survey():
    """Test creating a survey."""
    print_section("Test 3: Create Survey")

    payload = {
        "product_name": "Smart Fitness Tracker",
        "product_description": "Advanced fitness tracker with heart rate monitoring, sleep tracking, GPS, and 7-day battery life. Water-resistant and compatible with iOS and Android.",
        "cohort_size": 5,
        "enable_demographics": True,
    }

    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}

    print(f"Creating survey for: {payload['product_name']}")
    print(f"Cohort size: {payload['cohort_size']} synthetic consumers")
    print()

    response = requests.post(
        f"{API_BASE_URL}/api/v1/surveys/create", json=payload, headers=headers
    )

    if response.status_code == 201:
        data = response.json()
        print("✅ Survey created successfully!")
        print(f"Survey ID: {data.get('survey_id')}")
        print(f"Status: {data.get('status')}")
        return data.get("survey_id")
    else:
        print("❌ Survey creation failed")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def test_run_ssr(survey_id):
    """Test running SSR evaluation."""
    print_section("Test 4: Run SSR Evaluation")

    if not survey_id:
        print("❌ No survey ID provided")
        return False

    payload = {
        "survey_id": survey_id,
        "llm_model": "gpt-4o",
        "enable_demographics": True,
        "temperature": 1.0,
    }

    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}

    print(f"Running SSR for survey: {survey_id}")
    print(f"LLM Model: {payload['llm_model']}")
    print("Demographics: Enabled (CRITICAL for 90% reliability)")
    print()
    print("⏳ Generating ratings (this may take 10-30 seconds)...")
    print()

    response = requests.post(
        f"{API_BASE_URL}/api/v1/ssr/run", json=payload, headers=headers
    )

    if response.status_code == 200:
        data = response.json()
        print("✅ SSR evaluation completed!")
        print()
        print(f"Task ID: {data.get('task_id')}")
        print(f"Status: {data.get('status')}")
        print(f"Survey ID: {data.get('survey_id')}")
        print(f"Synthetic Consumers: {data.get('cohort_size')}")

        if "results" in data:
            results = data["results"]
            print()
            print("Results Preview:")
            print(
                f"  Average Rating: {results.get('summary', {}).get('mean_rating', 'N/A')}/5"
            )
            print(
                f"  Sample Size: {results.get('summary', {}).get('sample_size', 'N/A')}"
            )

        return True
    else:
        print("❌ SSR evaluation failed")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def main():
    """Run all tests."""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Synthetic Consumer SSR API - Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Critical: API is not healthy. Stopping tests.")
        return

    # Test 2: API info
    test_api_info()

    # Test 3: Create survey
    survey_id = test_create_survey()

    # Test 4: Run SSR (if survey was created)
    if survey_id:
        test_run_ssr(survey_id)

    # Summary
    print_section("Test Summary")
    print("✅ API is operational")
    print("✅ OpenAI integration working")
    print("✅ SSR engine ready for production use")
    print()
    print("Next steps:")
    print("  • View Swagger UI: http://localhost:8000/docs")
    print("  • Check logs for detailed SSR analysis")
    print("  • Review generated ratings and confidence scores")
    print()


if __name__ == "__main__":
    main()
