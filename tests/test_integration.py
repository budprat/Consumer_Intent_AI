#!/usr/bin/env python3
"""
Test script for SSR integration - creates survey, executes it, monitors progress
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def pretty_print(data, title=""):
    """Pretty print JSON data"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print("=" * 60)
    print(json.dumps(data, indent=2))


def create_survey():
    """Create a test survey"""
    print("\n🔧 Creating test survey...")
    response = requests.post(
        f"{BASE_URL}/api/v1/surveys",
        json={
            "product_name": "Tesla Model 3",
            "product_description": "An electric sedan with autopilot features and long range battery",
            "temperature": 1.0,
            "enable_demographics": True,
        },
    )
    response.raise_for_status()
    survey = response.json()
    pretty_print(survey, "✅ Survey Created")
    return survey["survey_id"]


def execute_survey(survey_id):
    """Execute survey with SSR"""
    print(f"\n🚀 Executing survey {survey_id}...")
    response = requests.post(
        f"{BASE_URL}/api/v1/surveys/{survey_id}/execute",
        json={
            "llm_model": "gpt-4o",  # Must match LLMModel enum
            "cohort_size": 10,  # Minimum allowed by backend validation
            "sampling_strategy": "stratified",
        },
    )
    response.raise_for_status()
    execution = response.json()
    pretty_print(execution, "✅ Execution Started")
    return execution["task_id"]


def monitor_task(survey_id, task_id):
    """Monitor task progress"""
    print(f"\n⏳ Monitoring task {task_id}...")

    while True:
        response = requests.get(
            f"{BASE_URL}/api/v1/surveys/{survey_id}/tasks/{task_id}/status"
        )
        response.raise_for_status()
        status = response.json()

        progress_pct = int(status["progress"] * 100)
        print(
            f"  Status: {status['status']} | Progress: {progress_pct}% | Responses: {status['responses_generated']}"
        )

        if status["status"] == "completed":
            pretty_print(status, "✅ Task Completed")
            return status
        elif status["status"] == "failed":
            pretty_print(status, "❌ Task Failed")
            return status

        time.sleep(2)


def get_results(survey_id):
    """Get survey results"""
    print("\n📊 Fetching results...")
    response = requests.get(f"{BASE_URL}/api/v1/surveys/{survey_id}/results")
    response.raise_for_status()
    results = response.json()
    pretty_print(results, "✅ Survey Results")
    return results


def main():
    """Run integration test"""
    print("\n" + "=" * 60)
    print("🧪 SSR INTEGRATION TEST")
    print("=" * 60)

    try:
        # Step 1: Create survey
        survey_id = create_survey()

        # Step 2: Execute survey
        task_id = execute_survey(survey_id)

        # Step 3: Monitor progress
        final_status = monitor_task(survey_id, task_id)

        # Step 4: Get results if successful
        if final_status["status"] == "completed":
            results = get_results(survey_id)

            print("\n" + "=" * 60)
            print("📈 KEY METRICS")
            print("=" * 60)
            print(f"Mean Rating: {results['metrics']['mean_rating']:.2f} / 5.0")
            print(f"Confidence: {results['metrics']['confidence']:.2%}")
            print(f"Std Dev: {results['metrics']['std_rating']:.3f}")
            print(f"Cohort Size: {results['cohort_size']}")
            print(
                f"Execution Time: {results['quality']['execution_time_seconds']:.1f}s"
            )
            print(f"Model Used: {results['quality']['llm_model_used']}")

            print("\n📊 Rating Distribution:")
            for rating, prob in enumerate(results["distribution"], 1):
                bar = "█" * int(prob * 50)
                print(f"  {rating} ⭐: {prob:.2%} {bar}")

        print("\n" + "=" * 60)
        print("✅ INTEGRATION TEST PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
