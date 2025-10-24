"use client";

import { SurveyResults } from "../../lib/types";

interface ResultsDisplayProps {
  results: SurveyResults;
}

export function ResultsDisplay({ results }: ResultsDisplayProps) {
  // Calculate percentages for distribution
  const distributionPercentages = results.distribution.map((prob: number) =>
    (prob * 100).toFixed(1),
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Survey Results
        </h2>
        <p className="text-sm text-gray-500">
          Completed on {new Date(results.completed_at).toLocaleString()}
        </p>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Mean Rating */}
        <div className="bg-white shadow rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Mean Rating</p>
              <p className="text-3xl font-bold text-indigo-600 mt-2">
                {results.metrics.mean_rating.toFixed(2)}
                <span className="text-lg text-gray-500"> / 5.0</span>
              </p>
            </div>
            <div className="h-12 w-12 bg-indigo-100 rounded-full flex items-center justify-center">
              <svg
                className="h-6 w-6 text-indigo-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
                />
              </svg>
            </div>
          </div>
        </div>

        {/* Confidence */}
        <div className="bg-white shadow rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Confidence</p>
              <p className="text-3xl font-bold text-green-600 mt-2">
                {(results.metrics.confidence * 100).toFixed(2)}%
              </p>
            </div>
            <div className="h-12 w-12 bg-green-100 rounded-full flex items-center justify-center">
              <svg
                className="h-6 w-6 text-green-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
          </div>
        </div>

        {/* Cohort Size */}
        <div className="bg-white shadow rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Cohort Size</p>
              <p className="text-3xl font-bold text-purple-600 mt-2">
                {results.cohort_size}
              </p>
              <p className="text-xs text-gray-500 mt-1">synthetic consumers</p>
            </div>
            <div className="h-12 w-12 bg-purple-100 rounded-full flex items-center justify-center">
              <svg
                className="h-6 w-6 text-purple-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Rating Distribution */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Purchase Intent Distribution
        </h3>
        <div className="space-y-3">
          {results.distribution.map((prob: number, index: number) => {
            const rating = index + 1;
            const percentage = prob * 100;
            const barWidth = `${percentage}%`;

            return (
              <div key={rating} className="flex items-center">
                <div className="w-16 flex items-center">
                  <span className="text-sm font-medium text-gray-700">
                    {rating}
                  </span>
                  <span className="ml-1 text-yellow-400">⭐</span>
                </div>
                <div className="flex-1 ml-4">
                  <div className="relative">
                    <div className="overflow-hidden h-8 text-xs flex rounded bg-gray-200">
                      <div
                        style={{ width: barWidth }}
                        className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-indigo-500 transition-all duration-500"
                      >
                        {percentage >= 10 && (
                          <span className="font-semibold">
                            {distributionPercentages[index]}%
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
                <div className="w-16 ml-4 text-right">
                  <span className="text-sm font-medium text-gray-700">
                    {distributionPercentages[index]}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Demographics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Age Distribution */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Age Groups
          </h3>
          <div className="space-y-2">
            {Object.entries(results.demographic_distribution.age_groups)
              .sort((a, b) => (b[1] as number) - (a[1] as number))
              .map(([ageGroup, percentage]) => (
                <div key={ageGroup} className="flex justify-between">
                  <span className="text-sm text-gray-600">{ageGroup}</span>
                  <span className="text-sm font-medium text-gray-900">
                    {((percentage as number) * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
          </div>
        </div>

        {/* Gender Distribution */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Gender</h3>
          <div className="space-y-2">
            {Object.entries(results.demographic_distribution.gender)
              .sort((a, b) => (b[1] as number) - (a[1] as number))
              .map(([gender, percentage]) => (
                <div key={gender} className="flex justify-between">
                  <span className="text-sm text-gray-600">{gender}</span>
                  <span className="text-sm font-medium text-gray-900">
                    {((percentage as number) * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
          </div>
        </div>

        {/* Income Distribution */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Income Levels
          </h3>
          <div className="space-y-2">
            {Object.entries(results.demographic_distribution.income)
              .sort((a, b) => (b[1] as number) - (a[1] as number))
              .map(([income, percentage]) => (
                <div key={income} className="flex justify-between">
                  <span className="text-sm text-gray-600">{income}</span>
                  <span className="text-sm font-medium text-gray-900">
                    {((percentage as number) * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Quality Metrics */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Quality Metrics
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-gray-600">Execution Time</p>
            <p className="text-xl font-semibold text-gray-900 mt-1">
              {results.quality.execution_time_seconds.toFixed(1)}s
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">LLM Model</p>
            <p className="text-xl font-semibold text-gray-900 mt-1">
              {results.quality.llm_model_used}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Demographic Representation</p>
            <p className="text-xl font-semibold text-gray-900 mt-1 capitalize">
              {results.quality.demographic_representation}
            </p>
          </div>
        </div>

        {/* Additional Stats */}
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Standard Deviation</p>
              <p className="text-lg font-medium text-gray-900 mt-1">
                {results.metrics.std_rating.toFixed(3)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Task ID</p>
              <p className="text-lg font-mono text-gray-700 mt-1 text-xs">
                {results.task_id}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
