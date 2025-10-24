"use client";

import { useState } from "react";
import { apiClient } from "../../lib/api";
import { ExecuteSurveyResponse } from "../../lib/types";

interface ExecutePanelProps {
  surveyId: string;
  onExecutionStart: (taskId: string) => void;
}

export function ExecutePanel({
  surveyId,
  onExecutionStart,
}: ExecutePanelProps) {
  const [llmModel, setLlmModel] = useState<"gpt-4o" | "gemini-2.0-flash">(
    "gpt-4o",
  );
  const [cohortSize, setCohortSize] = useState(10);
  const [isExecuting, setIsExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExecute = async () => {
    setIsExecuting(true);
    setError(null);

    try {
      const response: ExecuteSurveyResponse = await apiClient.execute(
        surveyId,
        {
          llm_model: llmModel,
          cohort_size: cohortSize,
          sampling_strategy: "stratified",
          async_execution: true,
        },
      );

      onExecutionStart(response.task_id);
    } catch (err: any) {
      setError(err.message || "Failed to execute survey");
      setIsExecuting(false);
    }
  };

  const estimatedTime = cohortSize * 3; // ~3 seconds per consumer

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h2 className="text-lg font-semibold mb-4">Execute Survey</h2>

      <div className="space-y-4">
        {/* LLM Model Selector */}
        <div>
          <label
            htmlFor="llm-model"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            LLM Model
          </label>
          <select
            id="llm-model"
            value={llmModel}
            onChange={(e) =>
              setLlmModel(e.target.value as "gpt-4o" | "gemini-2.0-flash")
            }
            disabled={isExecuting}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            <option value="gpt-4o">GPT-4o (Recommended)</option>
            <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
          </select>
        </div>

        {/* Cohort Size Input */}
        <div>
          <label
            htmlFor="cohort-size"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Cohort Size (Number of Synthetic Consumers)
          </label>
          <input
            id="cohort-size"
            type="number"
            min="10"
            max="100"
            value={cohortSize}
            onChange={(e) => setCohortSize(parseInt(e.target.value, 10))}
            disabled={isExecuting}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
          <p className="mt-1 text-xs text-gray-500">
            Recommended: 10-100 consumers. More consumers = higher accuracy but
            longer execution time (minimum: 10).
          </p>
        </div>

        {/* Info Box */}
        <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-blue-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3 flex-1">
              <p className="text-sm text-blue-700">
                <strong>What happens during execution:</strong>
              </p>
              <ul className="mt-2 text-sm text-blue-700 list-disc list-inside space-y-1">
                <li>
                  Generate {cohortSize} synthetic consumers with diverse
                  demographics
                </li>
                <li>
                  Each consumer provides purchase intent response using{" "}
                  {llmModel}
                </li>
                <li>
                  SSR algorithm calculates rating distribution (1-5 scale)
                </li>
                <li>Results aggregated with confidence scores</li>
              </ul>
              <p className="mt-2 text-xs text-blue-600">
                Estimated time: ~{estimatedTime} seconds
              </p>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg
                  className="h-5 w-5 text-red-400"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Execute Button */}
        <button
          onClick={handleExecute}
          disabled={isExecuting}
          className={`w-full px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
            isExecuting
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          }`}
        >
          {isExecuting ? (
            <span className="flex items-center justify-center">
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Starting Execution...
            </span>
          ) : (
            "Execute Survey"
          )}
        </button>
      </div>
    </div>
  );
}
