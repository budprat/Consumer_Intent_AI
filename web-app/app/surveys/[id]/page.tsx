"use client";

import { use, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { apiClient } from "../../../lib/api";
import { Survey, TaskStatus, SurveyResults } from "../../../lib/types";
import { ExecutePanel } from "../../../components/surveys/execute-panel";
import { ResultsDisplay } from "../../../components/surveys/results-display";
import { ArrowLeft } from "lucide-react";

export default function SurveyDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: surveyId } = use(params);
  const router = useRouter();

  const [survey, setSurvey] = useState<Survey | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Task polling state
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null);
  const [results, setResults] = useState<SurveyResults | null>(null);

  // Load survey on mount
  useEffect(() => {
    loadSurvey();
  }, [surveyId]);

  // Poll task status when execution is running
  useEffect(() => {
    if (!currentTaskId) return;

    const pollInterval = setInterval(async () => {
      try {
        const status = await apiClient.tasks.getStatus(surveyId, currentTaskId);
        setTaskStatus(status);

        if (status.status === "completed") {
          clearInterval(pollInterval);
          await loadResults();
        } else if (status.status === "failed") {
          clearInterval(pollInterval);
          setError(status.error || "Task failed");
        }
      } catch (err: any) {
        console.error("Failed to poll task status:", err);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [currentTaskId, surveyId]);

  const loadSurvey = async () => {
    try {
      setLoading(true);
      setError(null);
      const surveyData = await apiClient.surveys.get(surveyId);
      setSurvey(surveyData);
    } catch (err: any) {
      setError(err.message || "Failed to load survey");
    } finally {
      setLoading(false);
    }
  };

  const loadResults = async () => {
    try {
      const resultsData = await apiClient.surveys.getResults(surveyId);
      setResults(resultsData);
    } catch (err: any) {
      console.error("Failed to load results:", err);
      setError(err.message || "Failed to load results");
    }
  };

  const handleExecutionStart = (taskId: string) => {
    setCurrentTaskId(taskId);
    setTaskStatus({
      task_id: taskId,
      survey_id: surveyId,
      status: "pending",
      progress: 0,
      responses_generated: 0,
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error && !survey) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-red-900">Error</h2>
          <p className="text-red-700 mt-2">{error}</p>
          <button
            onClick={() => router.push("/")}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  if (!survey) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12">
        <p className="text-gray-600">Survey not found</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Back Button */}
      <button
        onClick={() => router.push("/")}
        className="flex items-center text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Dashboard
      </button>

      {/* Survey Header */}
      <div className="bg-white shadow rounded-lg p-6 mb-6">
        <h1 className="text-3xl font-bold text-gray-900">
          {survey.product_name}
        </h1>
        <div className="mt-4 flex items-center gap-6 text-sm text-gray-600">
          <div>
            <span className="font-medium">Status:</span>{" "}
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                survey.status === "pending"
                  ? "bg-yellow-100 text-yellow-800"
                  : survey.status === "running"
                    ? "bg-blue-100 text-blue-800"
                    : survey.status === "completed"
                      ? "bg-green-100 text-green-800"
                      : "bg-red-100 text-red-800"
              }`}
            >
              {survey.status}
            </span>
          </div>
          <div>
            <span className="font-medium">Created:</span>{" "}
            {new Date(survey.created_at).toLocaleDateString()}
          </div>
        </div>
      </div>

      {/* Content based on status */}
      {survey.status === "pending" && !currentTaskId && (
        <ExecutePanel
          surveyId={surveyId}
          onExecutionStart={handleExecutionStart}
        />
      )}

      {taskStatus &&
        (taskStatus.status === "pending" ||
          taskStatus.status === "running") && (
          <div className="bg-white shadow rounded-lg p-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-6">
              Survey Execution in Progress
            </h2>

            {/* Progress Bar */}
            <div className="mb-6">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Progress</span>
                <span>{Math.round(taskStatus.progress * 100)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                <div
                  className="bg-indigo-600 h-4 transition-all duration-500 ease-out"
                  style={{ width: `${taskStatus.progress * 100}%` }}
                />
              </div>
            </div>

            {/* Status Info */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-600">Status</p>
                <p className="text-lg font-medium text-gray-900 capitalize">
                  {taskStatus.status}
                </p>
              </div>
              <div>
                <p className="text-gray-600">Responses Generated</p>
                <p className="text-lg font-medium text-gray-900">
                  {taskStatus.responses_generated}
                </p>
              </div>
            </div>

            {/* Loading Animation */}
            <div className="mt-6 flex items-center justify-center">
              <div className="flex items-center space-x-2 text-gray-600">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-600"></div>
                <span>Generating synthetic consumer responses...</span>
              </div>
            </div>
          </div>
        )}

      {results && <ResultsDisplay results={results} />}

      {error && taskStatus?.status === "failed" && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-red-900">
            Execution Failed
          </h2>
          <p className="text-red-700 mt-2">{error}</p>
          <button
            onClick={() => {
              setError(null);
              setCurrentTaskId(null);
              setTaskStatus(null);
            }}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}
