"use client";

import { useSurveys } from "@/hooks/use-surveys";
import { SurveyCard } from "@/components/surveys/survey-card";
import { Button } from "@/components/ui/button";
import { Plus, Loader2, AlertCircle } from "lucide-react";
import Link from "next/link";
import { PageHeader } from "@/components/shared/page-header";

export default function DashboardPage() {
  const { data: surveys, isLoading, error } = useSurveys();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin text-slate-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
        <AlertCircle className="h-12 w-12 text-red-500" />
        <div className="text-center">
          <h2 className="text-lg font-semibold">Failed to load surveys</h2>
          <p className="text-sm text-slate-500 mt-1">
            {error instanceof Error ? error.message : "An error occurred"}
          </p>
        </div>
      </div>
    );
  }

  const surveysWithResults =
    surveys?.filter((s) => s.status === "completed") || [];
  const surveysInProgress =
    surveys?.filter((s) => s.status === "running") || [];
  const surveysPending = surveys?.filter((s) => s.status === "pending") || [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <PageHeader
          title="Purchase Intent Surveys"
          description="View and manage your SSR surveys"
        />
        <Button asChild>
          <Link href="/surveys/new">
            <Plus className="mr-2 h-4 w-4" />
            New Survey
          </Link>
        </Button>
      </div>

      {!surveys || surveys.length === 0 ? (
        <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
          <div className="text-center">
            <h2 className="text-lg font-semibold">No surveys yet</h2>
            <p className="text-sm text-slate-500 mt-1">
              Create your first survey to get started
            </p>
          </div>
          <Button asChild>
            <Link href="/surveys/new">
              <Plus className="mr-2 h-4 w-4" />
              Create Survey
            </Link>
          </Button>
        </div>
      ) : (
        <div className="space-y-8">
          {surveysInProgress.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">In Progress</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {surveysInProgress.map((survey) => (
                  <SurveyCard key={survey.survey_id} survey={survey} />
                ))}
              </div>
            </div>
          )}

          {surveysWithResults.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Completed</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {surveysWithResults.map((survey) => (
                  <SurveyCard key={survey.survey_id} survey={survey} />
                ))}
              </div>
            </div>
          )}

          {surveysPending.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Pending</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {surveysPending.map((survey) => (
                  <SurveyCard key={survey.survey_id} survey={survey} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
