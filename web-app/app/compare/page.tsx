// ABOUTME: Compare surveys page - A/B testing comparison interface for multiple surveys
// ABOUTME: Displays side-by-side SSR ratings, distribution charts, and detailed metrics table
"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useSurveys } from "@/hooks/use-surveys";
import { useSurveyResults } from "@/hooks/use-survey-results";
import { PageHeader } from "@/components/shared/page-header";
import { LoadingSpinner } from "@/components/shared/loading-spinner";
import { SSRRatingBadge } from "@/components/surveys/ssr-rating-badge";
import { DistributionChart } from "@/components/surveys/distribution-chart";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ArrowLeft, Plus, X } from "lucide-react";
import type { Survey, SurveyResults } from "@/lib/types";

export default function CompareSurveysPage() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <CompareContent />
    </Suspense>
  );
}

function CompareContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const { data: surveys, isLoading: surveysLoading } = useSurveys();

  // Parse IDs from URL on mount
  useEffect(() => {
    const idsParam = searchParams.get("ids");
    if (idsParam) {
      setSelectedIds(idsParam.split(","));
    }
  }, [searchParams]);

  // Update URL when selection changes
  const updateSelection = (ids: string[]) => {
    setSelectedIds(ids);
    const url = ids.length > 0 ? `/compare?ids=${ids.join(",")}` : "/compare";
    router.replace(url);
  };

  const addSurvey = (surveyId: string) => {
    if (!selectedIds.includes(surveyId)) {
      updateSelection([...selectedIds, surveyId]);
    }
  };

  const removeSurvey = (surveyId: string) => {
    updateSelection(selectedIds.filter((id) => id !== surveyId));
  };

  const completedSurveys =
    surveys?.filter((s) => s.status === "completed") || [];
  const availableToAdd = completedSurveys.filter(
    (s) => !selectedIds.includes(s.id),
  );

  const selectedSurveys = completedSurveys.filter((s) =>
    selectedIds.includes(s.id),
  );

  if (surveysLoading) {
    return <LoadingSpinner />;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="sm" onClick={() => router.push("/")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Dashboard
        </Button>
      </div>

      <PageHeader
        title="Compare Surveys"
        description="Side-by-side comparison of purchase intent ratings"
      />

      {/* Survey Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Select Surveys to Compare</CardTitle>
          <CardDescription>
            Choose 2-4 completed surveys for comparison
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            {availableToAdd.length > 0 ? (
              <Select onValueChange={addSurvey}>
                <SelectTrigger className="w-[300px]">
                  <SelectValue placeholder="Add survey..." />
                </SelectTrigger>
                <SelectContent>
                  {availableToAdd.map((survey) => (
                    <SelectItem key={survey.survey_id} value={survey.survey_id}>
                      {survey.product_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <p className="text-sm text-slate-500">
                {selectedIds.length > 0
                  ? "All available surveys selected"
                  : "No completed surveys available"}
              </p>
            )}
          </div>

          {selectedSurveys.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2">
              {selectedSurveys.map((survey) => (
                <div
                  key={survey.survey_id}
                  className="flex items-center gap-2 bg-slate-100 px-3 py-2 rounded-md"
                >
                  <span className="text-sm font-medium">
                    {survey.product_name}
                  </span>
                  <button
                    onClick={() => removeSurvey(survey.survey_id)}
                    className="text-slate-500 hover:text-slate-700"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Empty State */}
      {selectedSurveys.length === 0 && (
        <Card>
          <CardContent className="py-12 text-center">
            <div className="flex flex-col items-center gap-4">
              <Plus className="h-12 w-12 text-slate-300" />
              <div>
                <h3 className="text-lg font-semibold">No Surveys Selected</h3>
                <p className="text-sm text-slate-500 mt-1">
                  Select at least 2 completed surveys to compare their results
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Comparison View */}
      {selectedSurveys.length >= 2 && (
        <>
          {/* SSR Ratings Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>Purchase Intent Ratings</CardTitle>
              <CardDescription>
                Semantic Similarity Rating (SSR) comparison
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {selectedSurveys.map((survey) => (
                  <SurveyComparisonCard key={survey.survey_id} survey={survey} />
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Distribution Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>Rating Distributions</CardTitle>
              <CardDescription>
                Probability distributions across rating levels
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-8">
                {selectedSurveys.map((survey) => (
                  <div key={survey.survey_id} className="space-y-2">
                    <h4 className="font-medium">{survey.product_name}</h4>
                    <SurveyDistributionView surveyId={survey.survey_id} />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Comparison Table */}
          <Card>
            <CardHeader>
              <CardTitle>Detailed Metrics</CardTitle>
              <CardDescription>
                Side-by-side comparison of key statistics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-4 font-medium">
                        Metric
                      </th>
                      {selectedSurveys.map((survey) => (
                        <th
                          key={survey.survey_id}
                          className="text-left py-2 px-4 font-medium"
                        >
                          {survey.product_name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {selectedSurveys.map((_, index) =>
                      index === 0 ? (
                        <ComparisonTableRows
                          key="rows"
                          surveys={selectedSurveys}
                        />
                      ) : null,
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

// Helper component for survey rating card
function SurveyComparisonCard({ survey }: { survey: Survey }) {
  const { data: results, isLoading } = useSurveyResults(survey.survey_id, true);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8 flex justify-center">
          <LoadingSpinner />
        </CardContent>
      </Card>
    );
  }

  if (!results) {
    return (
      <Card>
        <CardContent className="py-8">
          <p className="text-sm text-slate-500 text-center">
            No results available
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{survey.product_name}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <SSRRatingBadge rating={results.rating} size="large" />
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-500">Confidence:</span>
            <span className="font-medium">
              {(results.confidence * 100).toFixed(1)}%
            </span>
          </div>
          {results.mean_rating && (
            <div className="flex justify-between">
              <span className="text-slate-500">Mean:</span>
              <span className="font-medium">
                {results.mean_rating.toFixed(2)}
              </span>
            </div>
          )}
          <div className="flex justify-between">
            <span className="text-slate-500">Cohort Size:</span>
            <span className="font-medium">{survey.cohort_size}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Helper component for distribution view
function SurveyDistributionView({ surveyId }: { surveyId: string }) {
  const { data: results, isLoading } = useSurveyResults(surveyId, true);

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (!results) {
    return <p className="text-sm text-slate-500">No results available</p>;
  }

  return (
    <DistributionChart
      distribution={results.distribution}
      meanRating={results.mean_rating}
    />
  );
}

// Helper component for single comparison row
function ComparisonTableRow({
  surveys,
  results,
}: {
  surveys: Survey[];
  results: (SurveyResults | undefined)[];
}) {
  const allResults = results.filter(Boolean) as SurveyResults[];

  if (allResults.length !== surveys.length) {
    return (
      <tr>
        <td
          colSpan={surveys.length + 1}
          className="py-4 text-center text-slate-500"
        >
          Loading comparison data...
        </td>
      </tr>
    );
  }

  const metrics = [
    {
      label: "SSR Rating",
      getValue: (r: SurveyResults) => r.rating.toFixed(0),
    },
    {
      label: "Confidence",
      getValue: (r: SurveyResults) => `${(r.confidence * 100).toFixed(1)}%`,
    },
    {
      label: "Mean Rating",
      getValue: (r: SurveyResults) => r.mean_rating?.toFixed(2) || "N/A",
    },
    {
      label: "Std Dev",
      getValue: (r: SurveyResults) => r.std_rating?.toFixed(2) || "N/A",
    },
  ];

  return (
    <>
      {metrics.map((metric, index) => (
        <tr key={index} className="border-b">
          <td className="py-3 px-4 text-slate-600">{metric.label}</td>
          {allResults.map((result, i) => (
            <td key={i} className="py-3 px-4 font-medium">
              {metric.getValue(result)}
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

// Wrapper component that uses hooks at the correct level
function ComparisonTableRows({ surveys }: { surveys: Survey[] }) {
  // Call hooks at the component level, not in a map
  const survey1Results = useSurveyResults(surveys[0]?.id, true);
  const survey2Results = useSurveyResults(surveys[1]?.id, true);
  const survey3Results = useSurveyResults(surveys[2]?.id, true);
  const survey4Results = useSurveyResults(surveys[3]?.id, true);

  const results = [
    survey1Results.data,
    survey2Results.data,
    survey3Results.data,
    survey4Results.data,
  ].slice(0, surveys.length);

  return <ComparisonTableRow surveys={surveys} results={results} />;
}
