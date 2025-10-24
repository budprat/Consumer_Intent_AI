import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";

export function useSurveyResults(surveyId: string, enabled: boolean = true) {
  return useQuery({
    queryKey: ["survey-results", surveyId],
    queryFn: () => apiClient.surveys.getResults(surveyId),
    enabled: enabled && !!surveyId,
    retry: false, // Don't retry if results aren't ready yet
  });
}
