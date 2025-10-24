import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";

export function useSurvey(surveyId: string) {
  return useQuery({
    queryKey: ["survey", surveyId],
    queryFn: () => apiClient.surveys.get(surveyId),
    enabled: !!surveyId,
    refetchInterval: (query) => {
      // Poll every 3 seconds if survey is running, stop if completed/failed
      const status = query.state.data?.status;
      return status === "running" ? 3000 : false;
    },
  });
}
