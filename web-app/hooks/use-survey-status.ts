import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";

export function useSurveyStatus(taskId: string | null) {
  return useQuery({
    queryKey: ["task-status", taskId],
    queryFn: () => apiClient.tasks.getStatus(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      // Poll every 3 seconds if running, stop if completed/failed
      return query.state.data?.status === "running" ? 3000 : false;
    },
  });
}
