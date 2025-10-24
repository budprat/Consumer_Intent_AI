import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";

export function useSurveys() {
  return useQuery({
    queryKey: ["surveys"],
    queryFn: () => apiClient.surveys.list(),
  });
}
