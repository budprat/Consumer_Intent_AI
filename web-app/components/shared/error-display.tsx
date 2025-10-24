import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ApiError } from "@/lib/types";

interface ErrorDisplayProps {
  error: Error | ApiError | unknown;
}

export function ErrorDisplay({ error }: ErrorDisplayProps) {
  const message =
    error instanceof Error ? error.message : "An unexpected error occurred";

  return (
    <div className="flex items-center justify-center min-h-[400px] p-4">
      <Alert variant="destructive" className="max-w-md">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{message}</AlertDescription>
      </Alert>
    </div>
  );
}
