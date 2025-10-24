import { Badge } from "@/components/ui/badge";
import { Loader2, CheckCircle2, XCircle, Clock } from "lucide-react";

interface SurveyStatusProps {
  status: "pending" | "running" | "completed" | "failed";
}

export function SurveyStatus({ status }: SurveyStatusProps) {
  const configs = {
    pending: {
      icon: Clock,
      label: "Pending",
      className: "bg-slate-100 text-slate-800 border-slate-300",
      animate: false,
    },
    running: {
      icon: Loader2,
      label: "Running",
      className: "bg-blue-100 text-blue-800 border-blue-300",
      animate: true,
    },
    completed: {
      icon: CheckCircle2,
      label: "Completed",
      className: "bg-green-100 text-green-800 border-green-300",
      animate: false,
    },
    failed: {
      icon: XCircle,
      label: "Failed",
      className: "bg-red-100 text-red-800 border-red-300",
      animate: false,
    },
  };

  const config = configs[status];
  const Icon = config.icon;

  return (
    <Badge variant="outline" className={config.className}>
      <Icon
        className={`mr-1 h-3 w-3 ${config.animate ? "animate-spin" : ""}`}
      />
      {config.label}
    </Badge>
  );
}
