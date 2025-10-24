import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SSRRatingBadge } from "./ssr-rating-badge";
import { SurveyStatus } from "./survey-status";
import { Survey } from "@/lib/types";
import { ArrowRight } from "lucide-react";

interface SurveyCardProps {
  survey: Survey;
  rating?: number;
}

export function SurveyCard({ survey, rating }: SurveyCardProps) {
  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-xl">{survey.product_name}</CardTitle>
            <p className="text-sm text-slate-500 mt-1">
              {formatDistanceToNow(new Date(survey.created_at), {
                addSuffix: true,
              })}
            </p>
          </div>
          <SurveyStatus status={survey.status} />
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-slate-600 line-clamp-2 mb-4">
          {survey.product_description}
        </p>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {rating && <SSRRatingBadge rating={rating} />}
            <span className="text-sm text-slate-500">
              {survey.cohort_size} responses
            </span>
          </div>

          <Button variant="ghost" size="sm" asChild>
            <Link href={`/surveys/${survey.survey_id}`}>
              View Details
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
